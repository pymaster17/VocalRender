"""
Multi-GPU data-parallel inference pool for VoxCPM.

Uses a work-stealing pattern: each GPU runs in its own thread and pulls
batch tasks from a shared queue.  Whichever GPU finishes first takes the
next batch, naturally handling uneven batch durations.

Python threads release the GIL during CUDA kernel launches and
``torch.no_grad()`` blocks, so GPU compute across threads is truly parallel.

Usage::

    pool = MultiGPUInferencePool(
        load_model_fn=lambda dev: load_svs_model(ckpt_dir, device=dev),
        devices=["cuda:0", "cuda:1"],
        batch_size=64,
        generate_kwargs=dict(cfg_value=2.0, inference_timesteps=10, max_len=2000),
    )
    # prompts: list of str
    results = pool.generate_all(prompts)   # list of (global_task_idx, audio_tensor)
    pool.shutdown()
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Sentinel & data types
# ---------------------------------------------------------------------------

_SHUTDOWN = object()


@dataclass
class _BatchTask:
    """A batch of prompts to be generated."""
    batch_id: int
    global_indices: List[int]          # position in the original prompt list
    prompts: List[str]
    prompt_audio_feats: Optional[List[Optional[torch.Tensor]]] = None


@dataclass
class _BatchResult:
    """Results returned by one batch generation call."""
    batch_id: int
    global_indices: List[int]
    audio_tensors: List[Optional[torch.Tensor]]
    device: str
    elapsed: float
    latent_tensors: Optional[List[Optional[torch.Tensor]]] = None


# ---------------------------------------------------------------------------
# GPU Worker
# ---------------------------------------------------------------------------

class _GPUWorker:
    """Runs on a dedicated thread, owns one model on one device."""

    def __init__(
        self,
        device: str,
        model: Any,
        generate_kwargs: Dict[str, Any],
        task_queue: "queue.Queue[_BatchTask | object]",
        result_list: List[_BatchResult],
        result_lock: threading.Lock,
        log_fn: Callable[[str], None],
    ):
        self.device = device
        self.model = model
        self.generate_kwargs = generate_kwargs
        self.task_queue = task_queue
        self.result_list = result_list
        self.result_lock = result_lock
        self.log_fn = log_fn

    def run(self) -> None:
        """Main loop: pull tasks until shutdown sentinel."""
        while True:
            task = self.task_queue.get()
            if task is _SHUTDOWN:
                self.task_queue.task_done()
                break

            assert isinstance(task, _BatchTask)
            t0 = time.time()
            latent_tensors: Optional[List[Optional[torch.Tensor]]] = None
            want_latents = bool(self.generate_kwargs.get("return_latents", False))
            try:
                with torch.no_grad():
                    out = self.model.generate_batch(
                        target_texts=task.prompts,
                        prompt_audio_feats=task.prompt_audio_feats,
                        **self.generate_kwargs,
                    )
                if want_latents:
                    audio_tensors, latent_tensors = out
                else:
                    audio_tensors = out
            except Exception as exc:
                self.log_fn(
                    f"  [{self.device}] Batch {task.batch_id} error: {exc}"
                )
                import traceback
                traceback.print_exc()
                audio_tensors = [None] * len(task.prompts)
                if want_latents:
                    latent_tensors = [None] * len(task.prompts)

            elapsed = time.time() - t0
            result = _BatchResult(
                batch_id=task.batch_id,
                global_indices=task.global_indices,
                audio_tensors=audio_tensors,
                device=self.device,
                elapsed=elapsed,
                latent_tensors=latent_tensors,
            )
            with self.result_lock:
                self.result_list.append(result)

            self.log_fn(
                f"  [{self.device}] Batch {task.batch_id} done "
                f"({len(task.prompts)} prompts, {elapsed:.1f}s)"
            )
            self.task_queue.task_done()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MultiGPUInferencePool:
    """Work-stealing thread pool for multi-GPU batch inference.

    Parameters
    ----------
    load_model_fn : callable(device_str) -> model
        Factory that loads a VoxCPM model onto the given device string.
    devices : list[str]
        CUDA device strings, e.g. ``["cuda:0", "cuda:1"]``.
    batch_size : int
        Per-GPU batch size.
    generate_kwargs : dict
        Keyword arguments forwarded to ``model.generate_batch()``,
        e.g. ``cfg_value``, ``inference_timesteps``, ``max_len``.
    log_fn : callable, optional
        Logging function.  Defaults to ``print(..., file=sys.stderr)``.
    """

    def __init__(
        self,
        load_model_fn: Callable[[str], Any],
        devices: List[str],
        batch_size: int = 64,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        self.devices = devices
        self.batch_size = batch_size
        self.generate_kwargs = generate_kwargs or {}
        self.log_fn = log_fn or (lambda msg: print(msg, file=sys.stderr))
        self._shut_down = False

        if len(devices) > 1:
            # This pool runs one Python worker thread per GPU. Keep PyTorch's
            # CPU thread pools single-threaded so we don't multiply host-side
            # launcher / BLAS threads across workers and starve CUDA dispatch.
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            try:
                if torch.get_num_threads() != 1:
                    torch.set_num_threads(1)
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # Can only be set once and before inter-op work starts.
                pass

        # ---- Load one model per GPU ----
        self.log_fn(
            f"[MultiGPU] Loading model on {len(devices)} device(s): {devices}"
        )
        self.models: List[Any] = []
        for dev in devices:
            t0 = time.time()
            model = load_model_fn(dev)
            elapsed = time.time() - t0
            self.models.append(model)
            self.log_fn(f"[MultiGPU] Model loaded on {dev} ({elapsed:.1f}s)")

        # ---- Shared work queue + result collection ----
        self._task_queue: queue.Queue = queue.Queue()
        self._result_list: List[_BatchResult] = []
        self._result_lock = threading.Lock()

        # ---- Spawn worker threads (one per GPU) ----
        self._workers: List[threading.Thread] = []
        self._gpu_workers: List[_GPUWorker] = []
        for dev, model in zip(devices, self.models):
            worker = _GPUWorker(
                device=dev,
                model=model,
                generate_kwargs=self.generate_kwargs,
                task_queue=self._task_queue,
                result_list=self._result_list,
                result_lock=self._result_lock,
                log_fn=self.log_fn,
            )
            thread = threading.Thread(
                target=worker.run, name=f"gpu-worker-{dev}", daemon=True
            )
            thread.start()
            self._workers.append(thread)
            self._gpu_workers.append(worker)

        self.log_fn(f"[MultiGPU] {len(self._workers)} worker thread(s) started")

    # ------------------------------------------------------------------
    # Public: generate_all
    # ------------------------------------------------------------------

    def generate_all(
        self,
        prompts: List[str],
        prompt_audio_feats_list: Optional[List[Optional[torch.Tensor]]] = None,
        sort_by: Optional[Callable[[str], float]] = len,
    ) -> List[Tuple[int, Optional[torch.Tensor]]]:
        """Generate audio for all *prompts* across GPUs.

        Prompts are split into batches of ``self.batch_size`` and enqueued.
        Each GPU thread pulls batches from the shared queue as it becomes
        idle (work-stealing).

        When ``sort_by`` is provided (default: ``len``), prompts are first
        ordered by the key so that each batch contains samples of similar
        expected output length. This minimises the wall-clock penalty of
        ``_inference_batch`` waiting for the longest sample in a heterogeneous
        batch (the AR loop only breaks when ALL samples have stopped).
        Pass ``sort_by=None`` to preserve input order.

        Returns
        -------
        list of (global_index, audio_tensor | None)
            Ordered by ``global_index`` (the position in the input list).
        """
        if self._shut_down:
            raise RuntimeError("Pool has been shut down")

        # Clear previous results
        with self._result_lock:
            self._result_list.clear()

        # Build dispatch order. Even if sort_by is None, we still keep a
        # uniform `order` list so batch construction stays index-based.
        if sort_by is not None and len(prompts) > 1:
            order = sorted(range(len(prompts)), key=lambda i: sort_by(prompts[i]))
        else:
            order = list(range(len(prompts)))

        # Enqueue batches over the (possibly reordered) sequence, but keep
        # ``global_indices`` pointing at the ORIGINAL position so callers
        # always receive results in input order.
        num_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        for bi in range(num_batches):
            start = bi * self.batch_size
            end = min(start + self.batch_size, len(prompts))
            batch_order = order[start:end]
            batch_pa = None
            if prompt_audio_feats_list is not None:
                batch_pa = [prompt_audio_feats_list[i] for i in batch_order]
            task = _BatchTask(
                batch_id=bi,
                global_indices=batch_order,
                prompts=[prompts[i] for i in batch_order],
                prompt_audio_feats=batch_pa,
            )
            self._task_queue.put(task)

        self.log_fn(
            f"[MultiGPU] Enqueued {num_batches} batch(es) "
            f"({len(prompts)} prompts, batch_size={self.batch_size})"
        )

        # Wait for all batches to complete
        self._task_queue.join()

        # Collect results. When ``return_latents`` was set in
        # ``generate_kwargs``, each entry is ``(gidx, audio, latent)``;
        # otherwise ``(gidx, audio)`` for backwards compatibility.
        want_latents = bool(self.generate_kwargs.get("return_latents", False))
        with self._result_lock:
            if want_latents:
                all_results_lat: List[Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]] = []
                for br in self._result_list:
                    lat_list = br.latent_tensors or [None] * len(br.global_indices)
                    for j, gidx in enumerate(br.global_indices):
                        a = br.audio_tensors[j] if j < len(br.audio_tensors) else None
                        l = lat_list[j] if j < len(lat_list) else None
                        all_results_lat.append((gidx, a, l))
                all_results_lat.sort(key=lambda x: x[0])
                return all_results_lat

            all_results: List[Tuple[int, Optional[torch.Tensor]]] = []
            for br in self._result_list:
                for j, gidx in enumerate(br.global_indices):
                    tensor = (
                        br.audio_tensors[j]
                        if j < len(br.audio_tensors)
                        else None
                    )
                    all_results.append((gidx, tensor))

        all_results.sort(key=lambda x: x[0])
        return all_results

    # ------------------------------------------------------------------
    # Public: shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop all worker threads and release models."""
        if self._shut_down:
            return
        self._shut_down = True

        # Send one shutdown sentinel per worker
        for _ in self._workers:
            self._task_queue.put(_SHUTDOWN)

        # Wait for threads to finish
        for t in self._workers:
            t.join(timeout=30)

        # Release model references
        self.models.clear()
        self._gpu_workers.clear()
        self.log_fn("[MultiGPU] Pool shut down")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
