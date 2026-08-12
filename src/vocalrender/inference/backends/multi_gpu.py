"""Process-based MultiGPU backend for batch TTS/SVS inference.

Unlike the training-internal ``vocalrender.evaluation.multi_gpu`` thread pool,
this backend keeps one persistent worker *process* per GPU. That matches the
training validation architecture more closely and avoids host-side contention
between multiple Python threads driving autoregressive decode loops.
"""
from __future__ import annotations

import inspect
import itertools
import json
import os
import queue
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base import TTSInferenceBackend, TTSRequest, TTSResult


_ARCH_SAMPLE_RATE = {"voxcpm": 44100, "voxcpm2": 48000}
_SHUTDOWN = "__shutdown__"


def _resolve_devices(devices_cfg) -> List[str]:
    if devices_cfg is None or devices_cfg == "auto":
        n = torch.cuda.device_count()
        if n == 0:
            raise RuntimeError("No CUDA device visible.")
        return [f"cuda:{i}" for i in range(n)]
    if isinstance(devices_cfg, str):
        return [devices_cfg if ":" in devices_cfg else f"{devices_cfg}:0"]
    if isinstance(devices_cfg, list) and devices_cfg:
        out: List[str] = []
        for d in devices_cfg:
            if isinstance(d, int):
                out.append(f"cuda:{d}")
            elif isinstance(d, str):
                out.append(d if ":" in d else f"cuda:{d}" if d.isdigit() else d)
            else:
                raise ValueError(f"Cannot parse device: {d!r}")
        return out
    raise ValueError(f"Unrecognized devices: {devices_cfg!r}")


def _resolve_ckpt_path(pretrained_path: str) -> Path:
    path = Path(pretrained_path)
    if (path / "latest").exists():
        path = path / "latest"
    return path


def _detect_arch(pretrained_path: str) -> str:
    with (_resolve_ckpt_path(pretrained_path) / "config.json").open() as f:
        return json.load(f).get("architecture", "voxcpm")


def _load_model_for_device(
    pretrained_path: str,
    device: str,
    *,
    enable_score_lm_head: bool = False,
):
    from vocalrender.training.model_factory import detect_model_classes

    resolved_path = str(_resolve_ckpt_path(pretrained_path))
    model_classes = detect_model_classes(resolved_path)
    model = model_classes.model_cls.from_local(
        resolved_path,
        **(
            {"enable_score_lm_head": enable_score_lm_head}
            if model_classes.arch == "voxcpm2"
            else {}
        ),
    )
    model = model.to(device).eval()
    model.device = device
    model.audio_vae = model.audio_vae.to(torch.float32).to(device)
    return model


def _encode_prompt_wav_standalone(pretrained_path: str, wav_path: str, padding_mode: str = "right") -> np.ndarray:
    import librosa
    import torchaudio

    from vocalrender.model.voxcpm2 import _trim_audio_silence_vad
    from vocalrender.training.vae_loader import load_audio_vae_for_eval

    ckpt_path = _resolve_ckpt_path(pretrained_path)
    with (ckpt_path / "config.json").open() as f:
        cfg = json.load(f)
    arch = str(cfg.get("architecture", "voxcpm")).lower()
    patch_size = int(cfg.get("patch_size", 2))
    chunk_size = int(cfg.get("chunk_size", 1280))

    audio_vae = load_audio_vae_for_eval(str(ckpt_path))
    in_sample_rate = int(getattr(audio_vae, "in_sample_rate"))
    latent_dim = int(getattr(audio_vae, "latent_dim"))
    patch_len = patch_size * chunk_size

    if arch == "voxcpm2":
        audio, _ = librosa.load(wav_path, sr=in_sample_rate, mono=True)
        audio = torch.from_numpy(audio).unsqueeze(0)
        audio = _trim_audio_silence_vad(audio, in_sample_rate, max_silence_ms=200.0)
    else:
        audio, sr = torchaudio.load(wav_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != in_sample_rate:
            audio = torchaudio.functional.resample(audio, sr, in_sample_rate)

    if audio.size(1) % patch_len != 0:
        padding_size = patch_len - audio.size(1) % patch_len
        pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
        audio = torch.nn.functional.pad(audio, pad)

    with torch.no_grad():
        feat = audio_vae.encode(audio.to(torch.float32), in_sample_rate).cpu()
    feat = feat.view(latent_dim, -1, patch_size).permute(1, 2, 0).contiguous()
    return feat.numpy().astype(np.float32)


def _worker_main(
    rank: int,
    worker_cfg: Dict[str, Any],
    task_queue,
    result_queue,
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = worker_cfg["devices"][rank]
    if torch.cuda.is_available() and device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":", 1)[1]))

    model = _load_model_for_device(
        worker_cfg["pretrained_path"],
        device,
        enable_score_lm_head=bool(worker_cfg.get("enable_score_lm_head")),
    )

    while True:
        task = task_queue.get()
        if task == _SHUTDOWN:
            break

        call_id = task["call_id"]
        batch_id = task["batch_id"]
        requests = task["requests"]
        kwargs = task["kwargs"]
        t0 = time.time()

        try:
            prompts = [r["target_text"] for r in requests]
            prompt_audio_feats = []
            any_prompt_audio = False
            for r in requests:
                arr = r.get("prompt_audio_feats")
                if arr is None:
                    arr = r.get("ref_audio_latents")
                if arr is None:
                    prompt_audio_feats.append(None)
                    continue
                t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
                prompt_audio_feats.append(t)
                any_prompt_audio = True
            pa_list = prompt_audio_feats if any_prompt_audio else None

            with torch.no_grad():
                generate_batch_params = inspect.signature(model.generate_batch).parameters
                generate_kwargs = dict(
                    target_texts=prompts,
                    min_len=2,
                    max_len=kwargs["max_gen_len"],
                    inference_timesteps=kwargs["inference_timesteps"],
                    cfg_value=kwargs["cfg_value"],
                    verbose=False,
                    temperature=kwargs["temperature"],
                    temperature_mode=kwargs["temperature_mode"],
                    fsq_temperature=kwargs["fsq_temperature"],
                    prompt_audio_feats=pa_list,
                    return_latents=kwargs["return_latents"],
                    return_latent_dtype=kwargs["return_latent_dtype"],
                )
                out = model.generate_batch(**generate_kwargs)

            if kwargs["return_latents"]:
                audio_tensors, latent_tensors = out
            else:
                audio_tensors = out
                latent_tensors = [None] * len(requests)

            batch_results = []
            for req, audio_t, latent_t in zip(requests, audio_tensors, latent_tensors):
                audio_np = None
                latent_np = None
                if kwargs["return_audio"] and isinstance(audio_t, torch.Tensor) and audio_t.numel() > 0:
                    audio_np = audio_t.detach().cpu().float().numpy().reshape(-1)
                if kwargs["return_latents"] and isinstance(latent_t, torch.Tensor) and latent_t.numel() > 0:
                    latent_np = latent_t.detach().cpu().float().numpy()
                batch_results.append({
                    "idx": req["idx"],
                    "audio": audio_np,
                    "latent": latent_np,
                    "error": None,
                })

            result_queue.put({
                "call_id": call_id,
                "batch_id": batch_id,
                "device": device,
                "elapsed": time.time() - t0,
                "results": batch_results,
            })
        except Exception as exc:  # noqa: BLE001
            result_queue.put({
                "call_id": call_id,
                "batch_id": batch_id,
                "device": device,
                "elapsed": time.time() - t0,
                "results": [
                    {
                        "idx": req["idx"],
                        "audio": None,
                        "latent": None,
                        "error": repr(exc),
                    }
                    for req in requests
                ],
            })


class MultiGPUBackend(TTSInferenceBackend):
    """Process-based backend holding one persistent worker process per GPU.

    Callers are expected to pre-order requests for batching efficiency. The
    backend preserves request order when forming batches and only sorts the
    final results by ``TTSRequest.idx``.
    """

    def __init__(
        self,
        pretrained_path: str,
        devices=None,
        batch_size: int = 16,
        return_latent_dtype: str = "float32",
        enable_score_lm_head: bool = False,
        log_fn=None,
    ) -> None:
        import torch.multiprocessing as mp

        self._pretrained_path = pretrained_path
        self._arch = _detect_arch(pretrained_path)
        self._sample_rate = _ARCH_SAMPLE_RATE.get(self._arch, 44100)
        self._return_latent_dtype = return_latent_dtype
        self._batch_size = int(batch_size)
        self._devices = _resolve_devices(devices)
        self._log_fn = log_fn or (lambda msg: None)
        self._shut_down = False
        self._generate_lock = __import__("threading").Lock()
        self._call_counter = itertools.count()
        self._ctx = mp.get_context("spawn")
        self._task_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._workers = []

        worker_cfg = {
            "pretrained_path": pretrained_path,
            "devices": self._devices,
            "enable_score_lm_head": bool(enable_score_lm_head),
        }
        self._log_fn(
            f"[MultiGPUBackend] Spawning {len(self._devices)} worker process(es): {self._devices}"
        )
        for rank in range(len(self._devices)):
            proc = self._ctx.Process(
                target=_worker_main,
                args=(rank, worker_cfg, self._task_queue, self._result_queue),
                daemon=True,
            )
            proc.start()
            self._workers.append(proc)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def arch(self) -> str:
        return self._arch

    def encode_prompt_wav(self, wav_path: str, padding_mode: str = "right") -> np.ndarray:
        return _encode_prompt_wav_standalone(
            self._pretrained_path, wav_path, padding_mode=padding_mode,
        )

    def generate(
        self,
        requests: List[TTSRequest],
        *,
        return_latents: bool,
        return_audio: bool,
        cfg_value: float,
        inference_timesteps: int,
        max_gen_len: int,
        temperature: float = 1.0,
        temperature_mode: str = "scale",
        fsq_temperature: float = 0.0,
    ) -> List[TTSResult]:
        if not requests:
            return []
        if self._shut_down:
            raise RuntimeError("MultiGPUBackend has been shut down")

        kwargs = {
            "return_latents": bool(return_latents),
            "return_audio": bool(return_audio),
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "max_gen_len": max_gen_len,
            "temperature": float(temperature),
            "temperature_mode": temperature_mode,
            "fsq_temperature": float(fsq_temperature),
            "return_latent_dtype": self._return_latent_dtype,
        }

        serialized = [
            {
                "idx": r.idx,
                "target_text": r.target_text,
                "prompt_audio_feats": (
                    np.asarray(r.prompt_audio_feats, dtype=np.float32)
                    if r.prompt_audio_feats is not None else None
                ),
                "ref_audio_latents": (
                    np.asarray(r.ref_audio_latents, dtype=np.float32)
                    if r.ref_audio_latents is not None else None
                ),
            }
            for r in requests
        ]

        with self._generate_lock:
            call_id = next(self._call_counter)
            num_batches = (len(serialized) + self._batch_size - 1) // self._batch_size
            for batch_id in range(num_batches):
                start = batch_id * self._batch_size
                end = min(start + self._batch_size, len(serialized))
                self._task_queue.put({
                    "call_id": call_id,
                    "batch_id": batch_id,
                    "requests": serialized[start:end],
                    "kwargs": kwargs,
                })

            gathered: List[TTSResult] = []
            pending = num_batches
            while pending > 0:
                try:
                    msg = self._result_queue.get(timeout=3600)
                except queue.Empty as exc:
                    raise TimeoutError("Timed out waiting for MultiGPUBackend workers") from exc
                if msg.get("call_id") != call_id:
                    continue
                pending -= 1
                self._log_fn(
                    f"  [{msg['device']}] Batch {msg['batch_id']} done "
                    f"({len(msg['results'])} prompts, {msg['elapsed']:.1f}s)"
                )
                for item in msg["results"]:
                    gathered.append(TTSResult(
                        idx=item["idx"],
                        latent=item["latent"],
                        audio=item["audio"],
                        error=item["error"],
                    ))

        gathered.sort(key=lambda r: r.idx)
        return gathered

    def shutdown(self) -> None:
        if self._shut_down:
            return
        for _ in self._workers:
            self._task_queue.put(_SHUTDOWN)
        for proc in self._workers:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        try:
            self._task_queue.close()
        except Exception:
            pass
        try:
            self._result_queue.close()
        except Exception:
            pass
        self._shut_down = True
