"""nano-vllm-voxcpm backend — out-of-process PagedAttention + continuous batching.

Internally drives ``AsyncVoxCPM2ServerPool`` (or ``AsyncVoxCPMServerPool``
for V1/V1.5) via a private event loop. Callers see a synchronous interface.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .base import TTSInferenceBackend, TTSRequest, TTSResult


_ARCH_SAMPLE_RATE = {"voxcpm": 44100, "voxcpm2": 48000}


def _ensure_vocab_size_matches_weights(pretrained_path: str) -> str:
    """Return a ckpt dir whose config.json vocab_size matches the saved
    ``base_lm.embed_tokens`` rows (and extended tokenizer length).

    SVS finetune checkpoints copy ``config.json`` verbatim from the base
    model, so ``lm_config.vocab_size`` lags behind the extended tokenizer /
    resized embedding saved in ``model.safetensors``. nano-vllm sizes its
    embedding from ``config.json`` directly — the skew triggers a CUDA
    gather OOB when SVS tokens (id ≥ 73510) are prefilled.

    ``load_svs_model()`` patches ``config.lm_config.vocab_size`` in-memory
    before instantiating the model; we replicate that here by writing a
    side-car dir with every original file symlinked and a single rewritten
    ``config.json``. Idempotent: returns the original dir when everything is
    already consistent.
    """
    from safetensors import safe_open
    from transformers import AutoTokenizer

    src = Path(pretrained_path)
    cfg_path = src / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg_vocab = (cfg.get("lm_config") or {}).get("vocab_size") or cfg.get("vocab_size")

    st_path = src / "model.safetensors"
    if not st_path.exists():
        return pretrained_path  # non-safetensors ckpts: defer to engine
    with safe_open(str(st_path), framework="pt") as f:
        emb_rows = None
        for key in ("base_lm.embed_tokens.weight", "base_lm.model.embed_tokens.weight"):
            if key in f.keys():
                emb_rows = f.get_slice(key).get_shape()[0]
                break
        if emb_rows is None:
            for k in f.keys():
                if k.startswith("base_lm") and "embed_tokens" in k:
                    emb_rows = f.get_slice(k).get_shape()[0]
                    break
    if emb_rows is None:
        return pretrained_path

    try:
        tok_len = len(AutoTokenizer.from_pretrained(str(src)))
    except Exception:
        tok_len = emb_rows

    if cfg_vocab == emb_rows == tok_len:
        return pretrained_path  # already consistent — no side-car needed

    # The embedding row count is authoritative — nano-vllm's loader sizes the
    # shard from ``config.vocab_size`` and then reads that many rows from the
    # safetensors. Inflating beyond ``emb_rows`` (e.g. when the tokenizer has
    # been extended but the weights were not resized — typical for a base TTS
    # ckpt whose tokenizer was bumped during SVS prep) would trigger a
    # ``narrow(..., size > dim)`` error. Clamp to what the weights actually
    # contain, and warn if the tokenizer has extra unmapped ids.
    if tok_len > emb_rows:
        print(
            f"[nano_vllm] WARNING: tokenizer length ({tok_len}) exceeds "
            f"embedding rows ({emb_rows}) in {src}; {tok_len - emb_rows} "
            "extra token id(s) have no embedding and must not appear in "
            "inputs. Capping config.vocab_size at emb_rows.",
            flush=True,
        )
    true_vocab = emb_rows

    # Side-car lives next to the ckpt so it moves/deletes together.
    sidecar = src.parent / f"{src.name}.nanovllm_patched"
    sidecar_cfg = sidecar / "config.json"
    if sidecar_cfg.exists():
        try:
            existing = json.loads(sidecar_cfg.read_text())
            existing_vocab = (existing.get("lm_config") or {}).get("vocab_size") or existing.get("vocab_size")
            if existing_vocab == true_vocab:
                return str(sidecar)
        except Exception:
            pass

    sidecar.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        dst = sidecar / f.name
        dst.unlink(missing_ok=True)
        if f.name == "config.json":
            continue
        os.symlink(f.resolve(), dst)

    patched = dict(cfg)
    if isinstance(patched.get("lm_config"), dict):
        patched["lm_config"] = dict(patched["lm_config"])
        patched["lm_config"]["vocab_size"] = true_vocab
    if "vocab_size" in patched:
        patched["vocab_size"] = true_vocab
    sidecar_cfg.write_text(json.dumps(patched, indent=2))
    return str(sidecar)


def _resolve_gpu_indices(devices_cfg) -> List[int]:
    """Convert a YAML ``devices`` field to nano-vllm's integer-index format."""
    if devices_cfg is None or devices_cfg == "auto":
        n = torch.cuda.device_count()
        if n == 0:
            raise RuntimeError("No CUDA device visible.")
        return list(range(n))
    if isinstance(devices_cfg, int):
        return [devices_cfg]
    if isinstance(devices_cfg, str):
        s = devices_cfg.strip()
        if ":" in s:
            return [int(s.split(":", 1)[1])]
        if s.isdigit():
            return [int(s)]
        raise ValueError(f"Cannot parse device string: {devices_cfg!r}")
    if isinstance(devices_cfg, list) and devices_cfg:
        out: List[int] = []
        for d in devices_cfg:
            if isinstance(d, int):
                out.append(d)
            elif isinstance(d, str) and ":" in d:
                out.append(int(d.split(":", 1)[1]))
            elif isinstance(d, str) and d.isdigit():
                out.append(int(d))
            else:
                raise ValueError(f"Cannot parse device: {d!r}")
        return out
    raise ValueError(f"Unrecognized devices: {devices_cfg!r}")


def _detect_arch(pretrained_path: str) -> str:
    with (Path(pretrained_path) / "config.json").open() as f:
        return json.load(f).get("architecture", "unknown")


class NanoVLLMBackend(TTSInferenceBackend):
    """Sync wrapper around nano-vllm-voxcpm's async server pool.

    A background thread owns a dedicated asyncio loop so that a fresh
    ``generate`` call doesn't fight with any outer async context the caller
    may be in (current callers are all sync CLIs, so this is just insurance).

    ``return_audio=True`` requires ``load_audio_vae=True``. We pre-serialize
    the reference wav once at the driver side (via ``encode_latents``) and
    broadcast the latent bytes to every request.

    Concurrency knobs are *per-GPU* — the backend multiplies by
    ``len(devices)`` internally so ``max_num_seqs`` / ``concurrency_multiplier``
    can stay fixed when scaling the GPU count:

      • ``max_num_seqs``           — scheduler depth inside each server
                                     (forwarded to every per-GPU server as-is).
      • ``concurrency_multiplier`` — driver-side oversubscription factor
                                     per server; the global admission
                                     semaphore is
                                     ``max_num_seqs × concurrency_multiplier
                                     × len(devices)``.
    """

    def __init__(
        self,
        pretrained_path: str,
        devices=None,
        max_num_seqs: int = 32,
        max_num_batched_tokens: int = 8192,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        concurrency_multiplier: int = 2,
        load_audio_vae: bool = False,
        inference_timesteps: int = 10,
    ) -> None:
        pretrained_path = _ensure_vocab_size_matches_weights(pretrained_path)
        self._pretrained_path = pretrained_path
        self._arch = _detect_arch(pretrained_path)
        if self._arch not in _ARCH_SAMPLE_RATE:
            raise ValueError(
                f"NanoVLLMBackend: unsupported arch={self._arch!r} at {pretrained_path}."
            )
        self._sample_rate = _ARCH_SAMPLE_RATE[self._arch]
        self._max_num_seqs = int(max_num_seqs)
        self._concurrency_mult = int(concurrency_multiplier)
        # ``load_audio_vae`` is only wired on the voxcpm2 server path; the V1
        # server rejects unknown kwargs. Swallow it silently for V1 so YAMLs
        # remain arch-agnostic.
        self._load_audio_vae = bool(load_audio_vae) and self._arch == "voxcpm2"

        from nanovllm_voxcpm import VoxCPM  # local import: optional dep

        gpu_indices = _resolve_gpu_indices(devices)
        self._devices = gpu_indices

        # Dedicated loop in a background thread — lets `.generate()` stay
        # sync regardless of the caller's asyncio state.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="nanovllm-backend-loop", daemon=True
        )
        self._loop_thread.start()

        from_pretrained_kwargs = dict(
            model=pretrained_path,
            inference_timesteps=int(inference_timesteps),
            max_num_batched_tokens=int(max_num_batched_tokens),
            max_num_seqs=self._max_num_seqs,
            max_model_len=int(max_model_len),
            gpu_memory_utilization=float(gpu_memory_utilization),
            devices=gpu_indices,
        )
        if self._arch == "voxcpm2":
            from_pretrained_kwargs["load_audio_vae"] = self._load_audio_vae
        self._server = self._submit(VoxCPM.from_pretrained)(**from_pretrained_kwargs)
        self._run_coro(self._server.wait_for_ready())
        self._shut_down = False

        # Driver-side standalone AudioVAE used by the
        # (return_latents=True, return_audio=True) path on V2: the server is
        # non-streaming in that mode (one final ``[T, P, D]`` latent message,
        # no waveform chunks), so the wrapper synthesizes audio locally.
        # Lazy-loaded; GPU when CUDA is available, CPU otherwise.
        self._decode_audio_vae = None  # type: ignore[assignment]
        self._decode_audio_vae_lock = threading.Lock()
        # Cap concurrent driver-side GPU decodes: without this each ``_one``
        # coroutine fires its own ``asyncio.to_thread``, and many simultaneous
        # AudioVAE forwards on the same cuda:0 (also hosting a worker at
        # ``gpu_memory_utilization`` ≥ 0.9) overshoot VRAM. The Snake
        # activation in audio_vae_v2.py holds 3 transient tensors per forward
        # so peak ≈ N · single-forward; cap=2 keeps peak under the worker's
        # gpu_memory_utilization headroom on the longest samples.
        self._decode_concurrency_sem = threading.Semaphore(2)

    # ------------------------------------------------------------------
    # Event-loop plumbing
    # ------------------------------------------------------------------

    def _submit(self, fn):
        """Wrap a sync factory so its call runs on the backend loop's thread."""
        def _wrapped(*args, **kwargs):
            fut = asyncio.run_coroutine_threadsafe(
                self._maybe_await(fn, *args, **kwargs), self._loop
            )
            return fut.result()

        return _wrapped

    @staticmethod
    async def _maybe_await(fn, *args, **kwargs):
        out = fn(*args, **kwargs)
        if asyncio.iscoroutine(out):
            out = await out
        return out

    def _run_coro(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def arch(self) -> str:
        return self._arch

    def encode_reference_wav(self, wav_path: str) -> bytes:
        """Encode a reference wav once; result is reused across requests."""
        if not self._load_audio_vae:
            raise RuntimeError(
                "encode_reference_wav requires load_audio_vae=True."
            )
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        return self._run_coro(self._server.encode_latents(wav_bytes, "wav"))

    def encode_prompt_wav(self, wav_path: str, padding_mode: str = "right") -> np.ndarray:
        """Driver-side prompt-audio encode for the static-ref path.

        Mirrors :meth:`MultiGPUBackend.encode_prompt_wav` so callers (notably
        ``_build_static_ref_prompt_audio_config``) treat both backends
        uniformly. Runs the AudioVAE on CPU via the standalone helper, so it
        does not contend with the nano-vllm workers' GPU memory.
        """
        from vocalrender.inference.backends.multi_gpu import _encode_prompt_wav_standalone
        return _encode_prompt_wav_standalone(
            self._pretrained_path, wav_path, padding_mode
        )

    def _ensure_decode_audio_vae(self):
        """Lazy-load the driver-side standalone AudioVAE used to decode
        ``[T, P, D]`` latents to waveform when callers request
        ``return_latents=True`` and ``return_audio=True`` together.

        Lives on the first nano-vllm worker GPU when CUDA is available (sits
        in the ``gpu_memory_utilization`` headroom; an AudioVAE-V2 in fp32 is
        ~800 MB so 10% of a 40 GB A100 covers it comfortably) — CPU decode is
        ~30× slower per sample and would back-pressure the server even with
        ``sem`` released early. Falls back to CPU when no CUDA is visible
        (smoke tests on head node, etc.). Lock-guarded because ``_one`` runs
        many requests concurrently and the first one to need a decode would
        otherwise race the lazy load.
        """
        if self._decode_audio_vae is not None:
            return self._decode_audio_vae
        with self._decode_audio_vae_lock:
            if self._decode_audio_vae is None:
                from vocalrender.training.vae_loader import load_audio_vae_for_eval

                audio_vae = load_audio_vae_for_eval(self._pretrained_path)
                if torch.cuda.is_available() and self._devices:
                    # ``self._devices`` is List[int] of physical GPU indices.
                    # The driver's view of those indices is set by the
                    # parent process's CUDA_VISIBLE_DEVICES; nano-vllm worker
                    # servers translate the same indices internally, so
                    # co-locating on ``cuda:{self._devices[0]}`` shares the
                    # device with one worker. The decode is a single small
                    # forward (~5 ms on A100) so it doesn't materially
                    # contend with the worker's decode steps.
                    self._decode_device = f"cuda:{self._devices[0]}"
                else:
                    self._decode_device = "cpu"
                self._decode_audio_vae = audio_vae.to(self._decode_device)
        return self._decode_audio_vae

    def _decode_latent_standalone(self, latent: np.ndarray) -> np.ndarray:
        """Decode a ``[T, P, D]`` latent tensor to a flat ``[N]`` waveform.

        Mirrors the rearrangement used in
        :func:`vocalrender.evaluation.audio_utils.decode_reference_audio`:
        ``[T, P, D] -> [1, T*P, D] -> [1, D, T*P]`` then ``audio_vae.decode``.
        Returns float32 numpy on CPU.
        """
        from einops import rearrange

        audio_vae = self._ensure_decode_audio_vae()
        # CPU prep is cheap and parallel-safe; only the GPU forward needs
        # serialization to keep VRAM peaks under the worker server's
        # ``gpu_memory_utilization`` headroom on cuda:0.
        feats = torch.from_numpy(np.ascontiguousarray(latent)).float()
        feats = rearrange(feats, "t p d -> 1 (t p) d").transpose(1, 2).contiguous()
        feats = feats.to(self._decode_device)
        with self._decode_concurrency_sem:
            with torch.no_grad():
                wav = audio_vae.decode(feats)
            # ``cpu()`` blocks on the CUDA stream, so it must stay inside the
            # sem — releasing earlier would let a peer thread start its own
            # forward before this one's activations are freed.
            out = wav.detach().cpu().float().numpy().reshape(-1)
        return out

    @staticmethod
    def _serialize_pa(np_arr: np.ndarray) -> bytes:
        """Serialise a ``[T, P, D]`` float prompt-audio latent to bytes for
        the server's ``ref_audio_latents`` byte path.

        The server reshapes the buffer to ``[-1, feat_dim]`` then to
        ``[T, P, D]`` (engine.py:148) — shapes must match the model's
        ``patch_size`` × ``feat_dim``.
        """
        arr = np.ascontiguousarray(np.asarray(np_arr, dtype=np.float32))
        if arr.ndim != 3:
            raise ValueError(
                f"prompt_audio_feats must be [T, P, D]; got shape={arr.shape}"
            )
        return arr.tobytes()

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
        # nano-vllm server only honours ``temperature``; ``temperature_mode``
        # and ``fsq_temperature`` would be silently dropped, which is a trap
        # for diverse sampling. Reject non-defaults up front.
        if temperature_mode != "scale" or float(fsq_temperature) != 0.0:
            raise NotImplementedError(
                "nano_vllm backend does not expose temperature_mode / "
                "fsq_temperature; switch to multi_gpu for diverse sampling."
            )
        # ``return_audio=True`` needs an AudioVAE somewhere:
        #   - when ``return_latents=False`` the server streams waveform
        #     chunks itself (needs server-side ``load_audio_vae=True``),
        #   - when ``return_latents=True`` the server is in non-streaming
        #     latent mode (no waveform from server) and the wrapper synthesizes
        #     audio driver-side via ``_decode_latent_standalone`` — that path
        #     loads its own CPU AudioVAE on demand and works regardless of
        #     the server-side ``load_audio_vae`` flag.
        if (
            self._arch == "voxcpm2"
            and return_audio
            and not return_latents
            and not self._load_audio_vae
        ):
            raise RuntimeError(
                "NanoVLLMBackend.generate(return_audio=True, return_latents=False) "
                "requires load_audio_vae=True (the server needs the AudioVAE to "
                "stream waveforms). Pass return_latents=True to use the "
                "driver-side standalone-decode path instead."
            )
        # V1 nano-vllm server always yields waveform chunks and has no
        # latent-dump mode; mixing return_latents with arch=voxcpm is a
        # configuration error the caller should resolve by switching backends.
        if self._arch == "voxcpm" and return_latents:
            raise NotImplementedError(
                "NanoVLLMBackend(arch=voxcpm) cannot return latents — the V1 "
                "server streams waveforms only. Use the multi_gpu backend if "
                "you need latent dumps."
            )
        # ``prompt_audio_feats`` is now wrapped through to the server's
        # ``ref_audio_latents`` byte path — the model-level layout is identical
        # (`<103> [zeros×T] <104>` prefix). See ``_one`` below for the
        # serialisation; ``ref_audio_latents`` and ``prompt_audio_feats`` are
        # mutually exclusive on a single request.

        coro = self._generate_async(
            requests,
            return_latents=return_latents,
            return_audio=return_audio,
            cfg_value=cfg_value,
            max_gen_len=max_gen_len,
            temperature=float(temperature),
        )
        # ``inference_timesteps`` was fixed at server boot; the nano-vllm
        # server exposes no per-request override. Silently ignored here (we
        # keep the kwarg for API parity with MultiGPUBackend).
        _ = inference_timesteps
        return self._run_coro(coro)

    async def _generate_async(
        self,
        requests: List[TTSRequest],
        *,
        return_latents: bool,
        return_audio: bool,
        cfg_value: float,
        max_gen_len: int,
        temperature: float = 1.0,
    ) -> List[TTSResult]:
        # Both ``max_num_seqs`` and ``concurrency_multiplier`` are *per-GPU*:
        # ``max_num_seqs`` is forwarded verbatim to every server (one server
        # per GPU), so the server-side total scheduler capacity is
        # ``max_num_seqs × len(devices)``. The driver-side admission pool must
        # scale the same way, otherwise the global semaphore caps each server
        # at ``1/len(devices)`` of its scheduler depth and the tail GPUs stay
        # idle. Multiplying by ``len(devices)`` here keeps both knobs
        # GPU-count-agnostic — scaling from 1 → 8 GPUs needs no YAML change.
        n_devices = max(1, len(self._devices))
        sem = asyncio.Semaphore(
            self._max_num_seqs * self._concurrency_mult * n_devices
        )

        async def _one(req: TTSRequest) -> TTSResult:
            latent_chunks: List[np.ndarray] = []
            audio_chunks: List[np.ndarray] = []
            err: Optional[str] = None
            # The admission ``sem`` exists to cap concurrent in-flight server
            # requests so we don't oversubscribe the per-GPU schedulers. It
            # MUST only wrap the server stream — if it also covered the
            # driver-side CPU latent->audio decode (the (return_latents=True,
            # return_audio=True) path), slow CPU decodes would hold sem slots
            # and starve the server's continuous batching. Empirical cost of
            # getting this wrong: decode_bs collapses from ~7 to ~1.2 after
            # the first few hundred samples on a 13.5k-request sweep.
            async with sem:
                # nano-vllm's chunk shape depends on return_latents flag we
                # pass to the server: on that path each chunk is
                # ``[patch_size, feat_dim]`` float32.
                try:
                    gen_kwargs = dict(
                        target_text=req.target_text,
                        cfg_value=cfg_value,
                        max_generate_length=max_gen_len,
                        temperature=temperature,
                    )
                    if self._arch == "voxcpm2":
                        # V2 server is chunk-type-aware; latents vs waveform
                        # toggle is explicit.
                        gen_kwargs["return_latents"] = bool(return_latents)
                        # ``ref_audio_latents`` (bytes from server-side encode)
                        # and ``prompt_audio_feats`` (np.ndarray pre-extracted
                        # latents) build the same `<103>[zeros×T]<104>` prefix
                        # server-side; a single request can carry only one.
                        if (
                            req.ref_audio_latents is not None
                            and req.prompt_audio_feats is not None
                        ):
                            raise ValueError(
                                "ref_audio_latents and prompt_audio_feats are "
                                "mutually exclusive on a single request."
                            )
                        if req.ref_audio_latents is not None:
                            gen_kwargs["ref_audio_latents"] = req.ref_audio_latents
                        elif req.prompt_audio_feats is not None:
                            gen_kwargs["ref_audio_latents"] = self._serialize_pa(
                                req.prompt_audio_feats
                            )
                    else:
                        if (
                            req.ref_audio_latents is not None
                            or req.prompt_audio_feats is not None
                        ):
                            raise NotImplementedError(
                                "ref_audio_latents / prompt_audio_feats are not "
                                "supported on arch=voxcpm (only voxcpm2 wires the "
                                "<103> prompt-audio prefix)."
                            )
                    async for chunk in self._server.generate(**gen_kwargs):
                        arr = np.asarray(chunk, dtype=np.float32)
                        # V2 latent chunk is either:
                        #   - per-step [P, D] (streaming mode; not used for
                        #     return_latents anymore), or
                        #   - the whole pre-stacked [T, P, D] as one final
                        #     message (non-streaming mode — cuts
                        #     driver-side IPC from ~80 msgs/seq to 1 msg/seq,
                        #     which is the real bottleneck at scale).
                        # V2/V1 waveform chunk: flat [N].
                        # The branches are mutually exclusive: in latent mode
                        # the chunk is a stacked latent and must NOT be reshaped
                        # as if it were waveform.
                        if return_latents and self._arch == "voxcpm2":
                            latent_chunks.append(arr)
                        elif return_audio:
                            audio_chunks.append(arr.reshape(-1))
                except Exception as e:  # noqa: BLE001
                    err = repr(e)
            # ``sem`` released here. Driver-side CPU decode runs outside the
            # admission lock so it cannot back-pressure new server requests.

            lat = None
            if return_latents and latent_chunks:
                # Server non-streaming latent path: a single already-stacked
                # [T, P, D] chunk. Skip the outer np.stack that would inject
                # a spurious leading dim.
                if len(latent_chunks) == 1 and latent_chunks[0].ndim == 3:
                    lat = latent_chunks[0]
                else:
                    lat = np.stack(latent_chunks, axis=0)
            aud = None
            if return_audio:
                if audio_chunks:
                    aud = np.concatenate(audio_chunks, axis=0)
                elif (
                    return_latents
                    and lat is not None
                    and self._arch == "voxcpm2"
                    and err is None
                ):
                    # Server was in latent-only non-streaming mode (no
                    # waveform chunks emitted). Synthesize the audio
                    # locally via the driver-side standalone AudioVAE.
                    # ``asyncio.to_thread`` keeps the per-request decode
                    # off the event loop so concurrent requests still
                    # interleave their server I/O with each other's CPU
                    # decode.
                    try:
                        aud = await asyncio.to_thread(
                            self._decode_latent_standalone, lat
                        )
                    except Exception as decode_err:  # noqa: BLE001
                        err = (
                            f"latent->audio standalone decode failed: "
                            f"{decode_err!r}"
                        )
            return TTSResult(idx=req.idx, latent=lat, audio=aud, error=err)

        tasks = [asyncio.create_task(_one(r)) for r in requests]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda r: r.idx)
        return results

    def shutdown(self) -> None:
        if self._shut_down:
            return
        try:
            self._run_coro(self._server.stop())
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=10)
        try:
            self._loop.close()
        except Exception:
            pass
        self._shut_down = True

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
