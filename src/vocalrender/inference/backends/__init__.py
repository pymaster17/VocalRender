"""TTS / SVS model inference backends.

Two interchangeable implementations of :class:`TTSInferenceBackend`:

- :class:`MultiGPUBackend` — process-per-GPU backend for external batch
  scripts. It keeps one persistent worker process per GPU and exposes the
  shared ``TTSRequest -> TTSResult`` interface used by both TTS and SVS
  inference entrypoints.
- :class:`NanoVLLMBackend` — out-of-process ``nano-vllm-voxcpm`` server
  pool (default for external batch scripts; 2–4× throughput via
  PagedAttention + continuous batching).

Scripts select a backend via YAML ``inference_backend.type`` and the
:func:`build_tts_backend` factory; training-internal validation paths bypass
the factory and keep using ``vocalrender.evaluation.multi_gpu.MultiGPUInferencePool``
directly.
"""
from .base import TTSInferenceBackend, TTSRequest, TTSResult
from .factory import build_tts_backend

__all__ = [
    "TTSInferenceBackend",
    "TTSRequest",
    "TTSResult",
    "build_tts_backend",
]
