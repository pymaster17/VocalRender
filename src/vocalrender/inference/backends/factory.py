"""TTS backend factory.

YAML schema::

    inference_backend:
      type: nano_vllm          # or multi_gpu
      # --- common ---
      pretrained_path: ...     # (also read from top-level if omitted)
      devices: auto            # or [0, 1] / ["cuda:0"]
      # --- nano_vllm ---
      max_num_seqs: 32
      max_num_batched_tokens: 8192
      max_model_len: 4096
      gpu_memory_utilization: 0.90
      concurrency_multiplier: 2
      load_audio_vae: false
      inference_timesteps: 10
      # --- multi_gpu ---
      batch_size: 16
      return_latent_dtype: float32

The factory accepts a top-level ``pretrained_path`` fall-through so existing
configs can keep their outer field and only add the ``inference_backend``
block.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import TTSInferenceBackend


def build_tts_backend(
    cfg: Dict[str, Any],
    *,
    pretrained_path: Optional[str] = None,
    log_fn=None,
) -> TTSInferenceBackend:
    """Instantiate a backend from a YAML ``inference_backend`` block.

    ``pretrained_path`` may be passed explicitly (picked up from the outer
    config) or nested inside ``cfg``. Unknown kwargs are forwarded to the
    backend constructor so YAML adds don't require factory changes.
    """
    if cfg is None:
        raise ValueError("inference_backend config is missing.")
    cfg = dict(cfg)
    backend_type = cfg.pop("type", None)
    if not backend_type:
        raise ValueError("inference_backend.type is required (multi_gpu | nano_vllm).")

    if pretrained_path is None:
        pretrained_path = cfg.pop("pretrained_path", None)
    else:
        cfg.pop("pretrained_path", None)
    if not pretrained_path:
        raise ValueError("pretrained_path must be provided to build_tts_backend.")

    if backend_type == "multi_gpu":
        from .multi_gpu import MultiGPUBackend

        return MultiGPUBackend(
            pretrained_path=pretrained_path,
            log_fn=log_fn,
            **cfg,
        )
    if backend_type == "nano_vllm":
        from .nano_vllm import NanoVLLMBackend

        return NanoVLLMBackend(
            pretrained_path=pretrained_path,
            **cfg,
        )
    raise ValueError(
        f"Unknown inference_backend.type={backend_type!r}; "
        "expected 'multi_gpu' or 'nano_vllm'."
    )
