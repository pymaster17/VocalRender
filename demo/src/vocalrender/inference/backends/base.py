"""Common data types + ABC for TTS inference backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TTSRequest:
    """One TTS inference request, backend-agnostic.

    ``prompt_audio_feats``: ``[T, P, D]`` float tensor of pre-encoded prompt
    latents (V1/V1.5/V2). Only the ``multi_gpu`` backend accepts this today.

    ``ref_audio_latents``: ``[T, P, D]`` float latents serialized by the
    nano-vllm server from a reference wav (V2-only speaker-stabilizer). The
    ``multi_gpu`` backend treats this as equivalent to ``prompt_audio_feats``.
    """

    idx: int
    target_text: str
    prompt_audio_feats: Optional[np.ndarray] = None
    ref_audio_latents: Optional[np.ndarray] = None


@dataclass
class TTSResult:
    """One backend generation result, returned in input order."""

    idx: int
    latent: Optional[np.ndarray] = None  # [T, P, D] float32
    audio: Optional[np.ndarray] = None  # [N] float32 at backend.sample_rate
    error: Optional[str] = None


class TTSInferenceBackend(ABC):
    """Synchronous TTS batch-inference interface.

    Backends that are internally async (nano-vllm) must bridge to a private
    event loop so callers stay sync. ``generate`` returns results sorted by
    ``TTSRequest.idx``.
    """

    @abstractmethod
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
        """Run a batch. Sampling kwargs are accepted by every backend but
        only ``multi_gpu`` honours ``temperature_mode`` / ``fsq_temperature``;
        the nano-vllm server exposes no per-request override for those and
        will raise ``NotImplementedError`` on non-default values."""

    @abstractmethod
    def shutdown(self) -> None: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def arch(self) -> str:
        """``"voxcpm"`` (V1/V1.5) or ``"voxcpm2"`` (V2)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
