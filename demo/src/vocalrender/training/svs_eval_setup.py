"""
SVS evaluator and AudioVAE loader setup for VoxCPM training.

Provides two factory helpers used by the SVS training script:

* :func:`build_svs_evaluator` — constructs the shared :class:`SVSEvaluator`
  instance with sensible defaults when no ``eval_metrics`` config is supplied.
* :func:`make_audio_vae_loader` — returns a zero-argument callable that
  lazily loads the AudioVAE from a pretrained checkpoint directory, suitable
  for passing to :func:`~vocalrender.training.validation.validate_svs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# Default eval_metrics when the user does not specify one in their YAML.
_DEFAULT_EVAL_METRICS = {
    "singmos": {"enabled": True, "model": "singmos_pro", "local_model_path": ""},
    "aes": {"enabled": True, "axes": ["CE", "PQ"], "local_ckpt": ""},
}


@dataclass
class SVSEvalArtifacts:
    """Artifacts produced by :func:`build_svs_evaluator`."""

    evaluator: object
    """Configured :class:`~vocalrender.evaluation.svs_metrics.SVSEvaluator` instance."""

    eval_ref_cache: dict = field(default_factory=dict)
    """Mutable reference-baseline cache, passed to each ``validate_svs`` call.

    The evaluator reads and writes this dict to avoid re-computing reference
    SingMOS / AES scores on every validation pass.
    """

    load_audio_vae_fn: Optional[Callable] = None
    """Zero-argument callable that returns a fresh AudioVAE on demand.

    ``None`` when ``pretrained_path`` is not supplied (e.g. during unit tests).
    """


def build_svs_evaluator(
    eval_metrics: Optional[dict],
    *,
    project_root: Path,
    sample_rate: int,
    log_fn,
) -> object:
    """Construct a :class:`SVSEvaluator` with default metric config fallback.

    Parameters
    ----------
    eval_metrics : dict or None
        Per-metric configuration dict from the training YAML.  When ``None``,
        the built-in defaults (SingMOS + AES CE/PQ) are used.
    project_root : Path
        Repository root, forwarded to ``SVSEvaluator`` for resolving relative
        checkpoint paths.
    sample_rate : int
        Audio sample rate (Hz).
    log_fn : callable
        One-argument logging function (e.g. ``tracker.print``).

    Returns
    -------
    SVSEvaluator
    """
    from vocalrender.evaluation.svs_metrics import SVSEvaluator

    if eval_metrics is None:
        eval_metrics = dict(_DEFAULT_EVAL_METRICS)

    return SVSEvaluator(
        eval_metrics,
        project_root=project_root,
        sample_rate=sample_rate,
        log_fn=log_fn,
    )


def make_audio_vae_loader(pretrained_path: str) -> Callable:
    """Return a zero-argument callable that lazily loads the AudioVAE.

    The returned function is passed to :func:`~vocalrender.training.validation.validate_svs`
    as ``load_audio_vae_fn``.  The AudioVAE is loaded fresh on each call so
    that it can be released from GPU memory immediately after each validation
    pass.

    Parameters
    ----------
    pretrained_path : str
        Path to the pretrained checkpoint directory (the same directory passed
        to ``VoxCPMModel.from_local``).

    Returns
    -------
    callable
        ``() -> AudioVAE``
    """
    from vocalrender.training.vae_loader import load_audio_vae_for_eval

    def _load():
        return load_audio_vae_for_eval(pretrained_path)

    return _load


def setup_svs_eval(
    eval_metrics: Optional[dict],
    pretrained_path: str,
    *,
    project_root: Path,
    sample_rate: int,
    log_fn,
    rank: int = 0,
) -> SVSEvalArtifacts:
    """One-shot helper that builds the evaluator and the VAE loader together.

    This is the entry point called by the SVS training script.  It also logs
    the active metric set on rank 0.

    Parameters
    ----------
    eval_metrics : dict or None
        Per-metric config dict from the training YAML (``None`` → defaults).
    pretrained_path : str
        Path to the pretrained checkpoint directory (for AudioVAE loading).
    project_root : Path
        Repository root for resolving relative eval checkpoint paths.
    sample_rate : int
        Audio sample rate (Hz).
    log_fn : callable
        Logging function, called only on rank 0.
    rank : int
        Current process rank (logging is suppressed on non-zero ranks).

    Returns
    -------
    SVSEvalArtifacts
    """
    evaluator = build_svs_evaluator(
        eval_metrics,
        project_root=project_root,
        sample_rate=sample_rate,
        log_fn=log_fn,
    )

    if rank == 0:
        active_metrics = evaluator.active_metrics_label()
        log_fn(f"[Eval] Active metrics: {active_metrics}")
        if evaluator.aes_axes:
            log_fn(f"[Eval] AES axes: {evaluator.aes_axes}")

    load_audio_vae_fn = make_audio_vae_loader(pretrained_path)

    return SVSEvalArtifacts(
        evaluator=evaluator,
        eval_ref_cache={},
        load_audio_vae_fn=load_audio_vae_fn,
    )
