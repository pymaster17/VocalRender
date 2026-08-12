"""
Shared SVS evaluation module.

`SVSEvaluator` is a single source of truth for SVS quality metrics
(SingMOS, Audiobox Aesthetics, plus pluggable custom backends). It is
consumed by both:

- `scripts/infer_vocalrender_svs.py` (standalone inference, writes JSON)
- `scripts/train_vocalrender_svs.py` (training validation, writes TensorBoard)

The evaluator parses an ``eval_metrics`` dict (the same schema both scripts
accept), lazy-loads only the backends whose sub-metrics are enabled, and
exposes ``evaluate(items, cached_refs=...)`` that returns a unified
``EvalResult`` with scalar metrics, validity counts, and reference-baseline
cache entries.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import torch

from vocalrender.evaluation.metrics import (
    SingMOSFrameCapture,
    compute_batch_aes_scores,
    compute_batch_singmos_scores,
    load_aes_predictor,
    load_singmos_predictor,
)


# ============================================================
# Data classes
# ============================================================

@dataclass
class EvalItem:
    """One evaluation record — generated + reference audio for a single sample.

    Both scripts build `list[EvalItem]` from their native buffers (infer's
    ``results`` list, or training's ``(sample_id, audio_np)`` tuples) via the
    adapter helpers below.
    """

    sample_id: int
    item_name: str
    lyrics: str                               # concatenated words minus SP/AP/""
    gt_words: Optional[List[str]] = None       # raw word tokens (inc. SP/AP/"")
    gt_pitch_count: int = 0                   # number of non-zero pitches
    gt_pitch_total: int = 0                   # total pitches
    ref_audio: Optional[np.ndarray] = None    # float32 mono
    gen_audio: Optional[np.ndarray] = None    # float32 mono


@dataclass
class EvalResult:
    """Container for a single ``evaluate()`` call output."""

    scalars: Dict[str, float] = field(default_factory=dict)
    valid_counts: Dict[str, int] = field(default_factory=dict)
    ref_baselines: Dict[str, Any] = field(default_factory=dict)
    # Generic per-sample scalar sink for custom metric backends (keyed by
    # ``sample_id`` -> {metric_name: value}). Enables per-item dumps for
    # metrics that aggregate internally.
    per_item_scalars: Dict[int, Dict[str, float]] = field(default_factory=dict)


# ============================================================
# Adapters
# ============================================================

_SP_LABELS = {"SP", "AP", ""}


def _extract_lyrics(gt_words: List[str]) -> str:
    return "".join(w for w in gt_words if w not in _SP_LABELS)


def items_from_infer_results(results: List[Dict], val_ds) -> List[EvalItem]:
    """Build ``EvalItem`` list from ``scripts/infer_vocalrender_svs.py`` results.

    ``results`` is the output of ``run_inference_batch``; each dict has at
    least ``sample_id`` and ``item_name``, and when ``return_audio=True``
    also ``audio_np`` / ``ref_audio_np``.
    """
    items: List[EvalItem] = []
    for r in results:
        sid = r["sample_id"]
        sample = val_ds[sid]
        gt_words_raw = [str(w) for w in sample.get("word", []) or []]
        gt_pitch = sample.get("pitch", []) or []
        items.append(EvalItem(
            sample_id=sid,
            item_name=str(r.get("item_name", f"sample_{sid}")),
            lyrics=_extract_lyrics(gt_words_raw),
            gt_words=gt_words_raw,
            gt_pitch_count=sum(int(p) > 0 for p in gt_pitch),
            gt_pitch_total=len(gt_pitch),
            ref_audio=r.get("ref_audio_np"),
            gen_audio=r.get("audio_np"),
        ))
    return items


def items_from_train_buffers(
    gen_audio_list: List,
    ref_audio_list: List,
    val_ds,
) -> List[EvalItem]:
    """Build ``EvalItem`` list from training's generated/reference buffers.

    ``gen_audio_list`` and ``ref_audio_list`` are ``[(sample_id, np.ndarray)]``
    pairs produced inside ``generate_sample_audio_svs``.
    """
    ref_map = dict(ref_audio_list) if ref_audio_list else {}

    items: List[EvalItem] = []
    for sid, gen_np in gen_audio_list:
        sample = val_ds[sid]
        gt_words_raw = [str(w) for w in sample.get("word", []) or []]
        gt_pitch = sample.get("pitch", []) or []
        items.append(EvalItem(
            sample_id=sid,
            item_name=str(sample.get("item_name", f"sample_{sid}")),
            lyrics=_extract_lyrics(gt_words_raw),
            gt_words=gt_words_raw,
            gt_pitch_count=sum(int(p) > 0 for p in gt_pitch),
            gt_pitch_total=len(gt_pitch),
            ref_audio=ref_map.get(sid),
            gen_audio=gen_np,
        ))
    return items


# ============================================================
# Custom metric backend registry (extension seam)
# ============================================================
#
# New metrics (FAD, singer-similarity/SECS, F0-RMSE/VUV/RCA, MCD, ...) register
# here WITHOUT editing the evaluator core. A backend is a
# ``(loader, runner, needs_ref)`` triple keyed by name; it is activated per-run
# by an ``eval_metrics.custom_metrics.<name>.enabled: true`` config toggle.
#
#   loader(evaluator, devices, cfg) -> model_handle | None
#       Called once in ``load_models`` for each enabled backend. May return
#       ``None`` (metric needs no loaded model — e.g. pure-DSP F0/MCD).
#   runner(evaluator, items, result, model_handle, cfg) -> None
#       Called once per ``evaluate`` after the built-in backends. Mutates
#       ``result.scalars`` / ``result.valid_counts`` in place. Per-sample
#       metrics read ``item.gen_audio`` / ``item.ref_audio`` (float32 mono at
#       ``evaluator.sample_rate``).

@dataclass
class MetricBackendSpec:
    loader: Callable
    runner: Callable
    needs_ref: bool = True


_CUSTOM_METRIC_BACKENDS: Dict[str, "MetricBackendSpec"] = {}


def register_metric_backend(
    name: str,
    loader: Callable,
    runner: Callable,
    *,
    needs_ref: bool = True,
) -> None:
    """Register a pluggable metric backend under ``name`` (idempotent overwrite)."""
    _CUSTOM_METRIC_BACKENDS[str(name)] = MetricBackendSpec(
        loader=loader, runner=runner, needs_ref=bool(needs_ref)
    )


def registered_metric_backends() -> List[str]:
    return sorted(_CUSTOM_METRIC_BACKENDS)


# ============================================================
# SVSEvaluator
# ============================================================

class SVSEvaluator:
    """Unified SVS metrics runner.

    The evaluator is **stateful only between `load_models()` and
    `unload_models()`**. All cross-call caching (reference-side baselines)
    flows through the ``cached_refs`` argument of ``evaluate()`` and is
    returned in ``EvalResult.ref_baselines``.
    """

    def __init__(
        self,
        eval_metrics_cfg: Optional[Dict[str, Any]],
        *,
        project_root=None,
        sample_rate: int = 44100,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        cfg = dict(eval_metrics_cfg) if eval_metrics_cfg else {}

        self._project_root = project_root
        self._sample_rate = int(sample_rate)
        self._log = log_fn if log_fn is not None else (lambda msg: print(msg, file=sys.stderr))

        # ---- SingMOS ----
        sm_cfg = cfg.get("singmos", {}) or {}
        self._use_singmos = bool(sm_cfg.get("enabled", False))
        self._singmos_model = sm_cfg.get("model", "singmos_pro")
        self._singmos_local_model = sm_cfg.get("local_model_path", "") or ""

        # ---- AES ----
        aes_cfg = cfg.get("aes", {}) or {}
        self._use_aes = bool(aes_cfg.get("enabled", False))
        self._aes_axes: List[str] = sorted(aes_cfg.get("axes", [])) if self._use_aes else []
        self._aes_local_ckpt = aes_cfg.get("local_ckpt", "") or ""

        # ---- Custom metric backends (pluggable seam) ----
        # ``eval_metrics.custom_metrics`` is ``{name: {enabled: bool, ...cfg}}``.
        # Only enabled + registered backends activate; unknown names warn and
        # are skipped (so a config can name a metric whose impl isn't imported
        # in this environment without hard-failing the whole eval).
        custom_cfg = cfg.get("custom_metrics", {}) or {}
        self._custom_metric_cfgs: Dict[str, Dict[str, Any]] = {}
        for name, section in custom_cfg.items():
            section = section or {}
            if not bool(section.get("enabled", False)):
                continue
            if name not in _CUSTOM_METRIC_BACKENDS:
                self._log(
                    f"[Metric] Warning: custom metric {name!r} enabled but not "
                    f"registered; skipping. Known: {registered_metric_backends()}"
                )
                continue
            self._custom_metric_cfgs[name] = dict(section)
        self._custom_metric_models: Dict[str, Any] = {}

        # ---- Loaded models (populated by load_models) ----
        self._device: Optional[torch.device] = None
        self._devices: List[str] = []
        self._singmos = None
        self._singmos_frame_capture: Optional[SingMOSFrameCapture] = None
        self._aes = None

    # ---- introspection ----

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def aes_axes(self) -> List[str]:
        return list(self._aes_axes)

    def needs_audio_arrays(self) -> bool:
        """Whether any enabled metric requires access to decoded audio arrays."""
        return bool(self._use_singmos or self._aes_axes or self._custom_metric_cfgs)

    def needs_ref_for(self, sample_id: int, cached_refs: Optional[Dict[str, Any]]) -> bool:
        """Whether ``sample_id``'s reference audio is still needed for uncached
        ref-side metrics. Used by training to avoid decoding refs that won't
        contribute anything new."""
        if not self.needs_audio_arrays():
            return False
        c = cached_refs or {}
        if self._use_singmos and c.get("singmos") is None:
            return True
        if self._aes_axes and c.get("aes") is None:
            return True
        for name, mcfg in self._custom_metric_cfgs.items():
            spec = _CUSTOM_METRIC_BACKENDS.get(name)
            if spec is not None and spec.needs_ref:
                return True
        return False

    def active_metrics_label(self) -> List[str]:
        """Short human-readable list of enabled metrics — for startup logs."""
        out: List[str] = []
        if self._use_singmos:
            out.append("singmos")
        if self._use_aes:
            out.append("aes")
        for name in self._custom_metric_cfgs:
            out.append(f"custom[{name}]")
        return out

    # ---- lifecycle ----

    def load_models(self, devices) -> None:
        """Lazy-load all configured backends. Idempotent.

        ``devices`` may be a single ``str``/``torch.device`` (legacy
        callers) or a list of device strings. Single-GPU evaluators
        (SingMOS, AES) bind to ``devices[0]`` (the primary); custom
        backends receive the full list.
        """
        if isinstance(devices, (str, torch.device)):
            devices = [devices]
        normalized = [str(d) for d in devices]
        self._devices = normalized
        primary = normalized[0]
        self._device = torch.device(primary)
        dev = self._device

        if self._use_singmos and self._singmos is None:
            try:
                self._log(f"[SingMOS] Loading SingMOS model: {self._singmos_model}")
                self._singmos = load_singmos_predictor(
                    device=dev,
                    model_name=self._singmos_model,
                    local_model_path=self._singmos_local_model or None,
                )
                self._singmos_frame_capture = SingMOSFrameCapture()
                self._singmos_frame_capture.register(self._singmos.decoder)
                self._singmos.eval()
                self._log(f"[SingMOS] SingMOS model loaded successfully")
            except Exception as e:
                self._log(f"[SingMOS] Warning: Failed to load SingMOS model: {e}")
                self._singmos = None
                self._singmos_frame_capture = None

        if self._aes_axes and self._aes is None:
            try:
                self._log(f"[AES] Loading Audiobox Aesthetics model...")
                self._aes = load_aes_predictor(
                    device=dev,
                    ckpt_path=self._aes_local_ckpt or None,
                )
                self._log(f"[AES] Audiobox Aesthetics model loaded successfully")
            except Exception as e:
                self._log(f"[AES] Warning: Failed to load Audiobox Aesthetics model: {e}")
                self._aes = None

        for name, mcfg in self._custom_metric_cfgs.items():
            if name in self._custom_metric_models:
                continue
            spec = _CUSTOM_METRIC_BACKENDS.get(name)
            if spec is None:
                continue
            try:
                self._log(f"[Metric] Loading custom metric backend {name!r}")
                self._custom_metric_models[name] = spec.loader(self, self._devices, mcfg)
            except Exception as e:  # noqa: BLE001
                self._log(f"[Metric] Warning: failed to load custom metric {name!r}: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                self._custom_metric_models[name] = None

    def unload_models(self) -> None:
        """Drop all loaded backends and free GPU memory.

        Safe to call even when ``load_models`` was never called or partially
        failed. Mirrors the train-loop post-validation cleanup pattern.
        """
        self._singmos = None
        self._singmos_frame_capture = None
        self._aes = None
        self._custom_metric_models = {}
        gc.collect()
        # NOTE: empty_cache() disabled — it frees FSDP internal buffers still
        # referenced by NCCL collectives, causing CUDA illegal memory access.
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    # ---- main entry ----

    def evaluate(
        self,
        items: List[EvalItem],
        *,
        cached_refs: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """Run every enabled backend over ``items`` and aggregate metrics.

        Args:
            items: Per-sample evaluation records.
            cached_refs: Reference-side baselines carried over from prior calls
                (see module docstring). Use ``None`` for fresh runs.

        Returns:
            ``EvalResult`` with merged scalars/valid_counts/ref_baselines.
        """
        result = EvalResult()
        # Start ref_baselines as a copy of incoming cache so unchanged slots
        # pass through; each backend may overwrite its own entries.
        if cached_refs:
            result.ref_baselines = dict(cached_refs)

        if not items:
            return result

        self._run_singmos(items, result)
        self._run_aes(items, result)
        self._run_custom_metrics(items, result)
        return result

    def _run_custom_metrics(self, items: List[EvalItem], result: EvalResult) -> None:
        """Run every enabled custom metric backend (extension seam)."""
        for name, mcfg in self._custom_metric_cfgs.items():
            spec = _CUSTOM_METRIC_BACKENDS.get(name)
            if spec is None:
                continue
            model = self._custom_metric_models.get(name)
            try:
                spec.runner(self, items, result, model, mcfg)
            except Exception as e:  # noqa: BLE001
                self._log(f"[Metric] Warning: custom metric {name!r} failed: {e}")
                import traceback
                traceback.print_exc(file=sys.stderr)

    # ---- per-backend ----

    def _run_singmos(self, items: List[EvalItem], result: EvalResult) -> None:
        if self._singmos is None or self._singmos_frame_capture is None:
            return
        gen_audios = [it.gen_audio for it in items if it.gen_audio is not None]
        if gen_audios:
            try:
                self._log(f"[SingMOS] Computing batch SingMOS for {len(gen_audios)} generated samples...")
                gen_scores = compute_batch_singmos_scores(
                    gen_audios, self._sample_rate,
                    self._singmos, self._singmos_frame_capture, self._device,
                )
                if gen_scores:
                    avg_gen = sum(gen_scores) / len(gen_scores)
                    result.scalars["singmos_gen"] = round(avg_gen, 4)
                    self._log(f"[SingMOS] Average generated SingMOS: {avg_gen:.4f} (n={len(gen_scores)})")
            except Exception as e:
                self._log(f"[Warning] Batch SingMOS for generated audio failed: {e}")

        cached = result.ref_baselines.get("singmos") if result.ref_baselines else None
        if cached is None:
            ref_audios = [it.ref_audio for it in items if it.ref_audio is not None]
            if ref_audios:
                try:
                    self._log(f"[SingMOS] Computing batch SingMOS for {len(ref_audios)} reference samples...")
                    ref_scores = compute_batch_singmos_scores(
                        ref_audios, self._sample_rate,
                        self._singmos, self._singmos_frame_capture, self._device,
                    )
                    if ref_scores:
                        avg_ref = sum(ref_scores) / len(ref_scores)
                        result.ref_baselines["singmos"] = float(round(avg_ref, 6))
                        result.scalars["singmos_ref"] = round(avg_ref, 4)
                        self._log(f"[SingMOS] Computed reference baseline: {avg_ref:.4f} (n={len(ref_scores)})")
                except Exception as e:
                    self._log(f"[Warning] Batch SingMOS for reference audio failed: {e}")
        else:
            result.scalars["singmos_ref"] = round(float(cached), 4)
            self._log(f"[SingMOS] Reference baseline (cached): {float(cached):.4f}")

    def _run_aes(self, items: List[EvalItem], result: EvalResult) -> None:
        if self._aes is None or not self._aes_axes:
            return
        gen_audios = [it.gen_audio for it in items if it.gen_audio is not None]
        if gen_audios:
            try:
                self._log(f"[AES] Computing batch AES ({', '.join(self._aes_axes)}) "
                          f"for {len(gen_audios)} generated samples...")
                gen_scores = compute_batch_aes_scores(gen_audios, self._sample_rate, self._aes)
                for axis in self._aes_axes:
                    avg = sum(s[axis] for s in gen_scores) / len(gen_scores)
                    result.scalars[f"aes_{axis}_gen"] = round(avg, 4)
                    self._log(f"[AES] Average generated {axis}: {avg:.4f} (n={len(gen_scores)})")
            except Exception as e:
                self._log(f"[Warning] Batch AES for generated audio failed: {e}")

        cached = result.ref_baselines.get("aes") if result.ref_baselines else None
        if cached is None:
            ref_audios = [it.ref_audio for it in items if it.ref_audio is not None]
            if ref_audios:
                try:
                    self._log(f"[AES] Computing batch AES ({', '.join(self._aes_axes)}) "
                              f"for {len(ref_audios)} reference samples...")
                    ref_scores = compute_batch_aes_scores(ref_audios, self._sample_rate, self._aes)
                    aes_ref = {
                        axis: float(round(sum(s[axis] for s in ref_scores) / len(ref_scores), 6))
                        for axis in self._aes_axes
                    }
                    result.ref_baselines["aes"] = aes_ref
                    for axis, val in aes_ref.items():
                        result.scalars[f"aes_{axis}_ref"] = round(val, 4)
                    self._log(f"[AES] Computed reference baseline: {aes_ref}")
                except Exception as e:
                    self._log(f"[Warning] Batch AES for reference audio failed: {e}")
        else:
            for axis, val in cached.items():
                result.scalars[f"aes_{axis}_ref"] = round(float(val), 4)
