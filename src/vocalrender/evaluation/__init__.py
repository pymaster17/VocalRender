"""
vocalrender.evaluation — Shared evaluation modules for SVS test and validation.

Provides unified interfaces for:
- Batch/single inference
- Audio utilities (normalization, reference audio decoding)
- Metrics (SingMOS, AES)
- Visualization (score condition figures, alignment/note plots)
- SVSEvaluator: single source of truth for training validation + standalone inference

Names are resolved lazily (PEP 562) so that inference-only users — e.g. the web
demo importing ``vocalrender.evaluation.audio_utils`` — do not pay for the
metrics / visualization imports (s3prl, matplotlib, ...).
"""

import importlib

_EXPORTS = {
    "normalize_audio": "vocalrender.evaluation.audio_utils",
    "decode_reference_audio": "vocalrender.evaluation.audio_utils",
    "run_inference_single": "vocalrender.evaluation.inference",
    "run_inference_batch": "vocalrender.evaluation.inference",
    "load_singmos_predictor": "vocalrender.evaluation.metrics",
    "SingMOSFrameCapture": "vocalrender.evaluation.metrics",
    "compute_singmos_score": "vocalrender.evaluation.metrics",
    "compute_batch_singmos_scores": "vocalrender.evaluation.metrics",
    "create_score_condition_figure": "vocalrender.evaluation.visualization",
    "SVSEvaluator": "vocalrender.evaluation.svs_metrics",
    "EvalItem": "vocalrender.evaluation.svs_metrics",
    "EvalResult": "vocalrender.evaluation.svs_metrics",
    "items_from_infer_results": "vocalrender.evaluation.svs_metrics",
    "items_from_train_buffers": "vocalrender.evaluation.svs_metrics",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
