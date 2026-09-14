"""
Training utilities for VoxCPM fine-tuning.

This package mirrors the training mechanics used in the minicpm-audio
tooling while relying solely on local audio-text datasets managed via
the HuggingFace ``datasets`` library.

Only core names used by ``from vocalrender.training import ...`` in training
scripts are re-exported here, and they are resolved lazily (PEP 562): the
inference path imports ``vocalrender.training.svs_data`` for syllable helpers
and must not drag in argbind / distributed training modules.  All other
symbols should be imported directly from their submodules (e.g.
``from vocalrender.training.svs_data import ...``).
"""

import importlib

_EXPORTS = {
    "Accelerator": "vocalrender.training.accelerator",
    "SVSTrainConfig": "vocalrender.training.config",
    "TrainingTracker": "vocalrender.training.tracker",
    "load_audio_text_datasets": "vocalrender.training.data",
    "HFVoxCPMDataset": "vocalrender.training.data",
    "build_dataloader": "vocalrender.training.data",
    "BatchProcessor": "vocalrender.training.data",
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
