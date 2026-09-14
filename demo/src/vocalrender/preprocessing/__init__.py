"""
SVS data preprocessing package.

Provides reusable library code for converting raw SVS data (folder-based or
JSON-based annotations) into preprocessed Arrow format for training.

Key public API:

* :class:`SVSPreprocessor` / :func:`create_lightweight_preprocessor` —
  audio VAE encoding and SVS token sequence construction.
* :func:`rebuild_svs_prompt` — reconstruct SVS prompt text from metadata.
* :func:`load_config`, :func:`load_all_datasets` — dataset loading.
* :func:`build_text_tensor`, :func:`estimate_duration_from_notes` — text
  tensor construction and duration estimation.
* :func:`process_and_save`, :func:`process_and_save_multigpu` — Arrow
  dataset writing (single-GPU and multi-GPU).
"""

from .svs_preprocessor import SVSPreprocessor, create_lightweight_preprocessor
from .svs_prompt import rebuild_svs_prompt
from .data_loaders import (
    load_config,
    load_samples_from_folder,
    load_samples_from_json_file,
    load_samples_from_weak_json_file,
    load_all_datasets,
    reconstruct_lyric_text,
)
from .text_tensor import build_text_tensor, estimate_duration_from_notes
from .arrow_writer import process_and_save, process_and_save_multigpu

__all__ = [
    "SVSPreprocessor",
    "create_lightweight_preprocessor",
    "rebuild_svs_prompt",
    "load_config",
    "load_samples_from_folder",
    "load_samples_from_json_file",
    "load_samples_from_weak_json_file",
    "load_all_datasets",
    "reconstruct_lyric_text",
    "build_text_tensor",
    "estimate_duration_from_notes",
    "process_and_save",
    "process_and_save_multigpu",
]
