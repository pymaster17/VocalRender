"""
Model architecture detection and class selection for VocalRender training.

Reads ``config.json`` from a pretrained checkpoint directory and returns
the appropriate model class and LoRA config class.  This avoids duplicating
the architecture detection logic across training and inference scripts.
"""

import json
import os
from dataclasses import dataclass

from vocalrender.model import VoxCPMModel, VoxCPM2Model
from vocalrender.model.voxcpm import LoRAConfig as LoRAConfigV1
from vocalrender.model.voxcpm2 import LoRAConfig as LoRAConfigV2


@dataclass
class ModelClasses:
    """Container for architecture-specific classes."""

    model_cls: type
    """VoxCPMModel or VoxCPM2Model."""

    lora_config_cls: type
    """LoRAConfig matching the detected architecture."""

    arch: str = "voxcpm"
    """Architecture string (``"voxcpm"`` or ``"voxcpm2"``)."""


def detect_model_classes(pretrained_path: str) -> ModelClasses:
    """Detect architecture from *pretrained_path*/config.json and return classes.

    Parameters
    ----------
    pretrained_path : str
        Path to a pretrained checkpoint directory.  Must contain a
        ``config.json`` with an optional ``"architecture"`` field.

    Returns
    -------
    ModelClasses
        Dataclass with ``model_cls``, ``lora_config_cls`` and ``arch``.
    """
    config_path = os.path.join(pretrained_path, "config.json")
    with open(config_path) as f:
        arch = json.load(f).get("architecture", "voxcpm").lower()

    if arch == "voxcpm2":
        model_cls = VoxCPM2Model
        lora_config_cls = LoRAConfigV2
    else:
        model_cls = VoxCPMModel
        lora_config_cls = LoRAConfigV1

    return ModelClasses(
        model_cls=model_cls,
        lora_config_cls=lora_config_cls,
        arch=arch,
    )
