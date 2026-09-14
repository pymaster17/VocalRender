"""
AudioVAE lazy loading for evaluation.

Loads AudioVAE (V1 or V2) from a pretrained checkpoint directory on demand,
keeping it separate from the training model to save GPU memory.
"""

import json
from pathlib import Path

import torch

from vocalrender.modules.audiovae import (
    AudioVAE,
    AudioVAEConfig,
    AudioVAEV2,
    AudioVAEConfigV2,
)

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def load_audio_vae_for_eval(pretrained_path: str) -> torch.nn.Module:
    """Load AudioVAE lazily for audio evaluation only.

    Auto-detects V1 vs V2 architecture from ``config.json`` and loads
    the corresponding VAE checkpoint in eval mode with all parameters
    frozen.

    Args:
        pretrained_path: Path to the pretrained model directory containing
            ``config.json`` and ``audiovae.safetensors`` (or ``.pth``).

    Returns:
        A frozen :class:`AudioVAE` or :class:`AudioVAEV2` in ``float32``.
    """
    pretrained_dir = Path(pretrained_path)
    config_path = pretrained_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {pretrained_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    architecture = str(model_cfg.get("architecture", "voxcpm")).lower()
    audio_vae_cfg = model_cfg.get("audio_vae_config") or {}

    if architecture == "voxcpm2":
        audio_vae = AudioVAEV2(
            config=AudioVAEConfigV2.model_validate(audio_vae_cfg) if audio_vae_cfg else AudioVAEConfigV2()
        )
    else:
        audio_vae = AudioVAE(
            config=AudioVAEConfig.model_validate(audio_vae_cfg) if audio_vae_cfg else AudioVAEConfig()
        )

    audiovae_path = pretrained_dir / "audiovae.safetensors"
    if not audiovae_path.exists():
        audiovae_path = pretrained_dir / "audiovae.pth"
    if not audiovae_path.exists():
        raise FileNotFoundError(
            f"AudioVAE checkpoint not found under {pretrained_dir}: "
            "expected audiovae.safetensors or audiovae.pth"
        )

    if audiovae_path.suffix == ".safetensors":
        if not SAFETENSORS_AVAILABLE:
            raise RuntimeError("safetensors is required to load audiovae.safetensors")
        vae_state_dict = load_file(str(audiovae_path), device="cpu")
    else:
        checkpoint = torch.load(audiovae_path, map_location="cpu", weights_only=True)
        vae_state_dict = checkpoint.get("state_dict", checkpoint)

    audio_vae.load_state_dict(vae_state_dict, strict=True)
    audio_vae = audio_vae.to(torch.float32).eval()
    for param in audio_vae.parameters():
        param.requires_grad_(False)
    return audio_vae
