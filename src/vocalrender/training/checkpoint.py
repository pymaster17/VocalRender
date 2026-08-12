"""
Checkpoint save / load utilities for VoxCPM training.

Supports both standard DDP and FSDP (ZeRO-2/3) state dict workflows.
Handles model weights, optimizer state, scheduler state, runtime resume
state, and extended SVS tokenizer.
"""

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from .accelerator import Accelerator

try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    save_dir: Path,
    rank: int = 0,
    accelerator: Optional[Accelerator] = None,
):
    """Load the latest checkpoint if it exists.

    Called by all ranks so that distributed state stays aligned.

    When ``accelerator`` is provided and FSDP is active, uses FSDP-aware
    state dict loading (collective operations — all ranks MUST call).

    Returns:
        Tuple of ``(resume_step, local_runtime_state)``.
    """
    latest = save_dir / "latest"
    if not latest.exists():
        return 0, None

    latest_folder = latest
    unwrapped = Accelerator.unwrap(model)
    lora_cfg = unwrapped.lora_config
    local_runtime_state = None
    use_fsdp = accelerator is not None and accelerator.is_fsdp

    # Under FSDP we call set_{model,optimizer}_state_dict with
    # broadcast_from_rank0=True — only rank 0's state dict is used; the
    # others can pass {}. So skip the disk read on non-zero ranks to avoid
    # 4-way concurrent reads of multi-GB optimizer.pth from shared storage
    # (the historical 10-20 min resume stall). DDP path already loads on
    # rank 0 only via the state_dict consumer, so this is FSDP-gated.
    rank0_only_io = use_fsdp

    if lora_cfg is not None:
        lora_weights_path = latest_folder / "lora_weights.safetensors"
        if not lora_weights_path.exists():
            lora_weights_path = latest_folder / "lora_weights.ckpt"
        if not lora_weights_path.exists():
            raise FileNotFoundError(f"Missing LoRA weights in checkpoint: {latest_folder}")

        if not rank0_only_io or rank == 0:
            if lora_weights_path.suffix == ".safetensors":
                state_dict = load_file(str(lora_weights_path))
            else:
                ckpt = torch.load(lora_weights_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt)
        else:
            state_dict = {}

        if use_fsdp:
            accelerator.load_model_state_dict(model, state_dict, strict=False)
        else:
            unwrapped.load_state_dict(state_dict, strict=False)

        if rank == 0:
            print(f"Loaded LoRA weights from {lora_weights_path}", file=sys.stderr)
    else:
        model_path = latest_folder / "model.safetensors"
        if not model_path.exists():
            model_path = latest_folder / "pytorch_model.bin"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model weights in checkpoint: {latest_folder}")

        if not rank0_only_io or rank == 0:
            if model_path.suffix == ".safetensors":
                state_dict = load_file(str(model_path))
            else:
                ckpt = torch.load(model_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt)
        else:
            state_dict = {}

        if use_fsdp:
            accelerator.load_model_state_dict(model, state_dict, strict=False)
        else:
            unwrapped.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(f"Loaded model weights from {model_path}", file=sys.stderr)

    optimizer_path = latest_folder / "optimizer.pth"
    if not optimizer_path.exists():
        raise FileNotFoundError(f"Missing optimizer state in checkpoint: {optimizer_path}")
    if not rank0_only_io or rank == 0:
        optim_state = torch.load(optimizer_path, map_location="cpu")
    else:
        optim_state = {}
    if use_fsdp:
        accelerator.load_optimizer_state_dict(model, optimizer, optim_state)
    else:
        optimizer.load_state_dict(optim_state)
    del optim_state
    if rank == 0:
        print(f"Loaded optimizer state from {optimizer_path}", file=sys.stderr)

    scheduler_path = latest_folder / "scheduler.pth"
    if not scheduler_path.exists():
        raise FileNotFoundError(f"Missing scheduler state in checkpoint: {scheduler_path}")
    scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
    if rank == 0:
        print(f"Loaded scheduler state from {scheduler_path}", file=sys.stderr)

    runtime_state_path = latest_folder / "runtime_state.pth"
    if not runtime_state_path.exists():
        raise FileNotFoundError(f"Missing runtime resume state in checkpoint: {runtime_state_path}")
    runtime_state = torch.load(runtime_state_path, map_location="cpu", weights_only=False)
    rank_states = runtime_state.get("rank_states")
    next_step = runtime_state.get("next_step")
    if not isinstance(rank_states, dict) or next_step is None:
        raise ValueError(
            f"Incomplete runtime_state.pth in checkpoint: expected rank_states and next_step at {runtime_state_path}"
        )
    if rank not in rank_states:
        raise ValueError(f"Missing runtime state for rank {rank} in checkpoint: {runtime_state_path}")

    local_runtime_state = rank_states[rank]
    resume_step = int(next_step)
    if rank == 0:
        print(f"Resuming from next_step {resume_step} (runtime_state)", file=sys.stderr)
    return resume_step, local_runtime_state


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    save_dir: Path,
    step: int,
    pretrained_path: str = None,
    hf_model_id: str = "",
    distribute: bool = False,
    tokenizer=None,
    runtime_state: dict = None,
    full_state_dict: dict = None,
    full_optimizer_state: dict = None,
    is_transient: bool = False,
    checkpoint_metadata: Optional[dict] = None,
):
    """Save checkpoint with extended tokenizer for SVS.

    Args:
        full_state_dict: Pre-gathered model state dict (required for FSDP,
            optional for DDP — when None, computed from model internally).
        full_optimizer_state: Pre-gathered optimizer state dict (same as above).
        is_transient: If True, tag folder as transient and prune prior
            transient folders so only the newest remains. Permanent
            checkpoints are never auto-pruned.
    """
    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    start_time = time.time()
    log(f"[Checkpoint] Starting save at step {step}...")
    if runtime_state is None:
        raise ValueError("runtime_state is required when saving checkpoints")

    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"step_{step:07d}_transient" if is_transient else f"step_{step:07d}"
    folder = save_dir / tag
    folder.mkdir(parents=True, exist_ok=True)

    # Use pre-gathered state dicts if available (FSDP path), else compute here
    if full_state_dict is not None:
        full_state = full_state_dict
    else:
        unwrapped = Accelerator.unwrap(model)
        full_state = unwrapped.state_dict()
    lora_cfg = Accelerator.unwrap(model).lora_config

    if lora_cfg is not None:
        state_dict = {k: v for k, v in full_state.items() if "lora_" in k}
        if SAFETENSORS_AVAILABLE:
            save_file(state_dict, folder / "lora_weights.safetensors")
        else:
            torch.save({"state_dict": state_dict}, folder / "lora_weights.ckpt")

        base_model_to_save = hf_model_id if distribute else (str(pretrained_path) if pretrained_path else None)
        lora_info = {
            "base_model": base_model_to_save,
            "lora_config": lora_cfg.model_dump() if hasattr(lora_cfg, "model_dump") else vars(lora_cfg),
            "svs_enabled": True,
        }
        with open(folder / "lora_config.json", "w", encoding="utf-8") as f:
            json.dump(lora_info, f, indent=2, ensure_ascii=False)
        log("[Checkpoint] Saved LoRA weights")
    else:
        state_dict = {k: v for k, v in full_state.items() if not k.startswith("audio_vae.")}
        if SAFETENSORS_AVAILABLE:
            save_file(state_dict, folder / "model.safetensors")
        else:
            torch.save({"state_dict": state_dict}, folder / "pytorch_model.bin")
        log(f"[Checkpoint] Saved model weights ({len(state_dict)} keys)")

        if pretrained_path:
            pretrained_dir = Path(pretrained_path)
            files_to_copy = [
                "config.json",
                "audiovae.pth",
                "audiovae.safetensors",
            ]
            for fname in files_to_copy:
                src = pretrained_dir / fname
                if src.exists():
                    shutil.copy2(src, folder / fname)

    # Save extended tokenizer (with SVS tokens)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(folder))
        log("[Checkpoint] Saved SVS tokenizer")

    if checkpoint_metadata is not None:
        with open(folder / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(checkpoint_metadata, f, indent=2, ensure_ascii=False)
        log("[Checkpoint] Saved training metadata")

    log("[Checkpoint] Saving optimizer state (this may take a while)...")
    optim_state_to_save = full_optimizer_state if full_optimizer_state is not None else optimizer.state_dict()
    torch.save(optim_state_to_save, folder / "optimizer.pth")
    del optim_state_to_save
    log("[Checkpoint] Saved optimizer state")

    log("[Checkpoint] Saving scheduler state...")
    torch.save(scheduler.state_dict(), folder / "scheduler.pth")
    log("[Checkpoint] Saved scheduler state")

    if runtime_state is not None:
        torch.save(runtime_state, folder / "runtime_state.pth")
        log("[Checkpoint] Saved runtime resume state")

    # Point ``latest`` at the new checkpoint via a relative symlink.
    # This avoids copying 20-30 GB of model/optimizer weights every save
    # and works across NFS / GPFS mounts (relative target stays valid if
    # the parent directory is moved).
    latest_link = save_dir / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            if latest_link.is_dir() and not latest_link.is_symlink():
                shutil.rmtree(latest_link)  # clean up old copytree-style dirs
            else:
                latest_link.unlink()
        latest_link.symlink_to(tag)  # relative: latest -> step_0050000
        log(f"[Checkpoint] Updated 'latest' symlink -> {tag}")
    except Exception as e:
        log(f"[Checkpoint] Warning: failed to update latest symlink at {latest_link}: {e}")

    if is_transient:
        for entry in save_dir.iterdir():
            if (
                entry.is_dir()
                and not entry.is_symlink()
                and entry.name.endswith("_transient")
                and entry.name != tag
            ):
                try:
                    shutil.rmtree(entry)
                    log(f"[Checkpoint] Pruned old transient checkpoint: {entry.name}")
                except Exception as e:
                    log(f"[Checkpoint] Warning: failed to prune {entry}: {e}")

    elapsed = time.time() - start_time
    log(f"[Checkpoint] Save completed in {elapsed:.1f}s")
