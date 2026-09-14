"""
Training resume utilities for VoxCPM.

Captures and restores per-rank dataloader position, Python/Torch/CUDA RNG
states, and sample-level progress for exact training resume after
checkpoint recovery.
"""

import random
from typing import Optional

import torch

from .accelerator import Accelerator


def set_train_loader_epoch(train_loader, epoch: int):
    """Update sampler epoch for both fixed and dynamic batching loaders."""
    sampler = getattr(train_loader, "sampler", None)
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)


def capture_local_runtime_state(
    accelerator: Accelerator,
    *,
    data_epoch: int,
    batches_seen_in_epoch: int,
    samples_seen: int,
) -> dict:
    """Capture local dataloader progress and RNG state for resumable training."""
    state = {
        "rank": int(accelerator.rank),
        "data_epoch": int(data_epoch),
        "batches_seen_in_epoch": int(batches_seen_in_epoch),
        "samples_seen": int(samples_seen),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state(accelerator.device).cpu()
    else:
        state["cuda_rng_state"] = None
    return state


def gather_runtime_state(
    runtime_state: dict,
    accelerator: Accelerator,
    next_step: int,
) -> Optional[dict]:
    """Gather per-rank runtime state so resume can recover each rank's local position."""
    payload = {
        "version": 1,
        "next_step": int(next_step),
    }
    if accelerator.world_size > 1:
        import torch.distributed as dist

        gathered = [None for _ in range(accelerator.world_size)] if accelerator.rank == 0 else None
        dist.gather_object(runtime_state, object_gather_list=gathered, dst=0)
        if accelerator.rank != 0:
            return None
        payload["rank_states"] = {int(state["rank"]): state for state in gathered if state is not None}
    else:
        payload["rank_states"] = {int(accelerator.rank): runtime_state}
    return payload


def gather_global_sample_progress(
    local_samples_seen: int,
    accelerator: Accelerator,
) -> Optional[dict]:
    """Gather exact sample progress across unique data-parallel replicas."""
    payload = {
        "rank": int(accelerator.rank),
        "samples_seen": int(local_samples_seen),
    }
    if accelerator.world_size > 1:
        import torch.distributed as dist

        gathered = [None for _ in range(accelerator.world_size)] if accelerator.rank == 0 else None
        dist.gather_object(payload, object_gather_list=gathered, dst=0)
        if accelerator.rank != 0:
            return None
        rank_states = {int(state["rank"]): int(state["samples_seen"]) for state in gathered if state is not None}
    else:
        rank_states = {int(accelerator.rank): int(local_samples_seen)}

    shard_group_size = accelerator.fsdp_sharding_group_size if accelerator.is_hybrid_shard else 1
    unique_data_ranks = range(0, accelerator.world_size, shard_group_size)
    global_samples_seen = sum(rank_states.get(rank, 0) for rank in unique_data_ranks)
    return {
        "global_samples_seen": int(global_samples_seen),
    }


def restore_local_runtime_state(
    runtime_state: Optional[dict],
    accelerator: Accelerator,
):
    """Restore main-process RNG state for resume before rebuilding dataloader iterators."""
    if not runtime_state:
        return
    python_rng_state = runtime_state.get("python_rng_state")
    if python_rng_state is not None:
        random.setstate(python_rng_state)
    torch_rng_state = runtime_state.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state.cpu())
    cuda_rng_state = runtime_state.get("cuda_rng_state")
    if cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state.cpu(), accelerator.device)


def restore_train_iterator(
    train_loader,
    *,
    data_epoch: int,
    batches_seen_in_epoch: int,
):
    """Rebuild the train iterator and advance it to the saved batch position.

    For dynamic-batch loaders the sampler yields precomputed, deterministic
    index lists, so we fast-forward by skipping at the index level
    (``set_start_batch``) without decoding any data. For the fixed-batch
    fallback (no ``_dynamic_sampler``) we replay ``next()`` as before.
    """
    set_train_loader_epoch(train_loader, data_epoch)

    sampler = getattr(train_loader, "_dynamic_sampler", None)
    if sampler is not None and hasattr(sampler, "set_start_batch"):
        sampler.set_start_batch(batches_seen_in_epoch)  # raises if out of range
        return iter(train_loader)

    train_iter = iter(train_loader)
    for _ in range(batches_seen_in_epoch):
        try:
            next(train_iter)
        except StopIteration as exc:
            raise RuntimeError(
                "Saved train dataloader position exceeds available batches for the restored epoch "
                f"(epoch={data_epoch}, batches_seen_in_epoch={batches_seen_in_epoch})."
            ) from exc
    return train_iter
