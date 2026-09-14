"""
GPU memory diagnostics, timing instrumentation, and per-rank diagnostics
for VoxCPM distributed training.
"""

from typing import Optional

import torch

from .accelerator import Accelerator
from .tracker import TrainingTracker


# ------------------------------------------------------------------ #
# Formatting helpers
# ------------------------------------------------------------------ #

def format_bytes(num_bytes) -> str:
    """Format a byte count into a human-readable string (KiB/MiB/GiB/TiB)."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def format_bytes_per_token(num_bytes, num_tokens: int) -> str:
    """Format a byte/token ratio."""
    if num_tokens <= 0:
        return "n/a"
    return f"{format_bytes(num_bytes / num_tokens)}/token"


# ------------------------------------------------------------------ #
# Device sync for accurate timing
# ------------------------------------------------------------------ #

def sync_device_for_timing(device: torch.device):
    """Synchronize CUDA device for accurate wall-clock timing."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


# ------------------------------------------------------------------ #
# Tensor storage analysis
# ------------------------------------------------------------------ #

def iter_tensor_leaves(obj):
    """Recursively yield all :class:`torch.Tensor` leaves from a nested structure."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_tensor_leaves(value)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from iter_tensor_leaves(value)


def unique_tensor_storage_bytes(tensors, device: torch.device) -> int:
    """Count unique GPU storage bytes across a collection of tensors."""
    total = 0
    seen_storages = set()

    for tensor in tensors:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            continue
        if tensor.device != device:
            continue

        try:
            storage = tensor.untyped_storage()
            storage_ptr = storage.data_ptr()
            storage_bytes = storage.nbytes()
        except Exception:
            storage_ptr = tensor.data_ptr()
            storage_bytes = tensor.numel() * tensor.element_size()

        if storage_ptr in seen_storages:
            continue
        seen_storages.add(storage_ptr)
        total += storage_bytes

    return total


# ------------------------------------------------------------------ #
# GPU memory breakdown logging
# ------------------------------------------------------------------ #

def log_cuda_memory_breakdown_once(
    tracker: TrainingTracker,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_tensors: dict,
    device: torch.device,
    step: int,
):
    """Log a one-time GPU memory attribution breakdown after the first optimizer step."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    model_bytes = unique_tensor_storage_bytes(
        list(model.parameters()) + list(model.buffers()),
        device,
    )
    grad_bytes = unique_tensor_storage_bytes(
        [param.grad for param in model.parameters() if param.grad is not None],
        device,
    )
    optimizer_state_bytes = unique_tensor_storage_bytes(
        iter_tensor_leaves(optimizer.state),
        device,
    )
    data_bytes = unique_tensor_storage_bytes(
        iter_tensor_leaves(batch_tensors),
        device,
    )

    total_allocated = torch.cuda.memory_allocated(device)
    total_reserved = torch.cuda.memory_reserved(device)
    unattributed_bytes = max(
        0,
        total_allocated - (model_bytes + grad_bytes + optimizer_state_bytes + data_bytes),
    )

    tracker.print(
        "[GPU Memory] "
        f"step {step} after first optimizer step | "
        f"model={format_bytes(model_bytes)}, "
        f"grad={format_bytes(grad_bytes)}, "
        f"optimizer_state={format_bytes(optimizer_state_bytes)}, "
        f"data(current micro-batch)={format_bytes(data_bytes)}, "
        f"allocated={format_bytes(total_allocated)}, "
        f"reserved={format_bytes(total_reserved)}, "
        f"unattributed={format_bytes(unattributed_bytes)}"
    )


def log_cuda_peak_memory_once(
    tracker: TrainingTracker,
    probe: dict,
    device: torch.device,
    step: int,
):
    """Log a one-time GPU peak memory analysis after the first training step."""
    if device.type != "cuda" or not torch.cuda.is_available() or not probe:
        return

    baseline_allocated = probe["baseline_allocated"]
    baseline_reserved = probe["baseline_reserved"]
    forward_peak = probe["forward_peak_allocated"]
    backward_peak = probe["backward_peak_allocated"]
    step_end_allocated = probe["step_end_allocated"]
    step_end_reserved = probe["step_end_reserved"]

    forward_delta = max(0, forward_peak - baseline_allocated)
    backward_delta = max(0, backward_peak - baseline_allocated)
    step_end_delta = max(0, step_end_allocated - baseline_allocated)

    tracker.print(
        "[GPU Peak] "
        f"step {step} first training step | "
        f"baseline_allocated={format_bytes(baseline_allocated)}, "
        f"baseline_reserved={format_bytes(baseline_reserved)}, "
        f"forward_peak={format_bytes(forward_peak)} "
        f"(+{format_bytes(forward_delta)}, "
        f"{probe['forward_peak_tokens']} tokens, "
        f"{format_bytes_per_token(forward_delta, probe['forward_peak_tokens'])}), "
        f"backward_peak={format_bytes(backward_peak)} "
        f"(+{format_bytes(backward_delta)}, "
        f"{probe['backward_peak_tokens']} tokens, "
        f"{format_bytes_per_token(backward_delta, probe['backward_peak_tokens'])}), "
        f"step_end_allocated={format_bytes(step_end_allocated)} "
        f"(+{format_bytes(step_end_delta)}), "
        f"step_end_reserved={format_bytes(step_end_reserved)}"
    )


# ------------------------------------------------------------------ #
# Per-rank diagnostics
# ------------------------------------------------------------------ #

def new_rank_diag_interval() -> dict:
    """Create a fresh per-rank diagnostics accumulation dict."""
    return {
        "data": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optim": 0.0,
        "step": 0.0,
        "tokens": 0,
        "micro_steps": 0,
        "steps": 0,
        "max_step": 0.0,
        "max_forward": 0.0,
        "max_backward": 0.0,
        "max_batch_size": 0,
        "max_seq_len": 0,
        "max_micro_tokens": 0,
    }


def update_rank_diag_interval(interval: dict, step_timing: dict):
    """Accumulate one step's timing into the diagnostics interval."""
    for key in ("data", "forward", "backward", "optim", "step"):
        interval[key] += step_timing[key]
    interval["tokens"] += step_timing["tokens"]
    interval["micro_steps"] += step_timing["micro_steps"]
    interval["steps"] += 1
    interval["max_step"] = max(interval["max_step"], step_timing["step"])
    interval["max_forward"] = max(interval["max_forward"], step_timing["forward"])
    interval["max_backward"] = max(interval["max_backward"], step_timing["backward"])
    interval["max_batch_size"] = max(interval["max_batch_size"], step_timing["max_batch_size"])
    interval["max_seq_len"] = max(interval["max_seq_len"], step_timing["max_seq_len"])
    interval["max_micro_tokens"] = max(interval["max_micro_tokens"], step_timing["max_micro_tokens"])


def gather_rank_diag_interval(
    interval: dict,
    accelerator: Accelerator,
) -> Optional[list]:
    """Gather per-rank diagnostics from all ranks to rank 0."""
    payload = dict(interval)
    payload["rank"] = int(accelerator.rank)

    if accelerator.world_size <= 1:
        return [payload]

    import torch.distributed as dist

    gathered = [None for _ in range(accelerator.world_size)] if accelerator.rank == 0 else None
    dist.gather_object(payload, object_gather_list=gathered, dst=0)
    if accelerator.rank != 0:
        return None
    return [item for item in gathered if item is not None]


def format_rank_diag_summary(step: int, gathered: list) -> str:
    """Format a human-readable per-rank diagnostics summary line."""
    parts = []
    for payload in sorted(gathered, key=lambda item: item["rank"]):
        interval_steps = max(int(payload.get("steps", 0)), 1)
        interval_step_total = max(float(payload.get("step", 0.0)), 1e-12)
        avg_step_ms = 1000.0 * float(payload.get("step", 0.0)) / interval_steps
        max_step_ms = 1000.0 * float(payload.get("max_step", 0.0))
        avg_bwd_ms = 1000.0 * float(payload.get("backward", 0.0)) / interval_steps
        max_bwd_ms = 1000.0 * float(payload.get("max_backward", 0.0))
        tokens_per_s = float(payload.get("tokens", 0)) / interval_step_total
        max_batch_size = int(payload.get("max_batch_size", 0))
        max_seq_len = int(payload.get("max_seq_len", 0))
        parts.append(
            f"r{payload['rank']}: "
            f"step={avg_step_ms:.0f}/{max_step_ms:.0f}ms "
            f"bwd={avg_bwd_ms:.0f}/{max_bwd_ms:.0f}ms "
            f"tok/s={tokens_per_s:.0f} "
            f"BxL<={max_batch_size}x{max_seq_len}"
        )
    return f"[RankDiag] step {step}: " + " | ".join(parts)
