from __future__ import annotations

import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from .checkpoint import load_checkpoint, save_checkpoint
from .resume import capture_local_runtime_state, gather_runtime_state
from .tracker import TrainingTracker


# ---------------------------------------------------------------------------
# Precision resolution
# ---------------------------------------------------------------------------


def dtype_name(dtype: Optional[torch.dtype]) -> str:
    """Return a short human-readable name for a torch dtype."""
    if dtype is None:
        return "none"
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return str(dtype)


def resolve_training_precision(train_precision: str) -> dict:
    """Parse a precision mode string and return a config dict.

    Supported modes (case-insensitive, with common aliases):

    =========  ============  ============  ==================================
    Mode       param_dtype   amp_dtype     Description
    =========  ============  ============  ==================================
    fp32       float32       —             Full precision, no autocast
    amp_bf16   float32       bfloat16      FP32 params + BF16 autocast
    bf16       bfloat16      bfloat16      BF16 params + BF16 autocast
    amp_fp16   float32       float16       FP32 params + FP16 autocast
    =========  ============  ============  ==================================
    """
    precision = str(train_precision).strip().lower()
    aliases = {
        "fp32": "fp32",
        "float32": "fp32",
        "amp_bf16": "amp_bf16",
        "bf16_amp": "amp_bf16",
        "mixed_bf16": "amp_bf16",
        "bf16_mixed": "amp_bf16",
        "bf16": "bf16",
        "amp_fp16": "amp_fp16",
        "fp16_amp": "amp_fp16",
        "mixed_fp16": "amp_fp16",
        "fp16_mixed": "amp_fp16",
    }
    mode = aliases.get(precision)
    if mode is None:
        raise ValueError(
            "Unsupported train_precision={!r}. Expected one of: "
            "'fp32', 'amp_bf16', 'bf16', 'amp_fp16'.".format(train_precision)
        )

    if mode == "fp32":
        return {
            "mode": mode,
            "amp": False,
            "amp_dtype": None,
            "param_dtype": torch.float32,
            "description": "fp32 params + no autocast",
        }
    if mode == "amp_bf16":
        return {
            "mode": mode,
            "amp": True,
            "amp_dtype": torch.bfloat16,
            "param_dtype": torch.float32,
            "description": "fp32 params + bf16 autocast",
        }
    if mode == "bf16":
        return {
            "mode": mode,
            "amp": True,
            "amp_dtype": torch.bfloat16,
            "param_dtype": torch.bfloat16,
            "description": "bf16 params + bf16 autocast",
        }
    return {
        "mode": mode,
        "amp": True,
        "amp_dtype": torch.float16,
        "param_dtype": torch.float32,
        "description": "fp32 params + fp16 autocast",
    }


# ---------------------------------------------------------------------------
# LR scheduler factory
# ---------------------------------------------------------------------------


def create_lr_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    *,
    warmup_steps: int,
    total_training_steps: int,
    allowed_types: Optional[Sequence[str]] = None,
) -> _LRScheduler:
    """Create a learning rate scheduler.

    ``scheduler_type`` is one of ``"cosine"``, ``"constant"``, ``"inverse_sqrt"``.
    ``allowed_types`` restricts the accepted set for callers that only
    support a subset of schedulers.
    """
    scheduler_type = scheduler_type.lower().strip()

    if allowed_types is not None and scheduler_type not in allowed_types:
        raise ValueError(
            f"Scheduler type '{scheduler_type}' is not allowed. "
            f"Allowed types: {list(allowed_types)}"
        )

    if scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    elif scheduler_type == "inverse_sqrt":
        def _inverse_sqrt_lr(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return (float(warmup_steps) / float(current_step)) ** 0.5

        return LambdaLR(optimizer, lr_lambda=_inverse_sqrt_lr)
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        raise ValueError(
            f"Unknown scheduler type '{scheduler_type}'. "
            f"Supported: 'cosine', 'constant', 'inverse_sqrt'."
        )


def scheduler_description(
    scheduler_type: str,
    *,
    learning_rate: float,
    warmup_steps: int,
    total_training_steps: int = 0,
) -> str:
    """Return a one-line description suitable for ``tracker.print()``."""
    scheduler_type = scheduler_type.lower().strip()
    tag = scheduler_type.upper()
    base = f"Using {tag} learning rate scheduler (lr={learning_rate}, warmup={warmup_steps}"
    if scheduler_type == "cosine":
        return base + f", total={total_training_steps})"
    return base + ")"


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def install_checkpoint_signal_handler(
    save_dir: Path,
    resume_state: dict,
    rank: int,
    start_step: int,
) -> None:
    """Install SIGTERM / SIGINT handlers that print resume hints and exit.

    The handler does **not** save a checkpoint at signal time — it calls
    ``os._exit(0)`` to skip atexit hooks (which can deadlock under DDP/FSDP)
    and points the user at the latest periodic checkpoint instead.

    *resume_state* is a mutable dict whose ``"step"`` key the training loop
    keeps up-to-date so the handler can report the current step.
    """

    def _handler(signum, frame):
        try:
            cur_step = int(resume_state.get("step", start_step))
        except Exception:
            cur_step = start_step
        if rank == 0:
            latest = save_dir / "latest"
            print(
                f"Signal {signum} received at step {cur_step}. "
                f"Skipping signal-time checkpoint save; resume from the latest periodic checkpoint: {latest}",
                file=sys.stderr,
            )
        os._exit(0)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


# ---------------------------------------------------------------------------
# Runtime context + checkpoint orchestration
# ---------------------------------------------------------------------------


@dataclass
class TrainingRuntimeContext:
    accelerator: object
    save_dir: Path
    tb_dir: Path
    writer: object | None
    tracker: TrainingTracker


@dataclass
class LoopProgressState:
    data_epoch: int = 0
    batches_seen_in_epoch: int = 0
    local_samples_seen: int = 0


@dataclass
class ResumeContext:
    start_step: int
    resume_runtime_state: Optional[dict]
    signal_state: dict


def create_training_runtime(
    *,
    accelerator,
    save_path: str,
    tensorboard: str = "",
    log_filename: str = "train.log",
) -> TrainingRuntimeContext:
    save_dir = Path(save_path)
    tb_dir = Path(tensorboard) if tensorboard else save_dir / "logs"

    if accelerator.rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
    accelerator.barrier()

    writer = SummaryWriter(log_dir=str(tb_dir)) if accelerator.rank == 0 else None
    tracker = TrainingTracker(
        writer=writer,
        log_file=str(save_dir / log_filename),
        rank=accelerator.rank,
    )
    return TrainingRuntimeContext(
        accelerator=accelerator,
        save_dir=save_dir,
        tb_dir=tb_dir,
        writer=writer,
        tracker=tracker,
    )


def build_optimizer_and_scheduler(
    model,
    *,
    learning_rate: float,
    weight_decay: float,
    scheduler_type: str,
    warmup_steps: int,
    total_training_steps: int,
    allowed_scheduler_types: Optional[Sequence[str]] = None,
):
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = create_lr_scheduler(
        optimizer,
        scheduler_type,
        warmup_steps=warmup_steps,
        total_training_steps=total_training_steps,
        allowed_types=allowed_scheduler_types,
    )
    description = scheduler_description(
        scheduler_type,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_training_steps=total_training_steps,
    )
    return optimizer, scheduler, description


def load_resume_context(
    model,
    optimizer,
    scheduler,
    *,
    runtime: TrainingRuntimeContext,
) -> ResumeContext:
    start_step, resume_runtime_state = load_checkpoint(
        model,
        optimizer,
        scheduler,
        runtime.save_dir,
        rank=runtime.accelerator.rank,
        accelerator=runtime.accelerator,
    )
    runtime.accelerator.barrier()
    if start_step > 0 and runtime.accelerator.rank == 0:
        runtime.tracker.print(f"Resuming training from step {start_step}")

    signal_state = {"step": start_step}
    install_checkpoint_signal_handler(
        runtime.save_dir,
        signal_state,
        runtime.accelerator.rank,
        start_step,
    )
    return ResumeContext(
        start_step=start_step,
        resume_runtime_state=resume_runtime_state,
        signal_state=signal_state,
    )


def gather_resume_payload(
    *,
    runtime: TrainingRuntimeContext,
    loop_state: LoopProgressState,
    next_step: int,
) -> Optional[dict]:
    return gather_runtime_state(
        capture_local_runtime_state(
            runtime.accelerator,
            data_epoch=loop_state.data_epoch,
            batches_seen_in_epoch=loop_state.batches_seen_in_epoch,
            samples_seen=loop_state.local_samples_seen,
        ),
        runtime.accelerator,
        next_step=next_step,
    )


def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    *,
    runtime: TrainingRuntimeContext,
    step: int,
    pretrained_path: str,
    hf_model_id: str,
    distribute: bool,
    tokenizer,
    loop_state: LoopProgressState,
    is_transient: bool = False,
    checkpoint_metadata: Optional[dict] = None,
) -> None:
    runtime_state = gather_resume_payload(
        runtime=runtime,
        loop_state=loop_state,
        next_step=step + 1,
    )
    model_sd = runtime.accelerator.model_state_dict(model)
    optim_sd = runtime.accelerator.optimizer_state_dict(model, optimizer)
    if runtime.accelerator.rank == 0:
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            runtime.save_dir,
            step,
            pretrained_path,
            hf_model_id,
            distribute,
            tokenizer,
            runtime_state=runtime_state,
            full_state_dict=model_sd,
            full_optimizer_state=optim_sd,
            is_transient=is_transient,
            checkpoint_metadata=checkpoint_metadata,
        )


def close_training_runtime(runtime: TrainingRuntimeContext) -> None:
    if runtime.writer is not None:
        runtime.writer.close()
