from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SVSEvalSchedule:
    run_val_loss: bool
    run_audio_eval: bool


def should_run_interval(
    *,
    step: int,
    interval: int,
    total_steps: int,
    include_step_zero: bool = False,
) -> bool:
    if total_steps <= 0:
        return False
    last_step = total_steps - 1
    if step == last_step:
        return True
    if step == 0:
        return include_step_zero
    if interval <= 0:
        return False
    return step % interval == 0


def build_svs_eval_schedule(
    *,
    step: int,
    total_steps: int,
    valid_interval: int,
    audio_eval_interval: int,
    has_validation: bool,
    validate_at_step_zero: bool = False,
    audio_validate_at_step_zero: bool = False,
) -> SVSEvalSchedule:
    if not has_validation:
        return SVSEvalSchedule(run_val_loss=False, run_audio_eval=False)
    return SVSEvalSchedule(
        run_val_loss=should_run_interval(
            step=step,
            interval=valid_interval,
            total_steps=total_steps,
            include_step_zero=validate_at_step_zero,
        ),
        # Audio eval at step 0 is opt-in via ``audio_validate_at_step_zero``
        # (separate from val_loss step-0 gating, since audio eval is orders
        # of magnitude more expensive). Useful for debugging the audio-eval
        # path without waiting for the first ``audio_eval_interval`` hit.
        run_audio_eval=should_run_interval(
            step=step,
            interval=audio_eval_interval,
            total_steps=total_steps,
            include_step_zero=audio_validate_at_step_zero,
        ),
    )


def should_save_checkpoint(
    *,
    step: int,
    save_interval: int,
    total_steps: int,
    skip_step_zero: bool = False,
) -> bool:
    if should_run_interval(
        step=step,
        interval=save_interval,
        total_steps=total_steps,
        include_step_zero=not skip_step_zero,
    ):
        if skip_step_zero and step == 0 and step != total_steps - 1:
            return False
        return True
    return False
