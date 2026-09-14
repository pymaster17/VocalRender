"""
SVS validation entry point for VoxCPM training.

Provides :func:`validate_svs`, which orchestrates:
  1. Loss evaluation across the full validation set (all ranks).
  2. Optional multi-GPU audio generation and quality-metric computation,
     delegated to :func:`~vocalrender.training.val_audio.generate_sample_audio_svs`.

Audio generation logic, TensorBoard tag mapping, and per-sample reference
decoding all live in :mod:`vocalrender.training.val_audio`.
"""

import gc
import io
import traceback
from typing import Optional

import torch

from .val_audio import generate_sample_audio_svs  # noqa: F401 — re-exported for callers


# ============================================================
# Loss reduction helper
# ============================================================

def _loss_reduction_denominator(key: str, processed: dict) -> torch.Tensor:
    """Return the denominator implied by each loss reduction."""
    loss_mask = processed["loss_mask"]
    device = loss_mask.device
    denom = loss_mask.sum(dtype=torch.float64)
    if key == "loss/diff":
        patch_size = int(processed["audio_feats"].shape[2])
        return denom * patch_size
    if key == "loss/stop":
        return denom
    batch_size = int(processed["text_tokens"].shape[0])
    return torch.tensor(float(batch_size), device=device, dtype=torch.float64)


# ============================================================
# validate_svs
# ============================================================

def validate_svs(model, val_loader, accelerator, tracker, lambdas,
                 writer=None, step=0, val_ds=None,
                 sample_rate=44100, tokenizer=None, valid_interval=1000,

                 run_loss_eval=True, run_audio_eval=True,
                 val_audio_samples=-1, val_tb_max_samples=5, val_max_len=300,
                 audio_eval_batch_size=16,
                 evaluator=None,
                 eval_ref_cache: Optional[dict] = None,
                 val_song_index=None, prompt_max_frames=50,
                 prompt_source_ds=None, prompt_source_song_index=None,
                 prompt_source_val_offset=None,
                 load_audio_vae_fn=None):
    """Validate on SVS data, optionally generate audio samples and compute quality scores.

    The ``eval_ref_cache`` dict is mutated in-place with freshly computed
    reference baselines so subsequent calls can reuse them.
    """
    import torch.distributed as dist

    model.eval()

    if run_loss_eval:
        loss_sums = {}
        loss_denoms = {}

        with torch.no_grad():
            for batch in val_loader:
                processed = {
                    "text_tokens": batch["text_tokens"].to(accelerator.device),
                    "text_mask": batch["text_mask"].to(accelerator.device),
                    "audio_feats": batch["audio_feats"].to(accelerator.device),
                    "audio_mask": batch["audio_mask"].to(accelerator.device),
                    "loss_mask": batch["loss_mask"].to(accelerator.device),
                    "labels": batch["labels"].to(accelerator.device),
                }

                with accelerator.autocast():
                    outputs = model(
                        processed["text_tokens"],
                        processed["text_mask"],
                        processed["audio_feats"],
                        processed["audio_mask"],
                        processed["loss_mask"],
                        processed["labels"],
                        progress=0.0,
                        sample_generate=False,
                    )

                for key, value in outputs.items():
                    if not key.startswith("loss/"):
                        continue
                    denom = _loss_reduction_denominator(key, processed)
                    denom_val = float(denom.item()) if hasattr(denom, "item") else float(denom)
                    if denom_val <= 0:
                        continue
                    loss_sums[key] = loss_sums.get(
                        key, torch.zeros((), device=accelerator.device, dtype=torch.float64),
                    ) + value.detach().to(torch.float64) * denom
                    loss_denoms[key] = loss_denoms.get(
                        key, torch.zeros((), device=accelerator.device, dtype=torch.float64),
                    ) + denom

        tracked_loss_keys = sorted({
            key for key in set(loss_sums) | set(loss_denoms) | set(lambdas)
            if key.startswith("loss/")
        })
        if tracked_loss_keys:
            for key in tracked_loss_keys:
                loss_sums.setdefault(
                    key,
                    torch.zeros((), device=accelerator.device, dtype=torch.float64),
                )
                loss_denoms.setdefault(
                    key,
                    torch.zeros((), device=accelerator.device, dtype=torch.float64),
                )
            for key in tracked_loss_keys:
                accelerator.all_reduce(loss_sums[key], op=dist.ReduceOp.SUM)
                accelerator.all_reduce(loss_denoms[key], op=dist.ReduceOp.SUM)

            val_metrics = {}
            total_loss = 0.0
            for key in tracked_loss_keys:
                denom = loss_denoms[key].item()
                if denom <= 0:
                    continue
                mean_loss = (loss_sums[key] / loss_denoms[key]).item()
                val_metrics[key] = mean_loss
                total_loss += lambdas.get(key, 1.0) * mean_loss
            if val_metrics:
                val_metrics["loss/total"] = total_loss
                tracker.log_metrics(val_metrics, split="val")
            elif accelerator.rank == 0:
                tracker.print(f"[val] Step {step}: validation loader produced zero effective loss weight")
        elif accelerator.rank == 0:
            tracker.print(f"[val] Step {step}: no validation batches produced loss values")
    elif accelerator.rank == 0:
        tracker.print(f"[val] Step {step}: skipping val_loss evaluation")

    # Generate sample audio — all ranks participate for multi-GPU speedup.
    # Results are gathered to rank 0 for TensorBoard logging & metric computation.
    accelerator.barrier()
    if eval_ref_cache is None:
        eval_ref_cache = {}
    if run_audio_eval and val_ds is not None and load_audio_vae_fn is not None and val_audio_samples != 0:
        # Rank 0 loads evaluator backends onto GPU; other ranks only run the
        # VAE for audio decoding of their shard.
        if accelerator.rank == 0 and evaluator is not None:
            evaluator.load_models(accelerator.device)
        _audio_vae = None
        try:
            _audio_vae = load_audio_vae_fn()
            if accelerator.rank == 0:
                tracker.print("[AudioVAE] Loaded lazily for audio evaluation")
            generate_sample_audio_svs(
                model, val_ds, _audio_vae, writer, step, accelerator, sample_rate,
                val_audio_samples=val_audio_samples,
                val_tb_max_samples=val_tb_max_samples,
                val_max_len=val_max_len,
                audio_eval_batch_size=audio_eval_batch_size,
                tokenizer=tokenizer, valid_interval=valid_interval, tracker=tracker,
                evaluator=evaluator,
                eval_ref_cache=eval_ref_cache,
                val_song_index=val_song_index,
                prompt_max_frames=prompt_max_frames,
                prompt_source_ds=prompt_source_ds,
                prompt_source_song_index=prompt_source_song_index,
                prompt_source_val_offset=prompt_source_val_offset,
            )
        except Exception as e:
            if accelerator.rank == 0:
                tracker.print(f"[Warning] Failed to generate sample audio: {e}")
                buf = io.StringIO()
                traceback.print_exc(file=buf)
                tracker.print(buf.getvalue())
        finally:
            if _audio_vae is not None:
                try:
                    _audio_vae.to("cpu")
                except Exception as e:
                    if accelerator.rank == 0:
                        tracker.print(f"[AudioVAE] Warning: failed to move back to CPU: {e}")
            # Drop all evaluator backends to free GPU memory for training.
            if accelerator.rank == 0 and evaluator is not None:
                evaluator.unload_models()
            del _audio_vae
            gc.collect()
    else:
        if run_audio_eval:
            missing = []
            if val_ds is None:
                missing.append("val_ds")
            if load_audio_vae_fn is None:
                missing.append("audio_vae_loader")
            if missing and accelerator.rank == 0:
                tracker.print(f"[Warning] Skip audio generation: missing {', '.join(missing)}")
        elif accelerator.rank == 0:
            tracker.print(f"[val] Step {step}: skipping heavy audio evaluation")
    accelerator.barrier()

    # Restore training mode for ALL ranks (important for DDP consistency)
    model.train()

    # FSDP2: explicitly reshard every FSDPModule after validation. The eval
    # forward (no_grad) goes through FSDP2 hooks but eval-mode + no_grad can
    # leave the per-module unshard/reshard state machine in a configuration
    # that breaks the next training backward (observed as CUDA illegal
    # memory access on the first post-val training step).
    if getattr(accelerator, "is_fsdp", False):
        try:
            from torch.distributed.fsdp import FSDPModule
            for m in model.modules():
                if isinstance(m, FSDPModule):
                    m.reshard()
            if torch.cuda.is_available():
                torch.cuda.synchronize(accelerator.device)
        except Exception as _e:
            if accelerator.rank == 0:
                tracker.print(f"[val] FSDP2 post-val reshard cleanup failed: {_e}")

    # NOTE: empty_cache() disabled — it was freeing FSDP internal buffers
    # still referenced by NCCL collectives, causing CUDA illegal memory access.
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    if accelerator.rank == 0:
        tracker.print(f"[val] Validation complete, eval models unloaded, model restored to training mode")
