"""
Multi-GPU audio generation for SVS validation.

Provides :func:`generate_sample_audio_svs`, which shards the validation set
across all DDP/FSDP ranks, runs batch inference on each shard, gathers results
to rank 0 via ``/dev/shm`` file exchange (avoids pickling large audio arrays
through NCCL), decodes reference audio, logs to TensorBoard, and invokes the
SVS evaluator for quality metrics.

The TB tag helpers below are private to this module — they encode the
legacy-compatible mapping used during training and intentionally differ from
the simpler ``val_audio/*`` convention used by the offline inference scripts.
"""

import gc
import io
import traceback
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange


# ============================================================
# SVS-specific TensorBoard tag mapping  (private to this module)
# ============================================================

# Map unified evaluator scalar keys -> legacy training TB tag names, so that
# existing TensorBoard histories remain continuous after the refactor. Keys
# not present here fall through to the default ``val/{scalar_key}`` tag.
_TRAIN_TB_TAG_MAP = {
    "singmos_gen": "val/singmos_generated_avg",
    "singmos_ref": "val/singmos_reference_baseline",
}


def _train_scalar_tb_tag(scalar_key: str) -> Optional[str]:
    """Translate an evaluator scalar key to a TensorBoard tag.

    Returns ``None`` if the scalar should be skipped (e.g. ``valid_count``
    entries are not plotted).
    """
    if scalar_key.endswith("_valid_count"):
        return None
    legacy = _TRAIN_TB_TAG_MAP.get(scalar_key)
    if legacy is not None:
        return legacy
    # AES: aes_{axis}_gen / aes_{axis}_ref -> val/aes_{axis}_{generated|reference}_avg
    if scalar_key.startswith("aes_") and scalar_key.endswith("_gen"):
        axis = scalar_key[len("aes_"):-len("_gen")]
        return f"val/aes_{axis}_generated_avg"
    if scalar_key.startswith("aes_") and scalar_key.endswith("_ref"):
        axis = scalar_key[len("aes_"):-len("_ref")]
        return f"val/aes_{axis}_reference_baseline"
    return f"val/{scalar_key}"


def _write_tb_scalars(writer, scalars: dict, step: int,
                      *, mode_label: Optional[str] = None) -> None:
    """Push evaluator scalar metrics to TensorBoard under legacy-compatible tags.

    When ``mode_label`` is provided, it is appended to each tag so multiple
    inference modes can coexist on the same TB run without overwriting each
    other.
    """
    if writer is None or not scalars:
        return
    for key, val in scalars.items():
        tag = _train_scalar_tb_tag(key)
        if tag is None:
            continue
        if mode_label:
            tag = f"{tag}_{mode_label}"
        try:
            writer.add_scalar(tag, val, global_step=step)
        except Exception:
            pass


# ============================================================
# generate_sample_audio_svs
# ============================================================

def generate_sample_audio_svs(model, val_ds, audio_vae, writer, step, accelerator, sample_rate=44100,
                               val_audio_samples=-1, val_tb_max_samples=5, val_max_len=300,
                               audio_eval_batch_size=16,
                               tokenizer=None, valid_interval=1000,
                               tracker=None,
                               evaluator=None,
                               eval_ref_cache: Optional[dict] = None,
                               val_song_index=None, prompt_max_frames=50,
                               prompt_source_ds=None, prompt_source_song_index=None,
                               prompt_source_val_offset=None):
    """Generate sample audio for SVS and log to TensorBoard.

    All DDP ranks participate in inference for multi-GPU speedup. Results are
    gathered to rank 0, where ``evaluator`` computes every enabled metric
    (SingMOS / AES). Per-sample reference-side baselines accumulate in
    ``eval_ref_cache`` across validation calls.
    """
    import numpy as np
    import torch.distributed as dist
    from datasets import Dataset

    from vocalrender.evaluation.audio_utils import normalize_audio, decode_reference_audio
    from vocalrender.evaluation.inference import run_inference_batch
    from vocalrender.evaluation.svs_metrics import items_from_train_buffers
    from vocalrender.evaluation.visualization import create_score_condition_figure

    rank = accelerator.rank
    world_size = accelerator.world_size
    is_distributed = dist.is_initialized() and world_size > 1

    log = tracker.print if tracker else print
    if eval_ref_cache is None:
        eval_ref_cache = {}

    if val_audio_samples == 0:
        if rank == 0:
            log(f"[SVS Audio] Audio generation disabled (val_audio_samples=0)")
        return

    # -1 means all samples
    if val_audio_samples < 0:
        num_samples = len(val_ds)
    else:
        num_samples = min(val_audio_samples, len(val_ds))

    if num_samples <= 0:
        if rank == 0:
            log("[SVS Audio] No validation samples selected for audio generation")
        return

    # --- Shard samples across ranks ---
    all_indices = list(range(num_samples))
    active_world_size = min(world_size, num_samples)
    my_indices = all_indices[rank::active_world_size] if rank < active_world_size else []

    # Precompute deterministic TB sample IDs from sample indices (rather
    # than from valid generations) — these get audio + score figures logged
    # to TensorBoard.
    if num_samples > 0 and val_tb_max_samples > 0:
        _k = min(val_tb_max_samples, num_samples)
        if _k >= num_samples:
            tb_sample_ids = set(range(num_samples))
        else:
            tb_sample_ids = {
                round(i * (num_samples - 1) / (_k - 1))
                for i in range(_k)
            }
    else:
        tb_sample_ids = set()

    if rank == 0:
        log(f"[SVS Audio] Starting multi-GPU audio generation for {num_samples} samples "
            f"at step {step} ({active_world_size}/{world_size} GPU(s) active, ~{len(my_indices)} samples/GPU)")

    unwrapped_model = accelerator.unwrap(model)

    try:
        # Set up model for generation on THIS rank
        unwrapped_model.eval()

        # Use object.__setattr__ to bypass nn.Module._modules registration.
        # This prevents FSDP from seeing audio_vae as a new submodule, which
        # would break its parameter sharding metadata.
        object.__setattr__(unwrapped_model, 'audio_vae', audio_vae.to(accelerator.device).to(torch.float32))

        my_results = []
        # The same-song prompt is drawn from the val-or-train pool here and
        # prepended at the sequence front downstream in run_inference_batch.
        _prompt_ds = prompt_source_ds if prompt_source_ds is not None else val_ds
        _prompt_song_idx = (
            prompt_source_song_index if prompt_source_song_index is not None
            else val_song_index
        )

        def _extract_pa_config(indices, *, log_extracted: bool = False):
            """Build a {"pre_extracted": [...]} prompt-audio config for ``indices``.

            Rank-symmetric: every rank enters the per-rank barrier loop the
            same number of times so FSDP collectives stay aligned.
            """
            if _prompt_song_idx is None or not indices:
                # Still must enter the barrier loop on other ranks.
                if _prompt_song_idx is not None and is_distributed:
                    for _ in range(active_world_size):
                        accelerator.barrier()
                return None
            import random as _random
            from vocalrender.evaluation.inference import _extract_batch_prompt_audio

            if prompt_source_val_offset is not None:
                _use_index_map = {local: prompt_source_val_offset + g
                                  for local, g in enumerate(indices)}
            else:
                _use_index_map = {local: g for local, g in enumerate(indices)}
            pre_extracted: list = []
            for r in range(active_world_size):
                if r == rank:
                    pre_extracted = _extract_batch_prompt_audio(
                        batch_indices=list(range(len(indices))),
                        dataset=_prompt_ds,
                        song_index=_prompt_song_idx,
                        idx_to_song={local: val_ds[g]["song_name"]
                                     for local, g in enumerate(indices)},
                        max_frames=prompt_max_frames,
                        rng=_random.Random(42),
                        index_map=_use_index_map,
                    )
                    if log_extracted:
                        log(f"[SVS Audio] Rank {rank}: pre-extracted "
                            f"{sum(1 for p in pre_extracted if p is not None)}/"
                            f"{len(pre_extracted)} prompt audio segments")
                if is_distributed:
                    accelerator.barrier()
            return {"pre_extracted": pre_extracted}

        pa_config = _extract_pa_config(my_indices, log_extracted=True)

        # Build a sub-dataset for THIS rank's shard
        my_subset = val_ds.select(my_indices) if my_indices else None
        if my_subset is not None and len(my_subset) > 0:
            # Use shared batch inference (with autocast to match training behaviour)
            # NOTE: empty_cache() disabled — it frees FSDP internal buffers
            # still referenced by NCCL collectives, causing CUDA illegal
            # memory access on the next training step. Same root cause as in
            # validation.py / runners/svs.py.
            # torch.cuda.empty_cache()
            with accelerator.summon_full_params(model), accelerator.autocast():
                # Diagnostic: verify FSDP unshard actually worked. FSDP2
                # ``fully_shard`` is in-place so there is no wrapper to peel.
                try:
                    _lyr0 = unwrapped_model.base_lm.layers[0]
                    _w = _lyr0.input_layernorm.weight
                    log(f"[SVS Audio] Rank {rank}: post-summon layer0 input_layernorm.weight shape={tuple(_w.shape)} numel={_w.numel()}")
                except Exception as _e:
                    log(f"[SVS Audio] Rank {rank}: summon diagnostic failed: {_e}")
                raw_results = run_inference_batch(
                    model=unwrapped_model,
                    val_ds=my_subset,
                    output_dir=None,
                    num_samples=len(my_subset),
                    batch_size=min(len(my_subset), audio_eval_batch_size),
                    cfg_value=2.0,
                    inference_timesteps=10,
                    max_len=val_max_len,
                    save_audio=False,
                    return_audio=True,
                    save_reference=False,
                    save_score=False,
                    sample_rate=sample_rate,
                    device=str(accelerator.device),
                    verbose=False,
                    log_fn=log if rank == 0 else (lambda msg: None),
                    prompt_audio_config=pa_config,
                )

            # Remap local sample_ids back to global indices
            for r in raw_results:
                local_id = r["sample_id"]
                r["sample_id"] = my_indices[local_id]
            my_results = raw_results

        log(f"[SVS Audio] Rank {rank}: inference returned {len(my_results)} results")

    except Exception as e:
        log(f"[Warning] Rank {rank}: batch inference failed: {e}")
        traceback.print_exc()
        my_results = []
    finally:
        object.__setattr__(unwrapped_model, 'audio_vae', None)

    # --- Gather results from all ranks to rank 0 ---
    # NOTE: We use file-based exchange instead of dist.gather_object because
    # the results contain large numpy audio arrays (~1 GB per rank) that are
    # too expensive to pickle through NCCL collective communication.
    if is_distributed:
        import pickle

        # Use /dev/shm (RAM-backed shared memory filesystem) for temp files.
        # Same speed as memory, shared across all processes on the same node,
        # and typically has ~50% of total system RAM available.
        save_dir = Path("/dev/shm")
        tmp_path = save_dir / f"_val_inference_rank{rank}_step{step}.pkl"

        # Each rank writes its results — wrapped in try/except to guarantee
        # all ranks reach the barrier below (prevents deadlock).
        write_ok = False
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(my_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            write_ok = True
            log(f"[SVS Audio] Rank {rank}: saved {len(my_results)} results to {tmp_path}")
        except Exception as e:
            log(f"[Warning] Rank {rank}: failed to write results: {e}")

        # ALL ranks must reach this barrier regardless of success/failure above
        accelerator.barrier()

        if rank == 0:
            results = []
            for r in range(world_size):
                r_path = save_dir / f"_val_inference_rank{r}_step{step}.pkl"
                try:
                    if r_path.exists():
                        with open(r_path, "rb") as f:
                            rank_results = pickle.load(f)
                        results.extend(rank_results)
                        r_path.unlink()  # cleanup
                except Exception as e:
                    log(f"[Warning] Failed to load results from rank {r}: {e}")
            results.sort(key=lambda x: x["sample_id"])
            log(f"[SVS Audio] Gathered {len(results)} results from {world_size} GPUs")

        # Keep non-zero ranks alive until rank 0 finishes reading / deleting all
        # per-rank files, otherwise they may race and remove their temp file early.
        accelerator.barrier()

        if rank != 0:
            # Non-rank-0 ranks do not perform CPU-only post-processing, but
            # they must wait until rank 0 finishes it. Otherwise they can
            # enter the next FSDP/checkpoint collective while rank 0 is still
            # running metrics, which eventually trips the NCCL watchdog.
            accelerator.barrier()
            return
    else:
        results = my_results
        if rank == 0:
            log(f"[SVS Audio] Batch inference returned {len(results)} results")

    # --- Post-process: log to TensorBoard, collect audio for batch SingMOS ---
    gen_audio_list = []   # (sample_id, audio_np) pairs for batch scoring
    # Lyric_only audio collected into a parallel list so it goes through the
    ref_audio_list = []   # (sample_id, audio_np) pairs for baseline scoring
    tb_logged = 0  # Counter for TensorBoard-logged samples
    gen_durations = []   # durations of generated audio for summary
    ref_durations = []   # durations of reference audio for summary

    # ``tb_sample_ids`` was precomputed at the top of this function from
    # raw sample indices (deterministic, evenly-spaced). Failed samples are
    # filtered at the per-sample logging step below via ``audio_np is None``.
    if rank == 0 and tb_sample_ids:
        log(
            f"[SVS Audio] TensorBoard: logging up to {len(tb_sample_ids)} "
            f"fixed sample IDs: {sorted(tb_sample_ids)}"
        )

    for r in results:
        i = r["sample_id"]
        gen_audio_np = r.get("audio_np")
        if gen_audio_np is None:
            continue

        # Log to TensorBoard for uniformly-spaced samples (by duration)
        log_to_tb = i in tb_sample_ids

        if log_to_tb:
            tag = f"svs_val_sample_{i}"
            writer.add_audio(f"{tag}/generated_audio", gen_audio_np, global_step=step, sample_rate=sample_rate)
            tb_logged += 1
        gen_durations.append(r['duration'])

        # Collect generated audio for batch SingMOS
        gen_audio_list.append((i, gen_audio_np))

        # Decode reference audio using shared utility
        ref_audio_np = None
        try:
            sample = val_ds[i]
            if "audio_feats" in sample:
                audio_vae.to(accelerator.device)
                ref_audio_np = decode_reference_audio(sample, audio_vae, str(accelerator.device), sample_rate)
                if ref_audio_np is not None:
                    ref_audio_np = normalize_audio(ref_audio_np)
                    if log_to_tb:
                        writer.add_audio(f"{tag}/reference_audio", ref_audio_np, global_step=step, sample_rate=sample_rate)
                    ref_durations.append(len(ref_audio_np) / sample_rate)
            elif "audio" in sample and isinstance(sample["audio"], dict) and "array" in sample["audio"]:
                import numpy as np
                ref_audio_np = np.array(sample["audio"]["array"], dtype=np.float32)
                ref_sr = sample["audio"].get("sampling_rate", sample_rate)
                if ref_sr != sample_rate:
                    import torchaudio.functional as F
                    ref_audio_np = F.resample(torch.from_numpy(ref_audio_np).unsqueeze(0), ref_sr, sample_rate).squeeze(0).numpy()
                ref_audio_np = normalize_audio(ref_audio_np)
                if log_to_tb:
                    writer.add_audio(f"{tag}/reference_audio", ref_audio_np, global_step=step, sample_rate=sample_rate)
        except Exception as e:
            log(f"[Warning] Failed to decode reference audio for sample {i}: {e}")

        # Collect reference audio only when an evaluator baseline/cache still
        # needs it. The evaluator decides based on its enabled backends and the
        # current cached-ref contents.
        if (
            ref_audio_np is not None
            and evaluator is not None
            and evaluator.needs_ref_for(i, eval_ref_cache)
        ):
            ref_audio_list.append((i, ref_audio_np))

        # Score figure (only for TensorBoard-logged samples).
        if log_to_tb:
            try:
                sample_for_fig = val_ds[i]
                bpm_val = sample_for_fig.get("bpm", 120)
                words_val = sample_for_fig.get("word", [])
                pitches_val = sample_for_fig.get("pitch", [])
                notes_val = sample_for_fig.get("note", [])

                if pitches_val:
                    fig = create_score_condition_figure(
                        bpm=int(bpm_val),
                        words=[str(w) for w in words_val],
                        pitches=[int(p) for p in pitches_val],
                        notes=[str(n) for n in notes_val],
                        step=step,
                    )
                    writer.add_figure(f"{tag}/score_condition", fig, global_step=step)
            except Exception as e:
                log(f"[Warning] Score figure failed for sample {i}: {e}")

    # --- Summary of generated and reference audio ---
    if gen_durations:
        log(f"[SVS Audio] Generated {len(gen_durations)} samples: "
            f"avg={sum(gen_durations)/len(gen_durations):.2f}s, "
            f"min={min(gen_durations):.2f}s, max={max(gen_durations):.2f}s, "
            f"total={sum(gen_durations):.1f}s")
    if ref_durations:
        log(f"[SVS Audio] Decoded {len(ref_durations)} reference audios: "
            f"avg={sum(ref_durations)/len(ref_durations):.2f}s, "
            f"min={min(ref_durations):.2f}s, max={max(ref_durations):.2f}s, "
            f"total={sum(ref_durations):.1f}s")
    log(f"[SVS Audio] TensorBoard: logged {tb_logged}/{len(gen_durations)} samples")

    # --- Unified metric computation via SVSEvaluator ---
    if evaluator is not None and gen_audio_list:
        try:
            items = items_from_train_buffers(
                gen_audio_list, ref_audio_list, val_ds
            )
            eval_result = evaluator.evaluate(
                items,
                cached_refs=eval_ref_cache,
            )
            eval_ref_cache.clear()
            eval_ref_cache.update(eval_result.ref_baselines)

            _write_tb_scalars(writer, eval_result.scalars, step)
            log(f"[SVS Audio] Evaluator metrics logged: {sorted(eval_result.scalars.keys())}")
        except Exception as e:
            log(f"[Warning] Evaluator metric computation failed: {e}")
            traceback.print_exc()

    # NOTE: empty_cache() disabled — it frees FSDP internal buffers still
    # referenced by NCCL collectives, causing CUDA illegal memory access on
    # the next training step. Same root cause as in validation.py /
    # runners/svs.py.
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     log(f"[SVS Audio] CUDA cache cleared after audio generation")
    if is_distributed:
        accelerator.barrier()
