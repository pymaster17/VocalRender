"""
Unified inference interface for SVS evaluation.

Provides both single-sample and batch inference that can be called from
test scripts and validation routines identically.
"""

import inspect
import sys
import time
import json
from pathlib import Path
from typing import Callable, List, Dict, Optional, Any

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from vocalrender.evaluation.audio_utils import normalize_audio, decode_reference_audio


def run_inference_single(model, svs_prompt: str, cfg_value: float = 2.0,
                         inference_timesteps: int = 10, max_len: int = 2000,
                         verbose: bool = True) -> Optional[np.ndarray]:
    """Run inference for a single SVS prompt.

    Args:
        model: VoxCPM model (in eval mode, on device).
        svs_prompt: SVS prompt string.
        cfg_value: CFG guidance scale.
        inference_timesteps: Number of diffusion steps.
        max_len: Maximum generation length (frames).
        verbose: Whether to show tqdm progress bar.

    Returns:
        Normalized 1-D numpy array of audio, or None on failure.
    """
    with torch.no_grad():
        generated = model.generate(
            target_text=svs_prompt,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            max_len=max_len,
            verbose=verbose,
        )

    if generated is None or len(generated) == 0:
        return None

    if isinstance(generated, torch.Tensor):
        audio_np = generated.cpu().float().numpy().flatten()
    else:
        audio_np = np.array(generated, dtype=np.float32).flatten()

    return normalize_audio(audio_np)


def run_inference_batch(
    model,
    val_ds,
    output_dir: Optional[Path] = None,
    num_samples: int = -1,
    batch_size: int = 4,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    max_len: int = 2000,
    save_audio: bool = True,
    return_audio: bool = False,
    save_reference: bool = False,
    save_score: bool = False,
    save_audio_max: int = -1,
    sample_rate: int = 44100,
    device: str = "cuda",
    verbose: bool = True,
    log_fn=None,
    prompt_audio_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run batch inference on a preprocessed validation dataset.

    Samples are grouped by estimated audio duration before batching, which
    minimises wasted computation from waiting for the longest sample in each
    batch.

    This function is used by **both** the standalone test script and the
    in-training validation loop.

    Args:
        model: VoxCPM model (in eval mode, on device).
        val_ds: HuggingFace Dataset with at least 'svs_prompt' and 'item_name' columns.
        output_dir: Directory to write wav/score files.  Required when *save_audio* is True.
        num_samples: Number of samples to generate (-1 for all).
        batch_size: Samples per GPU batch.
        cfg_value: CFG guidance scale.
        inference_timesteps: Diffusion steps.
        max_len: Maximum generation length (frames).
        save_audio: If True, write generated wav files to *output_dir*.
        return_audio: If True, include 'audio_np' and optionally 'ref_audio_np' numpy
                      arrays in each result dict (used by validation for TensorBoard).
        save_reference: If True, also decode + save reference audio.
        save_score: If True, render staff-notation score PNGs via lilypond.
        sample_rate: Audio sample rate.
        device: Device string.
        verbose: Whether to show progress bars / per-sample logs.
        log_fn: Optional logging function (e.g. ``tracker.print``).  Falls back
                to ``print(..., file=sys.stderr)``.
        prompt_audio_config: Optional dict to enable training-format prompt audio.
            Keys:
              - dataset: HF Dataset with preprocessed 'audio_feats' and 'audio_mask' columns
              - song_index: dict mapping song_name -> list of sample indices
              - max_frames: int (default 50)
              - seed: int (default 42, for reproducible prompt selection)

    Returns:
        List of result dicts, each containing at least:
            sample_id, item_name, svs_prompt (truncated), duration,
            and optionally generated_path, reference_path, score_path, audio_np, ref_audio_np.
    """
    import random as _random

    log = log_fn or (lambda msg: print(msg, file=sys.stderr))

    if save_audio:
        if output_dir is None:
            raise ValueError("output_dir is required when save_audio=True")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(val_ds)
    if num_samples > 0:
        total_samples = min(num_samples, total_samples)

    log(f"[SVS Batch Inference] Running batch inference on {total_samples} samples (batch_size={batch_size})")

    # ---- Prompt audio setup ----
    pa_dataset = None
    pa_song_index = None
    pa_max_frames = 50
    pa_rng = None
    pa_idx_to_song: Dict[int, str] = {}
    pa_index_map: Optional[Dict[int, int]] = None
    pa_pre_extracted: Optional[List[Optional[torch.Tensor]]] = None

    if prompt_audio_config is not None:
        # --- Mode 1: Pre-extracted features (no Arrow access needed) ---
        if "pre_extracted" in prompt_audio_config:
            pa_pre_extracted = prompt_audio_config["pre_extracted"]
            log(f"[SVS Batch Inference] Using {sum(1 for p in pa_pre_extracted if p is not None)}"
                f"/{len(pa_pre_extracted)} pre-extracted prompt audio segments")
        elif "dataset" not in prompt_audio_config:
            # Only gender ids supplied — skip Mode 2 extraction.
            pass
        else:
            # --- Mode 2: On-the-fly extraction from dataset ---
            pa_dataset = prompt_audio_config["dataset"]
            pa_song_index = prompt_audio_config["song_index"]
            pa_max_frames = prompt_audio_config.get("max_frames", 50)
            pa_seed = prompt_audio_config.get("seed", 42)
            pa_rng = _random.Random(pa_seed)
            pa_index_map = prompt_audio_config.get("index_map")  # local → global

            if pa_index_map is not None:
                # Build idx_to_song from the *inference* dataset's song_name column.
                # This is needed when the inference dataset is a sharded subset
                # whose local indices don't match the prompt source dataset.
                if "song_name" in (val_ds.column_names if hasattr(val_ds, 'column_names') else []):
                    for local_idx, sn in enumerate(val_ds["song_name"]):
                        if sn:
                            pa_idx_to_song[local_idx] = sn
                else:
                    log("[SVS Batch Inference] Warning: index_map provided but "
                        "inference dataset has no 'song_name' column; "
                        "falling back to song_index reverse lookup")
                    pa_index_map = None  # Disable index_map, fall through below

            if pa_index_map is None:
                # Default: build reverse lookup from song_index (indices match dataset)
                for song_name, indices in pa_song_index.items():
                    for idx in indices:
                        pa_idx_to_song[idx] = song_name

            multi_songs = {k: v for k, v in pa_song_index.items() if len(v) >= 2}
            eligible = sum(len(v) for v in multi_songs.values())
            log(f"[SVS Batch Inference] Prompt audio: {len(pa_song_index)} songs, "
                f"{len(multi_songs)} with ≥2 segments ({eligible} eligible samples), "
                f"max_frames={pa_max_frames}, seed={pa_seed}")

    # ---- Step 1: Prepare lightweight metadata ----
    col_names = val_ds.column_names if hasattr(val_ds, 'column_names') else []
    has_precomputed_duration = "estimated_duration" in col_names
    has_svs_prompt_col = "svs_prompt" in col_names

    if not has_svs_prompt_col:
        if total_samples > 0 and "svs_prompt" not in val_ds[0]:
            raise ValueError("Dataset missing 'svs_prompt' column. Please re-run preprocessing.")
        has_svs_prompt_col = True

    prep_columns = ["svs_prompt", "item_name"]
    if has_precomputed_duration:
        prep_columns.append("estimated_duration")

    # Pick which rows to infer on. ``num_samples <= 0`` keeps the whole split;
    # otherwise draw a deterministic random subset (fixed seed → reproducible).
    # Previously this was the first ``total_samples`` rows, which — because the
    # preprocessed Arrow keeps rows in shard order — was a length-biased cluster
    # (e.g. the first 100 muse rows are all ~2 s), making the eval set
    # unrepresentative.
    n_total = len(val_ds)
    if num_samples > 0 and total_samples < n_total:
        indices = sorted(_random.Random(42).sample(range(n_total), total_samples))
    else:
        indices = list(range(total_samples))
    prep_data = val_ds.select(indices).select_columns(prep_columns)

    all_samples_data = []
    for pos, orig in enumerate(tqdm(indices, desc="Preparing samples", disable=not verbose)):
        try:
            svs_prompt = prep_data[pos].get("svs_prompt", "")
            if not svs_prompt:
                log(f"[Sample {orig}] Warning: Empty prompt text")
                continue

            item_name = prep_data[pos].get("item_name", f"sample_{orig:05d}")
            if isinstance(item_name, bytes):
                item_name = item_name.decode('utf-8')

            if has_precomputed_duration:
                est_dur = float(prep_data[pos]["estimated_duration"])
            else:
                # Legacy data without estimated_duration: use a conservative default
                est_dur = 10.0

            all_samples_data.append({
                "index": orig,
                "svs_prompt": svs_prompt,
                "item_name": item_name,
                "estimated_duration": est_dur,
            })
        except Exception as e:
            log(f"[Sample {orig}] Error decoding prompt: {e}")
            continue

    if not all_samples_data:
        log("[SVS Batch Inference] No valid samples found!")
        return []

    # ---- Step 2: Sort by duration for efficient batching ----
    all_samples_data.sort(key=lambda x: x["estimated_duration"])
    durations = [s["estimated_duration"] for s in all_samples_data]
    log(f"[SVS Batch Inference] Duration range: {min(durations):.2f}s - {max(durations):.2f}s")

    # ---- Step 3: Batched generation ----
    results: List[Dict[str, Any]] = []
    score_tasks = []
    start_time = time.time()

    num_batches = (len(all_samples_data) + batch_size - 1) // batch_size

    # ``save_audio_max`` caps how many wav/score artefacts get written to
    # disk across the whole run; eval metrics still consume every result.
    saved_ids: set = set()

    def _can_save_to_disk() -> bool:
        return save_audio_max < 0 or len(saved_ids) < save_audio_max

    for batch_idx in tqdm(range(num_batches), desc="Batch Processing", disable=not verbose):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_samples_data))
        batch_data = all_samples_data[start_idx:end_idx]

        batch_prompts = [d["svs_prompt"] for d in batch_data]
        batch_item_names = [d["item_name"] for d in batch_data]
        batch_indices = [d["index"] for d in batch_data]
        batch_est_durations = [d["estimated_duration"] for d in batch_data]

        # Always log batch progress (even when verbose=False / tqdm disabled)
        min_dur = min(batch_est_durations)
        max_dur = max(batch_est_durations)
        log(f"[Batch {batch_idx+1}/{num_batches}] {len(batch_prompts)} samples "
            f"(est. {min_dur:.1f}s - {max_dur:.1f}s)")

        # ---- Extract prompt audio feats for this batch ----
        batch_prompt_audio = None
        if pa_pre_extracted is not None:
            # Use pre-extracted features (indexed by original sample index)
            batch_prompt_audio = [pa_pre_extracted[idx] for idx in batch_indices]
        elif pa_dataset is not None:
            batch_prompt_audio = _extract_batch_prompt_audio(
                batch_indices, pa_dataset, pa_song_index, pa_idx_to_song,
                pa_max_frames, pa_rng, index_map=pa_index_map,
            )
        if batch_prompt_audio is None or any(p is None for p in batch_prompt_audio):
            missing = (
                batch_indices
                if batch_prompt_audio is None
                else [
                    idx for idx, prompt in zip(batch_indices, batch_prompt_audio)
                    if prompt is None
                ]
            )
            raise RuntimeError(
                "Prompt-conditioned inference requires prompt audio for every "
                f"sample; missing prompt for indices {missing[:10]}"
                + ("..." if len(missing) > 10 else "")
            )
        try:
            with torch.no_grad():
                generate_kwargs = dict(
                    target_texts=batch_prompts,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    max_len=max_len,
                    verbose=verbose,
                    prompt_audio_feats=batch_prompt_audio,
                )
                audio_tensors = model.generate_batch(**generate_kwargs)

            for j, (audio_tensor, item_name, idx, prompt, est_dur) in enumerate(
                zip(audio_tensors, batch_item_names, batch_indices, batch_prompts, batch_est_durations)
            ):
                if audio_tensor is None or audio_tensor.numel() == 0:
                    log(f"[Sample {idx}] Warning: Generation failed")
                    continue

                audio_np = normalize_audio(audio_tensor.float().numpy().flatten())
                actual_duration = len(audio_np) / sample_rate

                result: Dict[str, Any] = {
                    "sample_id": idx,
                    "item_name": item_name,
                    "svs_prompt": prompt[:200],
                    "duration": actual_duration,
                    "estimated_duration": est_dur,
                }

                if return_audio:
                    result["audio_np"] = audio_np

                if save_audio and output_dir is not None and _can_save_to_disk():
                    import soundfile as sf
                    gen_path = output_dir / f"{item_name}_generated.wav"
                    sf.write(str(gen_path), audio_np, sample_rate)
                    result["generated_path"] = str(gen_path)
                    saved_ids.add(idx)
                    if verbose:
                        log(f"[Sample {idx}] Saved: {gen_path} ({actual_duration:.2f}s)")

                # Score / reference saves are paired with the generated wav
                # so the on-disk artefact set stays coherent under the cap.
                if save_score and output_dir is not None and idx in saved_ids:
                    full_sample_for_score = val_ds[idx]
                    score_tasks.append((
                        int(full_sample_for_score.get("bpm", 120)),
                        [str(w) for w in full_sample_for_score.get("word", [])],
                        [int(p) for p in full_sample_for_score.get("pitch", [])],
                        [str(n) for n in full_sample_for_score.get("note", [])],
                        [int(p) for p in full_sample_for_score.get("pitch2word", [])],
                        str(output_dir / f"{item_name}_score"),
                    ))

                # Reference audio
                if save_reference or return_audio:
                    full_sample = val_ds[idx]
                    ref_audio_np = decode_reference_audio(full_sample, model.audio_vae, device, sample_rate)
                    if ref_audio_np is not None:
                        ref_audio_np = normalize_audio(ref_audio_np)
                        if return_audio:
                            result["ref_audio_np"] = ref_audio_np
                        if (save_reference and save_audio and output_dir is not None
                                and idx in saved_ids):
                            import soundfile as sf
                            ref_path = output_dir / f"{item_name}_reference.wav"
                            sf.write(str(ref_path), ref_audio_np, sample_rate)
                            result["reference_path"] = str(ref_path)
                            result["reference_duration"] = len(ref_audio_np) / sample_rate

                results.append(result)

        except Exception as e:
            log(f"[Batch {batch_idx+1}] Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    elapsed_time = time.time() - start_time

    # Sort back by original sample_id
    results.sort(key=lambda x: x["sample_id"])



    log(f"[SVS Batch Inference] Done! {len(results)} samples in {elapsed_time:.2f}s "
        f"({elapsed_time/max(len(results),1):.2f}s/sample)")

    # Render scores in parallel
    if save_score and score_tasks:
        from vocalrender.utils.score_rendering import render_scores_parallel
        score_results = render_scores_parallel(score_tasks)
        for r in results:
            if output_dir is not None:
                score_key = str(output_dir / f"{r['item_name']}_score")
                if score_key in score_results and score_results[score_key]:
                    r["score_path"] = score_results[score_key]


    return results


def _extract_batch_prompt_audio(
    batch_indices: List[int],
    dataset,
    song_index: Dict[str, List[int]],
    idx_to_song: Dict[int, str],
    max_frames: int,
    rng,
    index_map: Optional[Dict[int, int]] = None,
) -> List[Optional[torch.Tensor]]:
    """Extract prompt audio features for a batch of samples.

    For each sample, selects a different segment from the same song,
    extracts the audio region from its preprocessed VAE latents,
    and crops to *max_frames*. Returns a list of ``[T, P, D]`` tensors
    (or ``None`` when no prompt is available for that sample).

    When *index_map* is provided, ``batch_indices`` are local indices
    into a sharded subset, while ``dataset`` and ``song_index`` refer
    to the full (un-sharded) dataset.  The map translates local → global
    so the current sample is properly excluded from candidates.

    This mirrors the extraction logic in
    ``HFPreprocessedSVSDataset._prepend_prompt_audio``.
    """
    result: List[Optional[torch.Tensor]] = []

    for idx in batch_indices:
        song_name = idx_to_song.get(idx)
        if not song_name:
            result.append(None)
            continue

        candidates = song_index.get(song_name, [])
        # Use index_map to get the global index for proper exclusion
        exclude_idx = index_map[idx] if index_map is not None else idx
        other = [c for c in candidates if c != exclude_idx]
        if not other:
            result.append(None)
            continue

        prompt_idx = rng.choice(other)
        prompt_item = dataset[prompt_idx]

        prompt_audio = torch.tensor(prompt_item["audio_feats"], dtype=torch.float32)
        prompt_audio_mask_raw = torch.tensor(prompt_item["audio_mask"], dtype=torch.int32)
        audio_positions = torch.where(prompt_audio_mask_raw == 1)[0]
        if len(audio_positions) == 0:
            result.append(None)
            continue

        start_pos = audio_positions[0].item()
        end_pos = audio_positions[-1].item() + 1
        region = prompt_audio[start_pos:end_pos]  # [T_prompt, P, D]

        T_prompt = region.shape[0]
        if T_prompt > max_frames:
            crop_start = rng.randint(0, T_prompt - max_frames)
            region = region[crop_start:crop_start + max_frames]

        result.append(region)

    return result




# ---------------------------------------------------------------------------
# Multi-GPU batch inference
# ---------------------------------------------------------------------------

def run_inference_batch_multi_gpu(
    load_model_fn: Callable,
    devices: List[str],
    val_ds,
    output_dir: Optional[Path] = None,
    num_samples: int = -1,
    batch_size: int = 4,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    max_len: int = 2000,
    save_audio: bool = True,
    return_audio: bool = False,
    save_reference: bool = False,
    save_score: bool = False,
    sample_rate: int = 44100,
    verbose: bool = True,
    log_fn=None,
    temperature: float = 1.0,
    temperature_mode: str = "scale",
    fsq_temperature: float = 0.0,
    prompt_audio_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Multi-GPU variant of :func:`run_inference_batch`.

    Instead of receiving a pre-loaded model, this takes a *model factory*
    ``load_model_fn(device_str) -> model`` and a list of device strings.
    One model is loaded per device and a work-stealing thread pool dispatches
    batches to idle GPUs automatically.

    The return value has the same schema as :func:`run_inference_batch`.
    """
    import random as _random
    from typing import Callable  # noqa: F811 — redefined for older Pythons
    from vocalrender.evaluation.multi_gpu import MultiGPUInferencePool

    log = log_fn or (lambda msg: print(msg, file=sys.stderr))

    if save_audio:
        if output_dir is None:
            raise ValueError("output_dir is required when save_audio=True")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(val_ds)
    if num_samples > 0:
        total_samples = min(num_samples, total_samples)

    log(f"[SVS Multi-GPU Inference] {total_samples} samples, "
        f"{len(devices)} GPU(s), batch_size={batch_size}")

    # ---- Prompt audio setup ----
    pa_dataset = None
    pa_song_index = None
    pa_max_frames = 50
    pa_rng = None
    pa_idx_to_song: Dict[int, str] = {}
    pa_index_map: Optional[Dict[int, int]] = None

    if prompt_audio_config is not None:
        pa_dataset = prompt_audio_config["dataset"]
        pa_song_index = prompt_audio_config["song_index"]
        pa_max_frames = prompt_audio_config.get("max_frames", 50)
        pa_seed = prompt_audio_config.get("seed", 42)
        pa_rng = _random.Random(pa_seed)
        pa_index_map = prompt_audio_config.get("index_map")  # local val idx -> prompt-pool idx

        if pa_index_map is not None and "song_name" in (val_ds.column_names if hasattr(val_ds, "column_names") else []):
            for local_idx, sn in enumerate(val_ds["song_name"]):
                if sn:
                    pa_idx_to_song[local_idx] = sn
        else:
            for song_name, indices in pa_song_index.items():
                for idx in indices:
                    pa_idx_to_song[idx] = song_name

    # ---- Step 1: Prepare lightweight metadata (same as single-GPU) ----
    col_names = val_ds.column_names if hasattr(val_ds, "column_names") else []
    has_precomputed_duration = "estimated_duration" in col_names

    prep_columns = ["svs_prompt", "item_name"]
    if has_precomputed_duration:
        prep_columns.append("estimated_duration")

    # Deterministic random subset (fixed seed) instead of the first N rows —
    # the preprocessed Arrow is in shard order, so the first N were a
    # length-biased cluster. ``num_samples <= 0`` keeps the whole split.
    n_total = len(val_ds)
    if num_samples > 0 and total_samples < n_total:
        indices = sorted(_random.Random(42).sample(range(n_total), total_samples))
    else:
        indices = list(range(total_samples))
    prep_data = val_ds.select(indices).select_columns(prep_columns)

    all_samples_data = []
    for pos, orig in enumerate(tqdm(indices, desc="Preparing samples", disable=not verbose)):
        try:
            svs_prompt = prep_data[pos].get("svs_prompt", "")
            if not svs_prompt:
                continue
            item_name = prep_data[pos].get("item_name", f"sample_{orig:05d}")
            if isinstance(item_name, bytes):
                item_name = item_name.decode("utf-8")
            est_dur = float(prep_data[pos]["estimated_duration"]) if has_precomputed_duration else 10.0
            all_samples_data.append({
                "index": orig,
                "svs_prompt": svs_prompt,
                "item_name": item_name,
                "estimated_duration": est_dur,
            })
        except Exception as e:
            log(f"[Sample {orig}] Error: {e}")
            continue

    if not all_samples_data:
        log("[SVS Multi-GPU Inference] No valid samples found!")
        return []

    # Sort by duration for efficient batching
    all_samples_data.sort(key=lambda x: x["estimated_duration"])

    # ---- Extract prompt audio for all samples (sorted order) ----
    prompt_audio_feats_all = None
    if pa_dataset is not None:
        sorted_indices = [s["index"] for s in all_samples_data]
        prompt_audio_feats_all = _extract_batch_prompt_audio(
            sorted_indices, pa_dataset, pa_song_index, pa_idx_to_song,
            pa_max_frames, pa_rng, index_map=pa_index_map,
        )
    if prompt_audio_feats_all is None or any(p is None for p in prompt_audio_feats_all):
        sorted_indices = [s["index"] for s in all_samples_data]
        missing = (
            sorted_indices
            if prompt_audio_feats_all is None
            else [
                idx for idx, prompt in zip(sorted_indices, prompt_audio_feats_all)
                if prompt is None
            ]
        )
        raise RuntimeError(
            "Prompt-conditioned inference requires prompt audio for every "
            f"sample; missing prompt for indices {missing[:10]}"
            + ("..." if len(missing) > 10 else "")
        )

    # ---- Step 2: Create pool & dispatch ----
    generate_kwargs = dict(
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        max_len=max_len,
        verbose=False,  # suppress per-batch tqdm inside model
        temperature=temperature,
        temperature_mode=temperature_mode,
        fsq_temperature=fsq_temperature,
    )

    pool = MultiGPUInferencePool(
        load_model_fn=load_model_fn,
        devices=devices,
        batch_size=batch_size,
        generate_kwargs=generate_kwargs,
        log_fn=log,
    )

    prompts = [s["svs_prompt"] for s in all_samples_data]
    start_time = time.time()

    try:
        # ``all_samples_data`` is already duration-sorted above. Preserve that
        # order so each dispatched batch keeps similar expected audio length;
        # re-sorting by prompt string length here breaks the duration bucketing
        # and can create pathological long-tail batches.
        raw_results = pool.generate_all(
            prompts,
            prompt_audio_feats_list=prompt_audio_feats_all,
            sort_by=None,
        )
    finally:
        pool.shutdown()

    elapsed_time = time.time() - start_time

    # ---- Step 3: Collect results ----
    # We need the first model for reference audio decoding.  Since pool has
    # been shut down we reload one lightweight reference — but it's easier to
    # just keep a reference to one model before shutdown.  For simplicity,
    # we load the model on the first device.
    ref_model = load_model_fn(devices[0]) if (save_reference or return_audio) else None

    results: List[Dict[str, Any]] = []
    score_tasks = []

    for prompt_idx, audio_tensor in raw_results:
        sd = all_samples_data[prompt_idx]
        idx = sd["index"]
        item_name = sd["item_name"]

        if audio_tensor is None or (hasattr(audio_tensor, "numel") and audio_tensor.numel() == 0):
            if verbose:
                log(f"[Sample {idx}] Warning: Generation failed")
            continue

        audio_np = normalize_audio(audio_tensor.float().numpy().flatten())
        actual_duration = len(audio_np) / sample_rate

        result: Dict[str, Any] = {
            "sample_id": idx,
            "item_name": item_name,
            "svs_prompt": sd["svs_prompt"][:200],
            "duration": actual_duration,
            "estimated_duration": sd["estimated_duration"],
        }

        if return_audio:
            result["audio_np"] = audio_np

        if save_audio and output_dir is not None:
            import soundfile as sf
            gen_path = output_dir / f"{item_name}_generated.wav"
            sf.write(str(gen_path), audio_np, sample_rate)
            result["generated_path"] = str(gen_path)

        if save_score and output_dir is not None:
            full_sample_for_score = val_ds[idx]
            score_tasks.append((
                int(full_sample_for_score.get("bpm", 120)),
                [str(w) for w in full_sample_for_score.get("word", [])],
                [int(p) for p in full_sample_for_score.get("pitch", [])],
                [str(n) for n in full_sample_for_score.get("note", [])],
                [int(p) for p in full_sample_for_score.get("pitch2word", [])],
                str(output_dir / f"{item_name}_score"),
            ))

        if (save_reference or return_audio) and ref_model is not None:
            full_sample = val_ds[idx]
            ref_audio_np = decode_reference_audio(
                full_sample, ref_model.audio_vae, devices[0], sample_rate
            )
            if ref_audio_np is not None:
                ref_audio_np = normalize_audio(ref_audio_np)
                if return_audio:
                    result["ref_audio_np"] = ref_audio_np
                if save_reference and save_audio and output_dir is not None:
                    import soundfile as sf
                    ref_path = output_dir / f"{item_name}_reference.wav"
                    sf.write(str(ref_path), ref_audio_np, sample_rate)
                    result["reference_path"] = str(ref_path)
                    result["reference_duration"] = len(ref_audio_np) / sample_rate

        results.append(result)

    # Sort back by original sample_id
    results.sort(key=lambda x: x["sample_id"])



    log(f"[SVS Multi-GPU Inference] Done! {len(results)} samples in {elapsed_time:.2f}s "
        f"({elapsed_time / max(len(results), 1):.2f}s/sample)")

    # Render scores in parallel
    if save_score and score_tasks:
        from vocalrender.utils.score_rendering import render_scores_parallel
        score_results = render_scores_parallel(score_tasks)
        for r in results:
            if output_dir is not None:
                score_key = str(output_dir / f"{r['item_name']}_score")
                if score_key in score_results and score_results[score_key]:
                    r["score_path"] = score_results[score_key]


    return results
