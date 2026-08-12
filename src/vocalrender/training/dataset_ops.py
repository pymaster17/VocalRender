"""
Dataset manipulation utilities for SVS training.

Pure functions for filtering, capping, upsampling, and building song indices
on HuggingFace :class:`datasets.Dataset` objects.  Extracted from
``train_vocalrender_svs.py`` to keep the training script focused on the training
loop itself.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from datasets import Dataset, concatenate_datasets


# ------------------------------------------------------------------
# Filtering / sampling / upsampling
# ------------------------------------------------------------------

def filter_train_datasets(
    train_ds: Dataset,
    train_datasets: List[str],
    *,
    log_fn: Optional[Callable] = None,
) -> Dataset:
    """Keep only samples whose ``dataset_name`` is in *train_datasets*.

    Args:
        train_ds: Full training dataset.
        train_datasets: Whitelist of dataset names to keep.
        log_fn: Optional callable for logging (e.g. ``tracker.print``).

    Returns:
        Filtered dataset.
    """
    if log_fn:
        log_fn(f"Filtering training data to datasets: {train_datasets}")

    original_len = len(train_ds)
    train_ds = train_ds.filter(
        lambda batch: [name in train_datasets for name in batch.get("dataset_name", [])],
        batched=True,
        batch_size=10000,
        num_proc=8,
        desc="Filtering training datasets",
    )
    if log_fn:
        log_fn(f"  Train: filtered {original_len} -> {len(train_ds)} samples")

    return train_ds


def cap_dataset_hours(
    train_ds: Dataset,
    dataset_name: str,
    max_hours: float,
    *,
    seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> Dataset:
    """Cap a named dataset subset to at most *max_hours* of audio.

    Samples are shuffled with a fixed seed, then selected greedily until
    the cumulative duration reaches *max_hours*.  All other datasets are
    kept intact.

    Args:
        train_ds: Full training dataset.
        dataset_name: Name of the dataset to cap (e.g. ``"cloudmusic"``).
        max_hours: Maximum hours of audio to keep.
        seed: Random seed for reproducible shuffling.
        log_fn: Optional callable for logging.

    Returns:
        Dataset with the named subset capped.
    """
    if log_fn:
        log_fn(f"Sampling {dataset_name} data capped at {max_hours}h...")

    # Columnar access — only reads small columns, not audio_feats
    all_names = train_ds["dataset_name"]
    all_durations = train_ds["audio_duration"]

    target_indices = []
    other_indices = []
    target_total_hours = 0.0
    for i, (name, dur) in enumerate(zip(all_names, all_durations)):
        if name == dataset_name:
            target_indices.append(i)
            target_total_hours += dur / 3600.0
        else:
            other_indices.append(i)

    if log_fn:
        log_fn(f"  {dataset_name}: {len(target_indices)} samples, {target_total_hours:.1f}h total")
        log_fn(f"  other datasets: {len(other_indices)} samples")

    max_seconds = max_hours * 3600.0
    if target_total_hours > max_hours:
        # Shuffle with fixed seed for reproducibility, then cumsum select
        import random as rnd
        rng = rnd.Random(seed)
        rng.shuffle(target_indices)

        selected = []
        cumulative = 0.0
        for idx in target_indices:
            cumulative += all_durations[idx]
            selected.append(idx)
            if cumulative >= max_seconds:
                break

        if log_fn:
            log_fn(f"  {dataset_name} sampled: {len(selected)} samples, "
                   f"{cumulative / 3600.0:.1f}h (target: {max_hours}h)")

        # Rebuild train_ds with selected target + all other datasets
        keep_indices = sorted(other_indices + selected)
        train_ds = train_ds.select(keep_indices)

        if log_fn:
            log_fn(f"  Final train set: {len(train_ds)} samples")
    else:
        if log_fn:
            log_fn(f"  {dataset_name} ({target_total_hours:.1f}h) already under cap "
                   f"({max_hours}h), keeping all")

    return train_ds


def upsample_datasets(
    train_ds: Dataset,
    upsample_config: Dict[str, int],
    *,
    log_fn: Optional[Callable] = None,
) -> Dataset:
    """Repeat dataset subsets to balance data distribution.

    Args:
        train_ds: Full training dataset.
        upsample_config: Mapping of ``dataset_name -> repeat_factor``.
            Datasets not in the mapping default to factor 1.
        log_fn: Optional callable for logging.

    Returns:
        Upsampled dataset (concatenation of repeated subsets).
    """
    if log_fn:
        log_fn(f"Applying dataset upsampling: {upsample_config}")

    all_names = train_ds["dataset_name"]
    all_durations = train_ds["audio_duration"]

    # Group indices by dataset name
    ds_groups = defaultdict(list)
    ds_hours = defaultdict(float)
    for i, (name, dur) in enumerate(zip(all_names, all_durations)):
        ds_groups[name].append(i)
        ds_hours[name] += dur / 3600.0

    if log_fn:
        log_fn("  Before upsampling:")
        for name in sorted(ds_groups):
            log_fn(f"    {name}: {len(ds_groups[name])} samples, {ds_hours[name]:.1f}h")

    # Build upsampled dataset by concatenating subsets
    parts = []
    for name, indices in sorted(ds_groups.items()):
        factor = int(upsample_config.get(name, 1))
        subset = train_ds.select(indices)
        for _ in range(max(factor, 1)):
            parts.append(subset)

    train_ds = concatenate_datasets(parts)

    if log_fn:
        # Report after upsampling
        up_names = train_ds["dataset_name"]
        up_durations = train_ds["audio_duration"]
        up_groups: Dict[str, List] = defaultdict(lambda: [0, 0.0])  # [count, hours]
        for name, dur in zip(up_names, up_durations):
            up_groups[name][0] += 1
            up_groups[name][1] += dur / 3600.0
        log_fn("  After upsampling:")
        for name in sorted(up_groups):
            cnt, hrs = up_groups[name]
            factor = int(upsample_config.get(name, 1))
            log_fn(f"    {name}: {cnt} samples, {hrs:.1f}h (x{factor})")
        log_fn(f"  Total: {len(train_ds)} samples")

    return train_ds


# ------------------------------------------------------------------
# Song index building (for prompt audio)
# ------------------------------------------------------------------

def build_song_index(
    ds: Dataset,
    *,
    log_fn: Optional[Callable] = None,
) -> Dict[str, List[int]]:
    """Build ``song_name -> [sample_indices]`` mapping from a dataset.

    Args:
        ds: HuggingFace dataset with a ``song_name`` column.
        log_fn: Optional callable for logging.

    Returns:
        Dict mapping each non-empty song name to a list of row indices.
    """
    song_names = ds["song_name"]
    song_index: Dict[str, List[int]] = {}
    for idx, sn in enumerate(song_names):
        if sn:  # Skip empty song names
            song_index.setdefault(sn, []).append(idx)

    if log_fn:
        multi_songs = {k: v for k, v in song_index.items() if len(v) >= 2}
        eligible = sum(len(v) for v in multi_songs.values())
        log_fn(f"Prompt audio: {len(song_index)} songs, "
               f"{len(multi_songs)} with >=2 segments ({eligible} prompt-eligible samples)")

    return song_index


def build_val_prompt_pool(
    train_ds: Dataset,
    val_ds: Dataset,
    train_song_index: Dict[str, List[int]],
    *,
    log_fn: Optional[Callable] = None,
) -> Tuple[Dataset, Dict[str, List[int]], Dict[str, List[int]], int]:
    """Build a combined prompt pool (train + val) for validation prompt audio.

    Validation samples can pick prompt audio from ANY same-song segment
    across the full dataset (train + val).

    Args:
        train_ds: Training dataset.
        val_ds: Validation dataset.
        train_song_index: Song index built from training data.
        log_fn: Optional callable for logging.

    Returns:
        Tuple of ``(prompt_pool_ds, prompt_pool_song_index, val_song_index,
        val_offset)`` where *val_offset* is ``len(train_ds)`` (the index
        offset for val samples in the combined pool).
    """
    val_song_names = val_ds["song_name"]
    val_song_index: Dict[str, List[int]] = {}
    for idx, sn in enumerate(val_song_names):
        if sn:
            val_song_index.setdefault(sn, []).append(idx)

    # Build combined prompt pool
    prompt_pool_ds = concatenate_datasets([train_ds, val_ds])
    prompt_pool_song_index = {sn: list(indices) for sn, indices in train_song_index.items()}
    val_offset = len(train_ds)
    for idx, sn in enumerate(val_song_names):
        if sn:
            prompt_pool_song_index.setdefault(sn, []).append(val_offset + idx)

    if log_fn:
        val_multi = {k: v for k, v in val_song_index.items() if len(v) >= 2}
        pool_multi = {k: v for k, v in prompt_pool_song_index.items() if len(v) >= 2}
        log_fn(f"Val prompt audio: {len(val_song_index)} songs, "
               f"{len(val_multi)} with >=2 segments (seed=42 for reproducibility)")
        log_fn(f"  Combined pool: {len(prompt_pool_song_index)} songs, "
               f"{len(pool_multi)} with >=2 segments ({sum(len(v) for v in pool_multi.values())} eligible)")

    return prompt_pool_ds, prompt_pool_song_index, val_song_index, val_offset
