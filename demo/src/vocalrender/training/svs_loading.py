"""Arrow dataset loading and train/validation splitting for SVS training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from datasets import Dataset, concatenate_datasets, load_from_disk


def _resolve_shard_paths(preprocessed_dir: Path) -> Optional[List[str]]:
    """Return the shard files in ``load_from_disk`` order, or ``None``.

    ``save_to_disk`` records the shard list (and their order) in
    ``state.json``'s ``_data_files``. Loading those shards individually and
    concatenating them reproduces *exactly* the row order ``load_from_disk``
    yields — but only when there is no ``_indices_data_files`` remap (i.e. the
    dataset wasn't saved after a ``.select``/``.shuffle``). When either the
    state file is missing or an indices remap is present, return ``None`` so
    callers fall back to the canonical (slow) ``load_from_disk``.
    """
    state_path = preprocessed_dir / "state.json"
    if not state_path.exists():
        return None
    with state_path.open() as f:
        state = json.load(f)
    if state.get("_indices_data_files"):
        return None
    data_files = state.get("_data_files")
    if not data_files:
        return None
    return [str(preprocessed_dir / entry["filename"]) for entry in data_files]


def fast_load_from_disk(preprocessed_dir: str) -> Dataset:
    """Drop-in for ``load_from_disk`` that skips its full-corpus overhead.

    On the 71-shard / 229 GB SVS corpus, ``datasets.load_from_disk`` takes
    ~385 s and ~24 GB RSS before returning (NFS-pathological concat/validate of
    every shard). Concatenating the shards via per-shard ``Dataset.from_file``
    mmaps yields the identical (lazy) dataset in ~20 s and ~1 GB. Row order is
    preserved (``state.json`` ``_data_files`` order), so any global row index
    stays consistent with a later ``load_from_disk`` read. Falls back to ``load_from_disk`` when the shard
    layout can't be resolved safely (see ``_resolve_shard_paths``).
    """
    preprocessed_dir = Path(preprocessed_dir)
    shards = _resolve_shard_paths(preprocessed_dir)
    if not shards:
        return load_from_disk(str(preprocessed_dir))
    if len(shards) == 1:
        return Dataset.from_file(shards[0])
    return concatenate_datasets([Dataset.from_file(s) for s in shards])


def scan_dataset_metadata(
    preprocessed_dir: str,
    columns: Sequence[str] = ("dataset_name", "song_name"),
) -> Dict[str, list]:
    """Read only small metadata ``columns``, per-shard, for fast row filtering.

    Reading e.g. ``dataset_name`` this way costs ~22 s / ~1.2 GB on the full
    3.5 M-row corpus — versus ~385 s / ~24 GB to ``load_from_disk`` first. Use
    it to decide *which* rows a script wants (which datasets / songs) before
    paying to materialize the heavy ``audio_feats`` blob via ``.select`` on a
    ``fast_load_from_disk`` view. Returned lists are aligned to the global row
    index (``state.json`` order), matching ``fast_load_from_disk`` /
    ``load_from_disk``.
    """
    preprocessed_dir = Path(preprocessed_dir)
    columns = list(columns)
    shards = _resolve_shard_paths(preprocessed_dir)
    if not shards:
        ds = load_from_disk(str(preprocessed_dir))
        return {c: ds[c] for c in columns}
    out: Dict[str, list] = {c: [] for c in columns}
    for shard in shards:
        ds = Dataset.from_file(shard)
        for c in columns:
            out[c].extend(ds[c])
        del ds
    return out


def load_preprocessed_svs_datasets(
    preprocessed_dir: str,
    val_datasets: Optional[List[str]] = None,
    val_songs: Optional[List[str]] = None,
    val_samples: int = 0,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Load preprocessed SVS datasets from Arrow format and split into train/val."""
    preprocessed_dir = Path(preprocessed_dir)
    if not preprocessed_dir.exists():
        raise ValueError(f"Preprocessed data not found at {preprocessed_dir}")

    val_datasets = val_datasets or []
    val_songs = val_songs or []

    if val_datasets or val_songs:
        val_datasets_set = set(val_datasets)
        val_songs_set = set(val_songs)
        # fast_load_from_disk is a lazy per-shard concat (~27 s vs ~385 s for
        # load_from_disk); reading the two small label columns off it is another
        # ~4 s and never faults the heavy audio_feats blob. ``.select`` below is
        # lazy, so only the rows actually used materialize.
        all_ds = fast_load_from_disk(str(preprocessed_dir))
        all_dataset_names = all_ds["dataset_name"]
        all_song_names = all_ds["song_name"]

        val_indices = []
        train_indices = []
        for idx, (dataset_name, song_name) in enumerate(zip(all_dataset_names, all_song_names)):
            if dataset_name in val_datasets_set or song_name in val_songs_set:
                val_indices.append(idx)
            else:
                train_indices.append(idx)

        train_ds = all_ds.select(train_indices) if train_indices else all_ds.select([])
        val_ds = all_ds.select(val_indices) if val_indices else None

        if val_datasets:
            found = {all_dataset_names[idx] for idx in val_indices} & val_datasets_set
            missing = val_datasets_set - found
            if missing:
                print(f"Warning: val datasets not found: {missing}")
            counts = {}
            for idx in val_indices:
                dataset_name = all_dataset_names[idx]
                if dataset_name in val_datasets_set:
                    counts[dataset_name] = counts.get(dataset_name, 0) + 1
            print(f"Validation datasets: {counts}")

        if val_songs:
            found = {all_song_names[idx] for idx in val_indices} & val_songs_set
            missing = val_songs_set - found
            if missing:
                print(f"Warning: {len(missing)} val songs not found: {missing}")
            print(f"Found {len(found)} validation songs")

        print(f"Split: {len(train_indices)} train, {len(val_indices)} val")
        return train_ds, val_ds

    all_ds = fast_load_from_disk(str(preprocessed_dir))

    if val_samples > 0:
        import random as rnd

        rnd_state = rnd.Random(42)
        indices = list(range(len(all_ds)))
        rnd_state.shuffle(indices)
        n_val = min(val_samples, len(all_ds) - 1)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        train_ds = all_ds.select(train_indices)
        val_ds = all_ds.select(val_indices) if val_indices else None
        print(f"Random split (seed=42): {len(train_indices)} train, {n_val} val")
        return train_ds, val_ds

    print(f"No validation split: {len(all_ds)} samples for training")
    return all_ds, None
