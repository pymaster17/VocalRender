"""
SVS data loading utilities for preprocessing.

Provides functions to load raw samples from various dataset formats:

* :func:`load_config` — YAML config loading and validation
* :func:`load_samples_from_folder` — folder-based dataset structure
* :func:`load_samples_from_json_file` — single JSON annotation file
* :func:`load_samples_from_weak_json_file` — weak-label JSON file
* :func:`load_all_datasets` — unified multi-dataset loader
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


# Silence / non-lyric markers — a sample containing only these has no actual
# lyric content. Such samples are dropped at load time.
_LYRIC_SKIP = {"AP", "SP", "<AP>", "<SP>", "<sil>", "<pause>",
               "", " ", "-", "_", "<unk>"}


def reconstruct_lyric_text(word_list: List[str]) -> str:
    """Join lyric syllables, dropping silence/special markers.

    Empty result ⇒ the clip is pure breath/silence or has no Chinese word
    tokens (e.g. humming, non-Chinese lyrics the annotator refused to
    transcribe). Such clips teach nothing useful to text→song SVS training.
    """
    kept: List[str] = []
    for w in word_list or []:
        if w is None:
            continue
        stripped = str(w).strip()
        if not stripped or stripped in _LYRIC_SKIP:
            continue
        if stripped.startswith("<") and stripped.endswith(">"):
            continue
        kept.append(stripped)
    return "".join(kept)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.

    Configuration format::

        datasets:
          - name: m4singer
            type: json_file
            json_path: /path/to/m4singer.json
            audio_root: /path/to/wavs
            song_id_indices: [0, 1]
          - name: cloudmusic
            type: folder_based
            dataset_root: /path/to/cloudmusic

    For validation, splitting is now deferred to training time.
    The preprocessing output is a flat directory containing all samples.

    Song ID determination:

    - For folder_based: song_id is the first-level folder name under dataset_root
    - For json_file: song_id is constructed by joining the elements at
      song_id_indices with '#'
    """
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Empty config file: {config_path}")

    # Apply defaults for optional parameters
    defaults = {
        "sample_rate": 44100,
        "max_samples": -1,
        "num_workers": 4,
        "device": "cuda",
        "shard_size": 1000,  # Number of samples per Arrow shard for streaming writes
        "vae_batch_size": 32,  # Max number of samples per VAE encode batch (hard cap)
        "vae_max_tokens": 44100 * 100,  # Max total audio samples per VAE batch (dynamic sizing)
        "num_gpus": -1,  # Number of GPUs for parallel VAE encoding (-1 = auto-detect)
        "dispatch_chunk_size": 2000,  # Multi-GPU: samples per work-queue chunk (dynamic dispatch granularity)
        "manifest_dir": None,  # Where to cache per-dataset sample manifests (None -> <output_dir>/.manifest_cache)
        "refresh_manifest": False,  # Force rescan + rewrite of manifests
    }

    for key, value in defaults.items():
        if key not in config:
            config[key] = value

    # Validate required parameters
    required = ["datasets", "output_dir", "pretrained_path"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required config parameters: {missing}")

    # Validate datasets configuration
    if not config["datasets"]:
        raise ValueError("'datasets' list cannot be empty. Add at least one dataset configuration.")

    # Validate each dataset entry
    for i, ds in enumerate(config["datasets"]):
        if "name" not in ds:
            raise ValueError(f"Dataset {i}: missing 'name' field")
        if "type" not in ds:
            raise ValueError(f"Dataset {i} ({ds['name']}): missing 'type' field")

        ds_type = ds["type"]
        if ds_type == "folder_based":
            if "dataset_root" not in ds:
                raise ValueError(f"Dataset '{ds['name']}': folder_based type requires 'dataset_root'")
        elif ds_type == "json_file":
            if "json_path" not in ds:
                raise ValueError(f"Dataset '{ds['name']}': json_file type requires 'json_path'")
            if "audio_root" not in ds:
                raise ValueError(f"Dataset '{ds['name']}': json_file type requires 'audio_root'")
            if "song_id_indices" not in ds and "song_id_slice" not in ds:
                raise ValueError(f"Dataset '{ds['name']}': json_file type requires 'song_id_indices' or 'song_id_slice'")
        elif ds_type == "weak_json_file":
            if "json_path" not in ds:
                raise ValueError(f"Dataset '{ds['name']}': weak_json_file type requires 'json_path'")
            if "audio_root" not in ds:
                raise ValueError(f"Dataset '{ds['name']}': weak_json_file type requires 'audio_root'")
        else:
            raise ValueError(f"Dataset '{ds['name']}': unknown type '{ds_type}'. Use 'folder_based', 'json_file', or 'weak_json_file'")

    return config


def _convert_words_to_syllables(words: List[str]) -> List[Dict]:
    """Convert word-only annotation to syllables (no pitch/note info).

    For weak-label datasets that only have word annotations.
    Each syllable only has a 'char' field — no 'pitch' or 'note'.
    """
    return [{"char": w} for w in words]


_SOLO_PRIMARY = {"男歌手": "male", "女歌手": "female"}


def _resolve_bpm(metadata: Dict) -> int:
    """Resolve BPM from a metadata dict.

    The cloudmusic export carries the tempo under one of two keys depending on
    provenance: ``bpm`` (predicted by an estimator) or ``bpmMeta`` (annotated
    by the song's author). Training does not distinguish the two, so whichever
    is present wins; ``bpm`` is preferred when (rarely) both exist.
    """
    for key in ("bpm", "bpmMeta"):
        val = metadata.get(key)
        if val:
            return val
    return 120


def _resolve_song_gender(metadata: Dict) -> Tuple[bool, Optional[str]]:
    """Decide whether a song is a single male/female solo and its gender.

    Returns ``(is_solo, gender)`` where ``gender`` ∈ {"male", "female", None}.
    A song counts as solo only when it has exactly one artist whose ``type``
    is a known male/female singer tag — this drops 组合/乐队/duets so that
    prompt/target pairs sampled from one song share a singer.
    """
    artists = metadata.get("artists") or []
    # ``artists`` format varies by export: dicts ({"id","name","type"}) in
    # cloudmusic, bare strings (just the name) in muchin, and absent/null in
    # Muse & songformdb. Only dict entries carry a singer ``type`` to resolve
    # gender; a string entry has no type, so it can never qualify as a known
    # male/female solo.
    if len(artists) == 1 and isinstance(artists[0], dict):
        atype = artists[0].get("type")
        if atype in _SOLO_PRIMARY:
            return True, _SOLO_PRIMARY[atype]
    return False, None


def load_samples_from_folder(
    dataset_name: str,
    dataset_root: str,
    max_samples: int = -1,
    audio_extensions: Tuple[str, ...] = (".opus", ".flac", ".wav", ".mp3"),
    solo_singer_only: bool = False,
    num_scan_workers: int = 12,
) -> List[Dict]:
    """Load raw samples from folder-based structure.

    Each song lives in its own folder under ``dataset_root``. The current
    cloudmusic export keeps *all* per-segment annotations inside a single
    song-level ``metadata.json`` (there are no longer per-segment ``.json``
    files); each entry in ``metadata['lyric']`` corresponds to one audio
    segment named ``<seg_id>.<ext>``::

        dataset_root/
            <song_folder>/
                metadata.json   # id/name/artists/bpm|bpmMeta/lyric[]
                0000.opus
                0001.opus
                ...

    where each ``lyric`` entry looks like::

        {"seg_id": "0000", "text": "...",
         "word": [...], "word_dur": [...],
         "pitch": [...], "note": [...], "pitch2word": [...]}

    The BPM is read from ``bpm`` or ``bpmMeta`` (see :func:`_resolve_bpm`).

    When ``solo_singer_only`` is set, songs that are not a single male/female
    solo (exactly one ``artists`` entry with ``type`` ∈ {男歌手, 女歌手}) are
    skipped entirely, keeping prompt/target pairs from one song on the same
    singer. For a solo song the normalized gender (``"male"``/``"female"``)
    is stamped onto every sample under the ``gender`` key so downstream
    consumers can pick gender-conditional reference audio without
    re-reading metadata.
    """
    from vocalrender.training.svs_data import convert_annotation_to_syllables
    from concurrent.futures import ThreadPoolExecutor, as_completed

    dataset_root = Path(dataset_root)
    song_folders = [d for d in dataset_root.iterdir() if d.is_dir()]

    print(f"[{dataset_name}] Scanning {len(song_folders)} song folders "
          f"with {num_scan_workers} threads...")

    # Per-folder worker — opens one metadata.json (the NFS-latency-bound step)
    # and assembles its segments. Pure/thread-safe: only reads files and calls
    # stateless helpers, so a thread pool just overlaps the open() round-trips.
    # The work is metadata-IOPS (a few KB per file), not bandwidth, so a bounded
    # pool stays gentle on shared NFS while hiding per-open latency.
    def _scan_folder(song_folder: Path) -> Dict:
        metadata_path = song_folder / "metadata.json"
        if not metadata_path.exists():
            return {"skip_no_meta": True}
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            return {"skip_no_meta": True}

        is_solo, song_gender = _resolve_song_gender(metadata)
        if solo_singer_only and not is_solo:
            return {"skip_non_solo": True}

        bpm = _resolve_bpm(metadata)

        # Map each segment id to its audio file (seg_id -> path). Annotations
        # reference segments by ``seg_id`` (e.g. "0000" -> "0000.opus").
        audio_by_stem = {
            f.stem: f
            for f in song_folder.iterdir()
            if f.is_file() and f.suffix.lower() in audio_extensions
        }

        folder_samples: List[Dict] = []
        weak = 0
        for seg in metadata.get("lyric", []):
            seg_id = str(seg.get("seg_id", "")).strip()
            audio_file = audio_by_stem.get(seg_id)
            if audio_file is None:
                continue

            words = seg.get("word", [])
            pitches = seg.get("pitch", [])
            notes = seg.get("note", [])
            pitch2word = seg.get("pitch2word", [])
            pitch_dur = seg.get("pitch_dur", [])
            word_dur = seg.get("word_dur", [])

            if not words:
                continue

            # Auto-detect: full score vs weak label (word-only)
            has_score = bool(pitches and notes and pitch2word)

            if has_score:
                syllables = convert_annotation_to_syllables(
                    words=words,
                    pitches=pitches,
                    notes=notes,
                    pitch2word=pitch2word,
                )
            else:
                syllables = _convert_words_to_syllables(words)
                weak += 1

            folder_samples.append({
                "audio_path": str(audio_file),
                "bpm": bpm if has_score else 0,
                "syllables": syllables,
                "notes": notes,  # Raw note list for duration estimation (empty for weak)
                "word": words,
                "pitch": pitches,
                "pitch_dur": pitch_dur,
                "pitch2word": pitch2word,
                "word_dur": word_dur,
                # seg_id is unique within a song; prefix with the folder so it
                # is globally unique.
                "item_name": f"{song_folder.name}/{seg_id}" if seg_id else audio_file.stem,
                "song_name": song_folder.name,  # For folder_based, song_name is the folder name
                "song_folder": song_folder.name,
                "dataset_name": dataset_name,
                "has_score": has_score,
                "gender": song_gender,
            })
        return {"samples": folder_samples, "weak": weak}

    samples: List[Dict] = []
    weak_count = 0
    skipped_non_solo = 0
    skipped_no_meta = 0

    with ThreadPoolExecutor(max_workers=max(1, num_scan_workers)) as ex:
        futures = [ex.submit(_scan_folder, sf) for sf in song_folders]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"Loading [{dataset_name}]"):
            res = fut.result()
            if res.get("skip_no_meta"):
                skipped_no_meta += 1
                continue
            if res.get("skip_non_solo"):
                skipped_non_solo += 1
                continue
            samples.extend(res["samples"])
            weak_count += res["weak"]
            # Folder order is non-deterministic under the pool, but ``max_samples``
            # only ever bounds calibration runs (production uses -1), so a slightly
            # different subset is acceptable; truncate once the cap is reached.
            if max_samples > 0 and len(samples) >= max_samples:
                samples = samples[:max_samples]
                break

    print(f"[{dataset_name}] Loaded {len(samples)} samples from folder structure"
          f" ({weak_count} weak-label)")
    if skipped_no_meta:
        print(f"[{dataset_name}] skipped {skipped_no_meta} folder(s) "
              f"without a readable metadata.json")
    if solo_singer_only:
        print(f"[{dataset_name}] solo_singer_only: skipped {skipped_non_solo} "
              f"non-solo / multi-singer / unknown-artist song folder(s)")
    return samples


def load_samples_from_json_file(
    dataset_name: str,
    json_path: str,
    audio_root: str,
    song_id_indices: Optional[List[int]] = None,
    song_id_slice: Optional[List[int]] = None,
    song_id_separator: str = "#",
    max_samples: int = -1,
) -> List[Dict]:
    """Load raw samples from a single JSON file format (like m4singer.json).

    Args:
        dataset_name: Name identifier for the dataset
        json_path: Path to the JSON annotation file
        audio_root: Root directory for audio files
        song_id_indices: List of indices in '#'-split item_name for constructing
            unique song ID, e.g., [0, 1] for "Alto-1#newboy#0000" -> "Alto-1#newboy"
        song_id_slice: [start, end] character positions for extracting song ID
            from item_name, e.g., [0, 4] for "2001000001" -> "2001"
        max_samples: Maximum number of samples to load

    JSON format (array of items)::

        [
            {
                "item_name": "Alto-1#newboy#0000",
                "word": ["好", "的", ...],
                "pitch": [59, 62, ...],
                "note": ["<NOTE_16>", ...],
                "pitch2word": [0, 1, ...],
                "bpm": 135,
                "wav_fn": "Alto-1#newboy/0000.wav"
            },
            ...
        ]
    """
    from vocalrender.training.svs_data import convert_annotation_to_syllables

    json_path = Path(json_path)
    audio_root = Path(audio_root)

    print(f"[{dataset_name}] Loading from JSON file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"[{dataset_name}] Found {len(items)} items in JSON file")

    samples = []
    weak_count = 0
    for item in tqdm(items, desc=f"Processing [{dataset_name}]"):
        if max_samples > 0 and len(samples) >= max_samples:
            break

        words = item.get("word", [])
        pitches = item.get("pitch", [])
        notes = item.get("note", [])
        pitch2word = item.get("pitch2word", [])
        pitch_dur = item.get("pitch_dur", [])
        word_dur = item.get("word_dur", [])
        bpm = item.get("bpm", 120)
        wav_fn = item.get("wav_fn", "")
        item_name = item.get("item_name", Path(wav_fn).stem)

        if not words or not wav_fn:
            continue

        audio_path = audio_root / wav_fn
        if not audio_path.exists():
            continue

        # Auto-detect: full score vs weak label (word-only)
        has_score = bool(pitches and notes and pitch2word)

        if has_score:
            syllables = convert_annotation_to_syllables(
                words=words,
                pitches=pitches,
                notes=notes,
                pitch2word=pitch2word,
            )
        else:
            syllables = _convert_words_to_syllables(words)
            weak_count += 1

        # Extract song_id from item_name
        if song_id_slice is not None:
            # Positional extraction: e.g., [0, 4] for "2001000001" -> "2001"
            song_name = item_name[song_id_slice[0]:song_id_slice[1]]
        elif song_id_indices is not None:
            # Split by separator and join selected indices
            # e.g., '#'-split [0, 1] for "Alto-1#newboy#0000" -> "Alto-1#newboy"
            # e.g., '_'-split [0, 1] for "0_一如年少模样_0" -> "0_一如年少模样"
            item_name_parts = item_name.split(song_id_separator)
            song_id_parts = []
            for idx in song_id_indices:
                if idx < len(item_name_parts):
                    song_id_parts.append(item_name_parts[idx])
            song_name = song_id_separator.join(song_id_parts) if song_id_parts else item_name
        else:
            song_name = item_name

        # Extract song_folder from wav_fn (e.g., "Alto-1#newboy/0000.wav" -> "Alto-1#newboy")
        song_folder = str(Path(wav_fn).parent) if "/" in wav_fn or "\\" in wav_fn else ""

        samples.append({
            "audio_path": str(audio_path),
            "bpm": bpm if has_score else 0,
            "syllables": syllables,
            "notes": notes,  # Raw note list for duration estimation (empty for weak)
            "word": words,
            "pitch": pitches,
            "pitch_dur": pitch_dur,
            "pitch2word": pitch2word,
            "word_dur": word_dur,
            "item_name": item_name,
            "song_name": song_name,  # Now contains singer#song format
            "song_folder": song_folder,
            "dataset_name": dataset_name,
            "has_score": has_score,
        })

    print(f"[{dataset_name}] Loaded {len(samples)} samples from JSON file"
          f" ({weak_count} weak-label)")
    return samples


def load_samples_from_weak_json_file(
    dataset_name: str,
    json_path: str,
    audio_root: str,
    min_confidence: Optional[str] = None,
    max_samples: int = -1,
) -> List[Dict]:
    """Load raw samples from a weak-label JSON file.

    Args:
        dataset_name: Name identifier for the dataset
        json_path: Path to the JSON annotation file
        audio_root: Root directory for audio files
        min_confidence: Minimum confidence level to include.
            "high" = only high; "medium" = medium+high; None = all
        max_samples: Maximum number of samples to load

    JSON format (array of items)::

        [
            {
                "audio_path": "song_folder/segment_NNNN/audio.wav",
                "transcription": "lyrics text",
                "confidence": "high" | "medium" | "low"
            },
            ...
        ]
    """
    json_path = Path(json_path)
    audio_root = Path(audio_root)

    print(f"[{dataset_name}] Loading from weak JSON file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"[{dataset_name}] Found {len(items)} items in weak JSON file")

    # Confidence filtering
    confidence_levels = {"high": 3, "medium": 2, "low": 1}
    min_conf_val = confidence_levels.get(min_confidence, 0) if min_confidence else 0

    samples = []
    skipped_conf = 0
    skipped_empty = 0
    for item in tqdm(items, desc=f"Processing [{dataset_name}]"):
        if max_samples > 0 and len(samples) >= max_samples:
            break

        transcription = item.get("transcription", "").strip()
        if not transcription:
            skipped_empty += 1
            continue

        confidence = item.get("confidence", "medium")
        if confidence_levels.get(confidence, 0) < min_conf_val:
            skipped_conf += 1
            continue

        wav_fn = item.get("audio_path", "")
        if not wav_fn:
            continue

        audio_path = audio_root / wav_fn
        if not audio_path.exists():
            continue

        # Each character becomes a word
        words = list(transcription)
        syllables = _convert_words_to_syllables(words)

        # Song name = first path component (e.g. "秦之声 (2024-01-01)-卖妙郎-王楠+")
        song_name = Path(wav_fn).parts[0] if Path(wav_fn).parts else ""

        # Item name from audio path stem
        item_name = str(Path(wav_fn).with_suffix(""))

        samples.append({
            "audio_path": str(audio_path),
            "bpm": 0,
            "syllables": syllables,
            "notes": [],
            "word": words,
            "pitch": [],
            "pitch_dur": [],
            "pitch2word": [],
            "word_dur": [],
            "item_name": item_name,
            "song_name": song_name,
            "song_folder": song_name,
            "dataset_name": dataset_name,
            "has_score": False,
        })

    print(f"[{dataset_name}] Loaded {len(samples)} samples from weak JSON file"
          f" (skipped: {skipped_empty} empty, {skipped_conf} low-confidence)")
    return samples


def _dataset_signature(ds_config: Dict) -> Dict:
    """Config fields that change which samples a dataset yields.

    Used as the manifest cache key. Note this captures *config*, not data
    content — annotation edits that leave the folder set unchanged are caught
    only by the folder-count guard (folder_based) or ``refresh_manifest``.
    """
    keys = ("name", "type", "dataset_root", "json_path", "audio_root",
            "solo_singer_only", "song_id_indices", "song_id_slice",
            "song_id_separator", "min_confidence")
    return {k: ds_config.get(k) for k in keys if k in ds_config}


def _manifest_path(manifest_dir: Path, ds_config: Dict) -> Path:
    sig = _dataset_signature(ds_config)
    h = hashlib.sha1(json.dumps(sig, sort_keys=True,
                                ensure_ascii=False).encode()).hexdigest()[:10]
    name = ds_config.get("name", "unknown")
    return manifest_dir / f"{name}__{h}.manifest.json"


def _current_folder_count(ds_config: Dict) -> Optional[int]:
    """Cheap directory-count guard for folder_based manifests (one iterdir)."""
    if ds_config.get("type") != "folder_based":
        return None
    root = ds_config.get("dataset_root")
    if not root:
        return None
    try:
        return sum(1 for d in Path(root).iterdir() if d.is_dir())
    except Exception:
        return None


def load_all_datasets(
    datasets_config: List[Dict],
    max_samples: int = -1,
    manifest_dir: Optional[str] = None,
    refresh_manifest: bool = False,
) -> List[Dict]:
    """Load samples from multiple datasets.

    Args:
        datasets_config: List of dataset configurations, each containing:

            - name: Dataset name (will be stored in output)
            - type: "folder_based" or "json_file"
            - For folder_based: dataset_root
            - For json_file: json_path, audio_root

        max_samples: Maximum total samples to load (-1 for all)
        manifest_dir: If set, cache each dataset's *filtered* sample list to a
            JSON manifest here and reuse it on subsequent runs, skipping the
            (NFS-latency-bound) folder scan. Only written on a full load
            (``max_samples == -1``) so a capped calibration run never poisons
            the cache. For folder_based datasets the manifest stores the folder
            count and is auto-invalidated if it no longer matches; **annotation
            edits that do not change the folder set are NOT detected** — pass
            ``refresh_manifest=True`` (or delete the manifest) after editing
            metadata.
        refresh_manifest: Force a rescan and rewrite of all manifests.

    Returns:
        Combined list of samples from all datasets
    """
    all_samples = []
    mdir = Path(manifest_dir) if manifest_dir else None
    if mdir is not None:
        mdir.mkdir(parents=True, exist_ok=True)

    for ds_config in datasets_config:
        ds_name = ds_config.get("name", "unknown")
        ds_type = ds_config.get("type", "folder_based")

        # Calculate remaining samples if max_samples is set
        remaining = max_samples - len(all_samples) if max_samples > 0 else -1
        if max_samples > 0 and remaining <= 0:
            print(f"[{ds_name}] Skipping: max_samples reached")
            break

        # ---- Manifest cache lookup --------------------------------------
        mpath = _manifest_path(mdir, ds_config) if mdir is not None else None
        if mpath is not None and mpath.exists() and not refresh_manifest:
            try:
                with open(mpath, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                cur_count = _current_folder_count(ds_config)
                stale = (cur_count is not None
                         and manifest.get("folder_count") != cur_count)
                if stale:
                    print(f"[{ds_name}] Manifest stale (folder count "
                          f"{manifest.get('folder_count')} -> {cur_count}), rescanning")
                else:
                    cached = manifest["samples"]
                    if max_samples > 0:
                        cached = cached[:remaining]
                    print(f"[{ds_name}] Loaded {len(cached)} samples from manifest "
                          f"cache: {mpath.name}  (delete it or set refresh_manifest "
                          f"if annotations changed)")
                    all_samples.extend(cached)
                    continue
            except Exception as e:
                print(f"[{ds_name}] Failed to read manifest ({e}), rescanning")

        if ds_type == "folder_based":
            dataset_root = ds_config.get("dataset_root")
            if not dataset_root:
                print(f"[{ds_name}] Warning: Missing 'dataset_root' for folder_based dataset, skipping")
                continue
            samples = load_samples_from_folder(
                dataset_name=ds_name,
                dataset_root=dataset_root,
                max_samples=remaining,
                solo_singer_only=bool(ds_config.get("solo_singer_only", False)),
            )
        elif ds_type == "json_file":
            json_path = ds_config.get("json_path")
            audio_root = ds_config.get("audio_root")
            if not json_path or not audio_root:
                print(f"[{ds_name}] Warning: Missing 'json_path' or 'audio_root' for json_file dataset, skipping")
                continue
            samples = load_samples_from_json_file(
                dataset_name=ds_name,
                json_path=json_path,
                audio_root=audio_root,
                song_id_indices=ds_config.get("song_id_indices"),
                song_id_slice=ds_config.get("song_id_slice"),
                song_id_separator=ds_config.get("song_id_separator", "#"),
                max_samples=remaining,
            )
        elif ds_type == "weak_json_file":
            json_path = ds_config.get("json_path")
            audio_root = ds_config.get("audio_root")
            if not json_path or not audio_root:
                print(f"[{ds_name}] Warning: Missing 'json_path' or 'audio_root' for weak_json_file dataset, skipping")
                continue
            samples = load_samples_from_weak_json_file(
                dataset_name=ds_name,
                json_path=json_path,
                audio_root=audio_root,
                min_confidence=ds_config.get("min_confidence"),
                max_samples=remaining,
            )
        else:
            print(f"[{ds_name}] Warning: Unknown dataset type '{ds_type}', skipping")
            continue

        # Drop samples with no transcribable lyric content (pure AP/SP clips,
        # humming, non-Chinese lyrics left untranscribed).
        original_n = len(samples)
        samples = [s for s in samples
                   if reconstruct_lyric_text(s.get("word", []))]
        dropped = original_n - len(samples)
        if dropped:
            print(f"[{ds_name}] Dropped {dropped} empty-lyric sample(s) "
                  f"(pure AP/SP, humming, or untranscribed non-Chinese)")

        # ---- Persist manifest (full loads only) -------------------------
        if mpath is not None and max_samples <= 0:
            try:
                tmp = mpath.with_suffix(".json.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({
                        "signature": _dataset_signature(ds_config),
                        "folder_count": _current_folder_count(ds_config),
                        "num_samples": len(samples),
                        "samples": samples,
                    }, f, ensure_ascii=False)
                tmp.replace(mpath)  # atomic — never leave a half-written manifest
                print(f"[{ds_name}] Wrote manifest cache: {mpath.name} "
                      f"({len(samples)} samples)")
            except Exception as e:
                print(f"[{ds_name}] Warning: failed to write manifest ({e})")

        all_samples.extend(samples)

    print(f"\nTotal samples loaded from {len(datasets_config)} dataset(s): {len(all_samples)}")
    return all_samples
