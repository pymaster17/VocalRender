"""
Arrow dataset writing for SVS preprocessing.

Provides :func:`process_and_save` (single-GPU) and
:func:`process_and_save_multigpu` (multi-GPU via a dynamic work queue) for
converting raw SVS samples into Arrow format suitable for training.

Multi-GPU dispatch is **dynamic**: the parent feeds duration-homogeneous chunks
into a bounded ``mp.Queue`` and each GPU worker pulls chunks on demand, so a
fast/uncontended card processes more work than a slow/shared one and the total
wall-clock is bounded by aggregate throughput rather than the slowest card.
"""

import json
import threading
import time
import queue as _queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from .svs_preprocessor import SVSPreprocessor
from .text_tensor import build_text_tensor, estimate_duration_from_notes


# ---------------------------------------------------------------------------
# Shared Arrow helpers (module level so both the single-GPU path and the
# multi-GPU worker loop reuse the exact same schema / writer / encode loop).
# ---------------------------------------------------------------------------
def _build_features():
    """HuggingFace ``Features`` describing one preprocessed SVS sample.

    ``audio_feats`` is stored as float16 (native arrow halffloat, 2 bytes) to
    halve on-disk size (~460G f32 -> ~230G). Loaders cast to f32/bf16 on read,
    and the extra error is bounded by 1 bf16 ULP (training runs amp_bf16), i.e.
    below the bf16 noise floor the flow-matching target already incurs.
    """
    from datasets import Features, Sequence, Value
    return Features({
        "packed_text_tokens": Sequence(Value("int32")),
        "audio_feats": Sequence(Sequence(Sequence(Value("float16")))),  # [T, P, D]
        "text_mask": Sequence(Value("int32")),
        "audio_mask": Sequence(Value("int32")),
        "loss_mask": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
        "audio_duration": Value("float64"),
        "estimated_duration": Value("float64"),  # Pre-computed for inference sorting
        "text_token_count": Value("int64"),
        "total_length": Value("int64"),
        "item_name": Value("string"),
        "song_name": Value("string"),  # Song identifier for validation splitting
        "song_folder": Value("string"),
        "dataset_name": Value("string"),  # Source dataset identifier
        # Structured metadata -- stored verbatim from source annotations
        # so that downstream tasks (score rendering, analysis) can use them
        # without reverse-parsing the SVS token sequence.
        "svs_prompt": Value("string"),  # Decoded SVS prompt text
        "bpm": Value("int64"),
        "word": Sequence(Value("string")),  # e.g. ["让", "我", "AP", ...]
        "pitch": Sequence(Value("int64")),  # MIDI pitch per note, e.g. [60, 58, 0, ...]
        "pitch_dur": Sequence(Value("float64")),  # Duration in seconds per pitch
        "note": Sequence(Value("string")),  # Note token strings, e.g. ["<NOTE_4>", ...]
        "pitch2word": Sequence(Value("int64")),  # Pitch-to-word alignment index
        "word_dur": Sequence(Value("float64")),  # Word-level durations in seconds
        "has_score": Value("bool"),  # True = full annotation, False = word-only weak label
    })


def _create_empty_batch():
    """Create an empty batch dictionary (one key per feature column)."""
    return {
        "packed_text_tokens": [],
        "audio_feats": [],
        "text_mask": [],
        "audio_mask": [],
        "loss_mask": [],
        "labels": [],
        "audio_duration": [],
        "estimated_duration": [],
        "text_token_count": [],
        "total_length": [],
        "item_name": [],
        "song_name": [],
        "song_folder": [],
        "dataset_name": [],
        "svs_prompt": [],
        "bpm": [],
        "word": [],
        "pitch": [],
        "pitch_dur": [],
        "note": [],
        "pitch2word": [],
        "word_dur": [],
        "has_score": [],
    }


def _to_arrow_array(values, arrow_type):
    """Convert a list of values to a PyArrow array matching the given type.

    Multi-dimensional numpy arrays (e.g. audio_feats [T,P,D]) must be converted
    to nested Python lists first, because pa.array() only accepts 1-D arrays.
    """
    import numpy as np
    import pyarrow as pa
    converted = [
        v.tolist() if isinstance(v, np.ndarray) and v.ndim > 1 else v
        for v in values
    ]
    return pa.array(converted, type=arrow_type)


class _ShardWriter:
    """Manages incremental Arrow IPC stream writing with shard rotation.

    A single writer can be fed many batches over its lifetime (across multiple
    work-queue chunks in the multi-GPU path); shards roll once they reach
    ``shard_size`` samples. ``rank_prefix`` keeps filenames unique per worker.
    """

    def __init__(self, output_path: Path, shard_size: int, arrow_schema,
                 rank_prefix: str = ""):
        self.output_path = output_path
        self.shard_size = shard_size
        self.arrow_schema = arrow_schema
        self.rank_prefix = rank_prefix
        self.shard_idx = 0
        self.shard_sample_count = 0
        self.shard_paths: List[str] = []
        self._file = None
        self._writer = None
        self._open_new_shard()

    def _open_new_shard(self):
        import pyarrow as pa
        shard_filename = f"data-{self.rank_prefix}{self.shard_idx:05d}.arrow"
        shard_path = self.output_path / shard_filename
        self.shard_paths.append(str(shard_path))
        self._file = pa.OSFile(str(shard_path), 'wb')
        self._writer = pa.ipc.new_stream(self._file, self.arrow_schema)

    def write_batch(self, batch_dict: Dict, n_samples: int):
        """Write a small batch dict as an Arrow RecordBatch.

        Handles shard rotation: if adding ``n_samples`` exceeds ``shard_size``,
        close the current shard and start a new one first.
        """
        import pyarrow as pa
        if n_samples == 0:
            return
        if self.shard_sample_count > 0 and self.shard_sample_count + n_samples > self.shard_size:
            self._close_current_shard()
            self.shard_idx += 1
            self._open_new_shard()
        arrays = [
            _to_arrow_array(batch_dict[field.name], field.type)
            for field in self.arrow_schema
        ]
        rb = pa.RecordBatch.from_arrays(arrays, schema=self.arrow_schema)
        self._writer.write_batch(rb)
        self.shard_sample_count += n_samples

    def _close_current_shard(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._file is not None:
            self._file.close()
            self._file = None
        print(f"  Wrote shard {self.rank_prefix}{self.shard_idx}: {self.shard_sample_count} samples")
        self.shard_sample_count = 0

    @property
    def num_shards(self) -> int:
        """Total shards opened (1-based count)."""
        return self.shard_idx + 1

    def close(self):
        """Close everything; call once at the end."""
        self._close_current_shard()


def _estimate_sort_duration(s):
    """Estimate a sample's audio duration for length-based sorting.

    Full-score samples use the note schedule; weak-label samples fall back to
    the audio file header (fast, no full decode).
    """
    import soundfile as sf
    if s["has_score"] and s["notes"]:
        return estimate_duration_from_notes(s["notes"], s["bpm"])
    try:
        info = sf.info(s["audio_path"])
        return info.duration
    except Exception:
        return 0.0


# Timing buckets used by the encode loop (and the single-GPU report).
_TIMING_KEYS = [
    "cpu_prep",     # audio_load + resample + build_text (threaded)
    "vae_encode",   # batch VAE encode (GPU)
    "assemble",     # process_sample + decode_prompt + batch_accum
    "shard_write",  # incremental Arrow RecordBatch writes
]


def _encode_into_shards(
    samples: List[Dict],
    preprocessor: SVSPreprocessor,
    shard_writer: _ShardWriter,
    num_workers: int,
    vae_batch_size: int,
    vae_max_tokens: int,
    pbar=None,
    timing_totals: Optional[Dict] = None,
    timing_counts: Optional[Dict] = None,
):
    """Encode ``samples`` and append them into the (already-open) shard_writer.

    Runs a thread-pool prefetch (overlaps audio I/O with GPU) plus dynamic
    VAE-batch sizing. Does NOT sort ``samples`` (caller guarantees ordering)
    and does NOT close ``shard_writer`` (so it can be reused across chunks).

    Returns ``(n_processed, n_skipped)``.
    """
    import torchaudio
    import soundfile as sf

    if timing_totals is None:
        timing_totals = {k: 0.0 for k in _TIMING_KEYS}
    if timing_counts is None:
        timing_counts = {k: 0 for k in _TIMING_KEYS}

    n_processed = 0
    n_skipped = 0

    _target_sr = preprocessor.sample_rate
    _tokenizer = preprocessor.tokenizer

    # -- Helper: prepare a single sample (runs in thread pool) ---------
    def _prepare_one(sample: Dict) -> Optional[Dict]:
        """Thread-safe single-sample preparation (CPU/IO only)."""
        try:
            audio_data, sr = sf.read(sample["audio_path"])
            waveform = torch.from_numpy(audio_data).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            if sr != _target_sr:
                resampler = torchaudio.transforms.Resample(sr, _target_sr)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)  # [T]

            estimated_duration = estimate_duration_from_notes(
                sample["notes"], sample["bpm"],
            ) if sample["has_score"] and sample["notes"] else (
                waveform.shape[-1] / _target_sr  # Weak label: use actual audio duration
            )
            text_tensor = build_text_tensor(
                sample["syllables"], sample["bpm"], _tokenizer,
                has_score=sample["has_score"],
            )
            if text_tensor.shape[0] == 0:
                return None

            return {
                "sample": sample,
                "waveform": waveform,
                "text_tensor": text_tensor,
                "estimated_duration": estimated_duration,
            }
        except Exception as e:
            print(f"Error loading {sample.get('audio_path', '?')}: {e}")
            return None

    # -- Helper: encode a mini-batch and assemble results ---------------
    def _encode_and_assemble(mini_batch: List[Dict]):
        """Run batch VAE encode + write results incrementally to Arrow."""
        nonlocal n_processed, n_skipped

        # Phase 2: Batch VAE encode
        t_vae_start = time.perf_counter()
        waveforms = [item["waveform"] for item in mini_batch]
        audio_results = preprocessor.encode_audio_batch(waveforms)
        t_vae_end = time.perf_counter()
        timing_totals["vae_encode"] += t_vae_end - t_vae_start
        timing_counts["vae_encode"] += len(mini_batch)

        # Phase 3: Assemble into a small batch dict, then write immediately
        t_asm_start = time.perf_counter()
        batch = _create_empty_batch()
        n_ok = 0
        for item, audio_pair in zip(mini_batch, audio_results):
            try:
                sample = item["sample"]
                has_score = sample.get("has_score", True)
                result = preprocessor.process_sample(
                    text_tensor=item["text_tensor"],
                    is_prompt=False,
                    precomputed_audio=audio_pair,
                    has_score=has_score,
                )
                svs_seq_ids = result["packed_text_tokens"].numpy()
                text_mask_np = result["text_mask"].numpy()
                text_token_ids = svs_seq_ids[text_mask_np == 1].tolist()
                if text_token_ids:
                    text_token_ids = text_token_ids[:-1]
                svs_prompt = preprocessor.tokenizer.decode(text_token_ids, skip_special_tokens=False)

                batch["packed_text_tokens"].append(result["packed_text_tokens"].numpy().astype('int32'))
                batch["audio_feats"].append(result["audio_feats"].numpy().astype('float16'))
                batch["text_mask"].append(result["text_mask"].numpy().astype('int32'))
                batch["audio_mask"].append(result["audio_mask"].numpy().astype('int32'))
                batch["loss_mask"].append(result["loss_mask"].numpy().astype('int32'))
                batch["labels"].append(result["labels"].numpy().astype('int32'))
                batch["audio_duration"].append(float(result["audio_duration"]))
                batch["estimated_duration"].append(float(item["estimated_duration"]))
                batch["text_token_count"].append(int(result["text_token_count"]))
                batch["total_length"].append(int(result["total_length"]))
                batch["item_name"].append(sample["item_name"])
                batch["song_name"].append(sample.get("song_name", ""))
                batch["song_folder"].append(sample["song_folder"])
                batch["dataset_name"].append(sample.get("dataset_name", "unknown"))
                batch["svs_prompt"].append(svs_prompt)
                batch["bpm"].append(int(sample["bpm"]))
                batch["word"].append(sample.get("word", []))
                batch["pitch"].append([int(p) for p in sample.get("pitch", [])])
                batch["pitch_dur"].append([float(d) for d in sample.get("pitch_dur", [])])
                batch["note"].append(sample.get("note", sample.get("notes", [])))
                batch["pitch2word"].append([int(p) for p in sample.get("pitch2word", [])])
                batch["word_dur"].append([float(d) for d in sample.get("word_dur", [])])
                batch["has_score"].append(has_score)
                n_ok += 1
                n_processed += 1
            except Exception as e:
                print(f"Error assembling {item['sample']['audio_path']}: {e}")
                n_skipped += 1
                continue
        t_asm_end = time.perf_counter()
        timing_totals["assemble"] += t_asm_end - t_asm_start
        timing_counts["assemble"] += len(mini_batch)

        # Incremental Arrow write (tiny batch -> negligible overhead)
        if n_ok > 0:
            t0 = time.perf_counter()
            shard_writer.write_batch(batch, n_ok)
            t1 = time.perf_counter()
            timing_totals["shard_write"] += t1 - t0
            timing_counts["shard_write"] += n_ok

    # ==================================================================
    # Main loop: ThreadPool prefetch + dynamic batch sizing
    # ==================================================================
    PREFETCH = max(num_workers * 8, vae_batch_size * 4)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        # Sliding window of futures -- bounded memory usage
        future_queue = deque()
        submit_idx = 0

        # Seed the prefetch window
        while submit_idx < min(PREFETCH, len(samples)):
            future_queue.append(pool.submit(_prepare_one, samples[submit_idx]))
            submit_idx += 1

        mini_batch = []
        batch_tokens = 0

        while future_queue:
            # Wait for next result (in order)
            t_cpu_start = time.perf_counter()
            item = future_queue.popleft().result()
            if pbar is not None:
                pbar.update(1)

            # Keep prefetch window full
            if submit_idx < len(samples):
                future_queue.append(pool.submit(_prepare_one, samples[submit_idx]))
                submit_idx += 1

            t_cpu_end = time.perf_counter()

            if item is None:
                n_skipped += 1
                continue

            timing_totals["cpu_prep"] += t_cpu_end - t_cpu_start
            timing_counts["cpu_prep"] += 1

            wav_len = item["waveform"].size(-1)

            # Dynamic batch sizing: flush if adding this sample would exceed limits
            if mini_batch and (batch_tokens + wav_len > vae_max_tokens
                               or len(mini_batch) >= vae_batch_size):
                _encode_and_assemble(mini_batch)
                mini_batch = []
                batch_tokens = 0

            mini_batch.append(item)
            batch_tokens += wav_len

        # Flush remaining mini-batch
        if mini_batch:
            _encode_and_assemble(mini_batch)

    return n_processed, n_skipped


def _print_timing_report(total_processed, wall_total, timing_totals, timing_counts, tag=""):
    print(f"\n{'='*70}")
    print(f"  FINAL Timing Report{tag}  ({total_processed} samples, wall: {wall_total:.1f}s)")
    print(f"{'='*70}")
    print(f"  {'Step':<16s}  {'Total (s)':>10s}  {'Avg (ms)':>10s}  {'% Wall':>8s}  {'Count':>6s}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}")
    for k in _TIMING_KEYS:
        tot = timing_totals[k]
        cnt = timing_counts[k]
        avg_ms = (tot / cnt * 1000) if cnt > 0 else 0
        pct = (tot / wall_total * 100) if wall_total > 0 else 0
        print(f"  {k:<16s}  {tot:>10.2f}  {avg_ms:>10.2f}  {pct:>7.1f}%  {cnt:>6d}")
    accounted = sum(timing_totals.values())
    overhead = wall_total - accounted
    if wall_total > 0:
        print(f"  {'overhead':<16s}  {overhead:>10.2f}  {'':>10s}  {overhead/wall_total*100:>7.1f}%")
    print(f"  {'TOTAL WALL':<16s}  {wall_total:>10.2f}")
    if wall_total > 0:
        print(f"  Throughput: {total_processed/wall_total:.1f} samples/sec")
    print(f"{'='*70}")


def _write_dataset_metadata(output_path: Path, num_rows: int, shard_paths: List[str]):
    """Write HuggingFace-compatible dataset_info.json / state.json."""
    features = _build_features()
    dataset_info = {
        "description": "SVS preprocessed dataset",
        "features": features.to_dict(),
        "num_rows": num_rows,
        "num_shards": len(shard_paths),
    }
    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    state = {
        "_data_files": [{"filename": Path(p).name} for p in shard_paths],
        "_fingerprint": None,
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    with open(output_path / "state.json", "w") as f:
        json.dump(state, f, indent=2)


def process_and_save(
    samples: List[Dict],
    preprocessor: SVSPreprocessor,
    output_path: Path,
    num_workers: int = 4,
    shard_size: int = 1000,
    vae_batch_size: int = 32,
    vae_max_tokens: int = 44100 * 100,
    rank_prefix: str = "",
):
    """
    Process samples and save to Arrow format using chunked/streaming writes.

    Uses ThreadPoolExecutor for async audio prefetching (overlaps I/O with GPU)
    and dynamic batch sizing based on total audio tokens (prevents OOM for
    long sequences while maximising GPU utilisation for short ones).

    Args:
        samples: List of raw sample dictionaries
        preprocessor: SVSPreprocessor instance
        output_path: Path to save the Arrow dataset
        num_workers: Number of workers for parallel audio loading
        shard_size: Number of samples per shard (default 1000)
        vae_batch_size: Hard cap on max samples per VAE batch (default 32)
        vae_max_tokens: Max total waveform samples per VAE batch (dynamic sizing)
        rank_prefix: Optional prefix for shard filenames (for multi-GPU)
    """
    output_path.mkdir(parents=True, exist_ok=True)

    features = _build_features()
    shard_writer = _ShardWriter(output_path, shard_size, features.arrow_schema, rank_prefix)

    # Sort by estimated audio duration so consecutive mini-batches contain
    # similar-length waveforms, minimising padding waste.
    samples.sort(key=_estimate_sort_duration)
    durations = [_estimate_sort_duration(s) for s in [samples[0], samples[-1]]]
    print(f"Sorted samples by estimated duration: {durations[0]:.2f}s - {durations[1]:.2f}s")
    print(f"Processing {len(samples)} samples with shard_size={shard_size}, "
          f"vae_batch_size(max)={vae_batch_size}, vae_max_tokens={vae_max_tokens}")

    timing_totals = {k: 0.0 for k in _TIMING_KEYS}
    timing_counts = {k: 0 for k in _TIMING_KEYS}
    wall_start = time.perf_counter()

    pbar = tqdm(total=len(samples), desc="Processing samples")
    total_processed, skipped = _encode_into_shards(
        samples, preprocessor, shard_writer,
        num_workers=num_workers,
        vae_batch_size=vae_batch_size,
        vae_max_tokens=vae_max_tokens,
        pbar=pbar,
        timing_totals=timing_totals,
        timing_counts=timing_counts,
    )
    pbar.close()

    shard_writer.close()
    shard_paths = shard_writer.shard_paths

    wall_total = time.perf_counter() - wall_start
    print(f"\nProcessed {total_processed} samples across {shard_writer.num_shards} shards, skipped {skipped}")
    _print_timing_report(total_processed, wall_total, timing_totals, timing_counts)

    if total_processed == 0:
        print("No samples processed!")
        return

    _write_dataset_metadata(output_path, total_processed, shard_paths)
    print(f"Saved to {output_path}")


def _gpu_worker_loop(
    rank: int,
    world_size: int,
    task_queue,
    pretrained_path: str,
    sample_rate: int,
    output_path: Path,
    num_workers: int,
    shard_size: int,
    vae_batch_size: int,
    vae_max_tokens: int,
):
    """Dynamic multi-GPU worker: pull chunks off ``task_queue`` until a ``None``
    sentinel, encoding each into this rank's own rolling shards.

    Fast cards drain more chunks than slow/contended ones, so the work
    self-balances. Metadata is written by the parent (this worker only writes
    ``data-r{rank}-*.arrow`` shards).
    """
    device = f"cuda:{rank}"
    print(f"[GPU {rank}] worker starting on {device}")

    preprocessor = SVSPreprocessor(
        pretrained_path=pretrained_path,
        sample_rate=sample_rate,
        device=device,
    )
    features = _build_features()
    shard_writer = _ShardWriter(output_path, shard_size, features.arrow_schema,
                                rank_prefix=f"r{rank}-")

    timing_totals = {k: 0.0 for k in _TIMING_KEYS}
    timing_counts = {k: 0 for k in _TIMING_KEYS}
    wall_start = time.perf_counter()
    pbar = tqdm(desc=f"[GPU {rank}]", position=rank)

    total_processed = 0
    skipped = 0
    chunks_done = 0
    per_rank_workers = max(1, num_workers // world_size)

    while True:
        chunk = task_queue.get()
        if chunk is None:  # sentinel: no more work
            break
        n_proc, n_skip = _encode_into_shards(
            chunk, preprocessor, shard_writer,
            num_workers=per_rank_workers,
            vae_batch_size=vae_batch_size,
            vae_max_tokens=vae_max_tokens,
            pbar=pbar,
            timing_totals=timing_totals,
            timing_counts=timing_counts,
        )
        total_processed += n_proc
        skipped += n_skip
        chunks_done += 1

    pbar.close()
    shard_writer.close()
    wall_total = time.perf_counter() - wall_start
    print(f"[GPU {rank}] Done: {total_processed} samples, {chunks_done} chunks, "
          f"{shard_writer.num_shards} shards, skipped {skipped}")
    _print_timing_report(total_processed, wall_total, timing_totals, timing_counts,
                         tag=f" [GPU {rank}]")


def process_and_save_multigpu(
    samples: List[Dict],
    params: Dict,
    output_path: Path,
    num_gpus: int,
):
    """
    Multi-GPU parallel preprocessing via a dynamic work queue.

    The parent sorts samples globally by duration, splits them into
    duration-homogeneous chunks, and feeds them into a bounded ``mp.Queue``.
    ``num_gpus`` workers each pull chunks on demand (one VAE per GPU) and write
    rank-prefixed shards into ``output_path``. After all workers finish, the
    parent generates the dataset metadata. Because dispatch is dynamic, a
    fast/uncontended GPU processes more chunks than a slow/shared one, so the
    wall-clock is bounded by aggregate throughput rather than the slowest card.
    """
    import torch.multiprocessing as mp

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nMulti-GPU processing (dynamic dispatch): {num_gpus} GPUs, {len(samples)} samples")

    # Global duration sort so each contiguous chunk is length-homogeneous
    # (keeps per-VAE-batch padding tight inside every worker).
    samples.sort(key=_estimate_sort_duration)
    durations = [_estimate_sort_duration(s) for s in [samples[0], samples[-1]]]
    print(f"  Sorted by estimated duration: {durations[0]:.2f}s - {durations[1]:.2f}s")

    chunk_size = int(params.get("dispatch_chunk_size", 2000))
    chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]
    print(f"  {len(chunks)} chunks of up to {chunk_size} samples; queue maxsize={num_gpus * 3}")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=num_gpus * 3)

    procs = []
    for rank in range(num_gpus):
        p = ctx.Process(
            target=_gpu_worker_loop,
            args=(
                rank,
                num_gpus,
                task_queue,
                params["pretrained_path"],
                params["sample_rate"],
                output_path,
                params["num_workers"],
                params["shard_size"],
                params["vae_batch_size"],
                params["vae_max_tokens"],
            ),
        )
        p.start()
        procs.append(p)

    # Feeder thread: push chunks (blocking on maxsize) then one sentinel per
    # worker. Aborts early if a worker has already died so it never blocks
    # forever against a full queue with no live consumer.
    def _worker_died():
        return any(p.exitcode is not None and p.exitcode != 0 for p in procs)

    def _feed():
        for c in chunks:
            while True:
                if _worker_died():
                    return
                try:
                    task_queue.put(c, timeout=5)
                    break
                except _queue.Full:
                    continue
        for _ in range(num_gpus):
            while True:
                if _worker_died():
                    return
                try:
                    task_queue.put(None, timeout=5)
                    break
                except _queue.Full:
                    continue

    feeder = threading.Thread(target=_feed, name="chunk-feeder", daemon=True)
    feeder.start()

    # Wait for workers; if any crashes, terminate the rest so blocked
    # ``queue.get()`` calls don't deadlock the join.
    while any(p.is_alive() for p in procs):
        if _worker_died():
            for p in procs:
                if p.is_alive():
                    p.terminate()
            break
        time.sleep(2)
    for p in procs:
        p.join()
    feeder.join(timeout=10)

    failed = [(rank, p.exitcode) for rank, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"Multi-GPU worker(s) failed (rank, exitcode): {failed}")

    # After all workers finish, generate dataset metadata from the shards.
    import pyarrow as pa

    shard_files = sorted(output_path.glob("data-*.arrow"))
    shard_paths = [str(p) for p in shard_files]

    total_samples = 0
    for sp in shard_files:
        with pa.OSFile(str(sp), 'rb') as f:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()
            total_samples += len(table)

    print(f"\nAll GPUs done. {total_samples} samples across {len(shard_paths)} shards.")
    _write_dataset_metadata(output_path, total_samples, shard_paths)
    print(f"Saved to {output_path}")
