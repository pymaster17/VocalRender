"""
Raw folder-based SVS data loading and annotation processing utilities.

This module contains the original folder-based SVS data pipeline, including:

* Annotation parsing: :func:`convert_annotation_to_syllables`,
  :func:`convert_words_to_syllables`, :func:`expand_syllables`
* Text tensor construction: :func:`build_svs_text_tensor`,
  :func:`tokenize_svs_batch`, :func:`get_note_duration_map`
* Dataset / DataLoader: :class:`HFSVSDataset`, :func:`load_svs_datasets`,
  :func:`build_svs_dataloader`

The preprocessed Arrow-based pipeline (used by active training) remains in
:mod:`vocalrender.training.svs_data`.  All public names here are re-exported from
``svs_data`` for backward compatibility.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, Audio
from transformers import PreTrainedTokenizer

DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_ID_COLUMN = "dataset_id"


# ============================================================
# Annotation helpers (shared with preprocessed pipeline)
# ============================================================

def convert_annotation_to_syllables(
    words: List[str],
    pitches: List[int],
    notes: List[str],
    pitch2word: List[int],
) -> List[Dict]:
    """
    Convert folder-based annotation format to syllables format.

    The annotation format has separate lists:
    - words: ["下", "一", "步", "AP", "慢", "慢", ...]
    - pitches: [57, 57, 54, 0, 59, 60, ...]  per pitch segment
    - notes: ["<NOTE_4>", "<NOTE_8>", ...]  per pitch segment
    - pitch2word: [0, 1, 2, 3, 4, 5, ...]  maps each pitch to word index

    Returns syllables in the format:
    [{"char": "下", "pitch": 57, "note": "<NOTE_4>"}, ...]

    This handles melisma (one word -> multiple pitches) by checking pitch2word mapping.
    """
    syllables = []

    # Build word to pitch indices mapping
    word_to_pitches = {}
    for pitch_idx, word_idx in enumerate(pitch2word):
        if word_idx not in word_to_pitches:
            word_to_pitches[word_idx] = []
        word_to_pitches[word_idx].append(pitch_idx)

    # Generate syllables for each word
    for word_idx, word in enumerate(words):
        pitch_indices = word_to_pitches.get(word_idx, [])

        if not pitch_indices:
            # Word has no pitch mapping, skip
            continue

        if len(pitch_indices) == 1:
            # Single pitch for this word
            pitch_idx = pitch_indices[0]
            pitch = pitches[pitch_idx] if pitch_idx < len(pitches) else 0
            note = notes[pitch_idx] if pitch_idx < len(notes) else "<NOTE_4>"
            syllables.append({
                "char": word,
                "pitch": pitch,
                "note": note,
            })
        else:
            # Melisma: multiple pitches for one word
            pitch_list = [pitches[i] if i < len(pitches) else 0 for i in pitch_indices]
            note_list = [notes[i] if i < len(notes) else "<NOTE_4>" for i in pitch_indices]
            syllables.append({
                "char": word,
                "pitch": pitch_list,
                "note": note_list,
            })

    return syllables


def convert_words_to_syllables(words: List[str]) -> List[Dict]:
    """Convert word-only annotation to syllables (no pitch/note info).

    For weak-label datasets that only have word annotations.
    Each syllable only has a 'char' field — no 'pitch' or 'note'.
    """
    return [{"char": w} for w in words]


def expand_syllables(syllables: List[Dict]) -> List[Dict]:
    """
    Expand syllables with array pitch/note (melisma) into individual rows.

    Each expanded row includes a ``word_idx`` field that tracks which original
    syllable (word) it came from.  Rows sharing the same ``word_idx`` are true
    melisma; rows with consecutive but *different* ``word_idx`` values are
    distinct words even if the character text is identical.

    Example: {"char": "几", "pitch": [60, 64], "note": ["<NOTE_4>", "<NOTE_8>"]}
    -> [{"char": "几", "pitch": 60, "note": "<NOTE_4>", "word_idx": 0},
        {"char": "几", "pitch": 64, "note": "<NOTE_8>", "word_idx": 0}]
    """
    expanded = []
    for word_idx, syl in enumerate(syllables):
        pitch = syl['pitch']
        note = syl['note']
        char = syl['char']

        if isinstance(pitch, list):
            # Melisma: one char -> multiple notes
            notes = note if isinstance(note, list) else [note] * len(pitch)
            for p, n in zip(pitch, notes):
                expanded.append({'char': char, 'pitch': p, 'note': n, 'word_idx': word_idx})
        else:
            expanded.append({'char': char, 'pitch': pitch, 'note': note, 'word_idx': word_idx})

    return expanded


def get_note_duration_map() -> Dict[str, float]:
    """
    Get note token to duration value mapping from svs_utils.

    Returns the dur_units dict from svs_utils.get_svs_token_maps().
    Unit: Quarter note = 1.0
    """
    from vocalrender.model.svs_utils import get_svs_token_maps
    _, _, _, dur_units, _ = get_svs_token_maps()
    return dur_units


# ============================================================
# Text tensor construction
# ============================================================

def build_svs_text_tensor(
    syllables: List[Dict],
    bpm: int,
    tokenizer: PreTrainedTokenizer,
) -> torch.Tensor:
    """
    Build text tensor with variable-width text columns.

    Returns tensor of shape [L, max_text_cols + 4] where:
    - Columns 0 to max_text_cols-1: Text token IDs (0 = padding)
    - Column max_text_cols: Pitch MIDI value (0-127)
    - Column max_text_cols+1: Note duration value (scaled by 8, e.g., quarter=8)
    - Column max_text_cols+2: BPM value (global, same for all rows)
    - Column max_text_cols+3: Word index (from expand_syllables, for melisma detection)

    Note duration values are scaled: dur_units * 8 (quarter note = 8)
    """
    # Expand melisma
    expanded = expand_syllables(syllables)

    if not expanded:
        # Return empty tensor if no syllables
        return torch.zeros((0, 5), dtype=torch.long)

    # Get duration mapping
    dur_units = get_note_duration_map()

    # First pass: tokenize all chars to find max tokens
    tokenized = []
    max_text_tokens = 1
    for syl in expanded:
        char = syl['char']
        ids = tokenizer.encode(char, add_special_tokens=False)
        tokenized.append(ids)
        max_text_tokens = max(max_text_tokens, len(ids))

    # Second pass: build padded rows
    # Layout: [text_ids..., pitch, note_val, bpm, word_idx]
    rows = []
    for i, syl in enumerate(expanded):
        ids = tokenized[i]
        padded_ids = ids + [0] * (max_text_tokens - len(ids))

        pitch = int(syl['pitch'])
        word_idx = int(syl.get('word_idx', i))

        # Parse note duration: use dur_units mapping, scale by 8
        note_str = syl['note']
        if isinstance(note_str, str) and note_str in dur_units:
            # Scale by 8 so quarter note = 8 (integer friendly)
            note_val = int(dur_units[note_str] * 8)
        else:
            # Default to quarter note
            note_val = 8

        row = padded_ids + [pitch, note_val, bpm, word_idx]
        rows.append(row)

    return torch.tensor(rows, dtype=torch.long)


def tokenize_svs_batch(
    batch: Dict,
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    """
    Tokenize a batch of SVS samples.

    For each sample, converts syllables to a 2D text tensor.
    Returns a dict with 'text_tensor' as a list of tensors (variable shapes).
    """
    text_tensors = []

    # Deserialize JSON strings
    syllables_list = [json.loads(s) for s in batch["syllables_json"]]

    for syllables, bpm in zip(syllables_list, batch["bpm"]):
        tensor = build_svs_text_tensor(syllables, bpm, tokenizer)
        # Convert to list for HF datasets compatibility
        text_tensors.append(tensor.tolist())

    return {"text_tensor": text_tensors}


# ============================================================
# Raw folder-based dataset loading
# ============================================================

def load_svs_datasets(
    dataset_root: str,
    val_samples: int = 0,
    sample_rate: int = 44100,
    tokenizer: PreTrainedTokenizer = None,
    audio_extensions: Tuple[str, ...] = (".flac", ".wav", ".mp3"),
    max_samples: int = -1,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load SVS datasets from a folder-based structure.

    Expected structure:
        dataset_root/
            song_folder_1/
                metadata.json (contains bpm)
                segment_1.flac
                segment_1.json (contains word, pitch, note, pitch2word)
                segment_2.flac
                segment_2.json
                ...
            song_folder_2/
                ...

    JSON annotation format:
    {
        "item_name": "segment_name",
        "word": ["下", "一", "步", "AP", "SP"],  # includes AP/SP special tokens
        "pitch": [60, 62, 64, 0, 0],  # pitch values per pitch segment (0 for AP/SP)
        "note": ["<NOTE_4>", "<NOTE_8>", ...],  # note duration tokens
        "pitch2word": [0, 1, 2, 3, 4],  # pitch to word index mapping
    }

    Args:
        dataset_root: Root directory containing song folders
        val_samples: Number of samples for validation set (0 to disable)
        sample_rate: Target audio sample rate
        tokenizer: Tokenizer for text encoding (optional, used for pre-tokenization)
        audio_extensions: Tuple of audio file extensions to search for
        max_samples: Maximum number of samples to load (-1 for all)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")

    samples = []
    song_folders = [d for d in dataset_root.iterdir() if d.is_dir()]

    for song_folder in song_folders:
        # Try to load BPM from metadata.json
        metadata_path = song_folder / "metadata.json"
        bpm = 120  # default BPM
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    bpm = metadata.get("bpm", 120)
            except (json.JSONDecodeError, IOError):
                pass

        # Find all audio files and their corresponding JSON annotations
        for audio_file in song_folder.iterdir():
            if not audio_file.is_file():
                continue
            if audio_file.suffix.lower() not in audio_extensions:
                continue

            # Find corresponding JSON file
            json_path = audio_file.with_suffix(".json")
            if not json_path.exists():
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    annotation = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Extract required fields
            words = annotation.get("word", [])
            pitches = annotation.get("pitch", [])
            notes = annotation.get("note", [])
            pitch2word = annotation.get("pitch2word", [])

            if not words or not pitches or not notes:
                continue

            # Convert to syllables format for compatibility with existing processing
            syllables = convert_annotation_to_syllables(
                words=words,
                pitches=pitches,
                notes=notes,
                pitch2word=pitch2word,
            )

            samples.append({
                "audio": str(audio_file),
                "bpm": bpm,
                "syllables": syllables,
                "item_name": annotation.get("item_name", audio_file.stem),
                "song_folder": song_folder.name,
            })

            if max_samples > 0 and len(samples) >= max_samples:
                break

        if max_samples > 0 and len(samples) >= max_samples:
            break

    if not samples:
        raise ValueError(f"No valid samples found in {dataset_root}")

    # Split into train/val using absolute sample count
    if val_samples > 0:
        import random
        random.seed(42)
        random.shuffle(samples)
        # Ensure we don't take more val samples than available
        val_count = min(val_samples, len(samples) - 1)
        val_sample_list = samples[:val_count]
        train_samples = samples[val_count:]
    else:
        train_samples = samples
        val_sample_list = []

    # Convert to HuggingFace datasets
    def create_dataset(sample_list: List[Dict]) -> Dataset:
        if not sample_list:
            return None
        # Serialize syllables to JSON strings to avoid PyArrow mixed type errors
        ds = Dataset.from_dict({
            "audio": [s["audio"] for s in sample_list],
            "bpm": [s["bpm"] for s in sample_list],
            "syllables_json": [json.dumps(s["syllables"], ensure_ascii=False) for s in sample_list],
            "item_name": [s["item_name"] for s in sample_list],
            "song_folder": [s["song_folder"] for s in sample_list],
        })
        ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
        ds = ds.add_column("dataset_id", [0] * len(ds))
        ds = ds.add_column("task_type", ["svs"] * len(ds))
        return ds

    train_ds = create_dataset(train_samples)
    val_ds = create_dataset(val_sample_list) if val_sample_list else None

    return train_ds, val_ds


# ============================================================
# Raw SVS Dataset and DataLoader
# ============================================================

class HFSVSDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for SVS data.

    Returns samples with:
    - text_tensor: 2D tensor [L, cols]
    - audio_array: Audio waveform
    - task_type: "svs"
    - dataset_id: Dataset identifier
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        audio = item[DEFAULT_AUDIO_COLUMN]

        return {
            "text_tensor": torch.tensor(item["text_tensor"], dtype=torch.long),
            "audio_array": audio["array"],
            "audio_sampling_rate": audio["sampling_rate"],
            "dataset_id": item.get(DEFAULT_ID_COLUMN, 0),
            "task_type": item.get("task_type", "svs"),
            "is_prompt": item.get("is_prompt", False),
        }

    @staticmethod
    def pad_2d_sequences(seqs: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
        """
        Pad 2D sequences to the same shape [max_L, max_cols].
        """
        if not seqs:
            return torch.empty(0)

        max_L = max(seq.shape[0] for seq in seqs)
        max_cols = max(seq.shape[1] for seq in seqs)

        padded = []
        for seq in seqs:
            L, C = seq.shape
            if L < max_L or C < max_cols:
                new_seq = torch.full((max_L, max_cols), pad_value, dtype=seq.dtype)
                new_seq[:L, :C] = seq
                seq = new_seq
            padded.append(seq)

        return torch.stack(padded)

    @staticmethod
    def pad_1d_sequences(seqs: List[torch.Tensor], pad_value: float) -> torch.Tensor:
        """Pad 1D sequences to the same length."""
        if not seqs:
            return torch.empty(0)
        max_len = max(seq.shape[0] for seq in seqs)
        padded = []
        for seq in seqs:
            if seq.shape[0] < max_len:
                pad_width = (0, max_len - seq.shape[0])
                seq = torch.nn.functional.pad(seq, pad_width, value=pad_value)
            padded.append(seq)
        return torch.stack(padded)

    @classmethod
    def collate_fn(cls, batch: List[Dict]) -> Dict:
        """
        Collate SVS batch.

        Returns dict with:
        - text_tokens: [B, L, C] tensor (2D text tensors)
        - audio_tokens: [B, T] audio waveforms
        - task_ids: All set to 2 (SVS task ID)
        - dataset_ids: Dataset identifiers
        - is_prompts: List of bool
        """
        text_tensors = [sample["text_tensor"] for sample in batch]
        audio_tensors = [torch.tensor(sample["audio_array"], dtype=torch.float32) for sample in batch]
        dataset_ids = torch.tensor([sample["dataset_id"] for sample in batch], dtype=torch.int32)
        is_prompts = [bool(sample.get("is_prompt", False)) for sample in batch]

        text_padded = cls.pad_2d_sequences(text_tensors, pad_value=0)
        audio_padded = cls.pad_1d_sequences(audio_tensors, pad_value=-100.0)

        # SVS task_id = 2
        task_ids = torch.full((text_padded.size(0),), 2, dtype=torch.int32)

        return {
            "text_tokens": text_padded,
            "audio_tokens": audio_padded,
            "task_ids": task_ids,
            "dataset_ids": dataset_ids,
            "is_prompts": is_prompts,
        }


def build_svs_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """Build DataLoader for SVS dataset."""
    torch_dataset = HFSVSDataset(hf_dataset)
    return accelerator.prepare_dataloader(
        torch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=HFSVSDataset.collate_fn,
        drop_last=drop_last,
    )
