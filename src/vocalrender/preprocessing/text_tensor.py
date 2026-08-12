"""
Text tensor construction for SVS preprocessing.

Provides :func:`build_text_tensor` (used during preprocessing and inference)
and :func:`estimate_duration_from_notes` (used for sorting / duration estimation).
"""

from typing import Dict, List

import torch


def estimate_duration_from_notes(notes: List[str], bpm: int) -> float:
    """
    Estimate audio duration from the raw note token list and BPM.

    Each entry in `notes` corresponds to one pitch segment, so we just
    sum up their durations directly — no need to go through syllables.

    Args:
        notes: Raw list of note token strings, e.g. ["<NOTE_4>", "<NOTE_8>", ...]
        bpm: Beats per minute

    Returns:
        Estimated duration in seconds
    """
    from vocalrender.model.svs_utils import get_svs_token_maps

    _, _, _, dur_units, _ = get_svs_token_maps()

    # Calculate seconds per quarter note
    unit_duration = 60.0 / bpm

    # Sum up all note durations
    total_units = sum(dur_units.get(n, 1.0) for n in notes)

    return total_units * unit_duration


def build_text_tensor(
    syllables: List[Dict], bpm: int, tokenizer, has_score: bool = True,
) -> torch.Tensor:
    """Build text tensor from syllables.

    For weak-label samples (has_score=False), syllables only have 'char' field.
    pitch/note/bpm columns are filled with 0.
    """
    from vocalrender.training.svs_data import expand_syllables
    from vocalrender.model.svs_utils import get_svs_token_maps

    _, note_tokens_list, _, _, _ = get_svs_token_maps()

    if has_score:
        expanded = expand_syllables(syllables)
    else:
        # Weak label: no melisma, just enumerate words
        expanded = [{'char': s['char'], 'word_idx': i} for i, s in enumerate(syllables)]

    if not expanded:
        return torch.zeros((0, 5), dtype=torch.long)

    # Tokenize all chars
    tokenized = []
    max_text_tokens = 1
    for syl in expanded:
        char = syl['char']
        ids = tokenizer.encode(char, add_special_tokens=False)
        tokenized.append(ids)
        max_text_tokens = max(max_text_tokens, len(ids))

    # Build rows
    # Layout: [text_ids..., pitch, note_idx, bpm, word_idx]
    rows = []
    for i, syl in enumerate(expanded):
        ids = tokenized[i]
        padded_ids = ids + [0] * (max_text_tokens - len(ids))

        word_idx = int(syl.get('word_idx', i))

        if has_score:
            pitch = int(syl['pitch'])
            note_str = syl['note']
            # Find note token index in the list
            if isinstance(note_str, str) and note_str in note_tokens_list:
                note_idx = note_tokens_list.index(note_str)
            else:
                note_idx = 6  # Default to <NOTE_4> which is index 6 in the list
        else:
            # Weak label: pitch/note are unknown, fill with 0
            pitch = 0
            note_idx = 0

        row = padded_ids + [pitch, note_idx, bpm, word_idx]
        rows.append(row)

    return torch.tensor(rows, dtype=torch.long)
