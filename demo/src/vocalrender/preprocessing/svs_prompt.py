"""
SVS prompt reconstruction from preprocessed metadata.

Provides :func:`rebuild_svs_prompt` for reconstructing the SVS text prompt
from stored annotation fields (word, pitch, note, pitch2word, bpm).
"""

from typing import Dict

from .text_tensor import build_text_tensor
from .svs_preprocessor import create_lightweight_preprocessor


def rebuild_svs_prompt(
    sample: Dict,
    tokenizer,
    preprocessor=None,
    force_lyrics_only: bool = False,
) -> str:
    """Rebuild SVS prompt string from metadata stored in preprocessed dataset.

    Works with both JSON label entries and preprocessed Arrow samples.

    Args:
        sample: Dict with word, pitch, note, pitch2word, bpm, has_score fields.
        tokenizer: LlamaTokenizerFast with SVS tokens added.
        preprocessor: Optional SVSPreprocessor (created if None).
        force_lyrics_only: If True, always use lyrics-only mode (``<SVS_MASK>``
            for BPM/pitch/note), regardless of ``has_score`` in the sample.

    Returns:
        SVS prompt string (without trailing ``<audio_start>``).
    """
    from vocalrender.training.svs_data import convert_annotation_to_syllables

    words = list(sample.get("word", []))
    bpm = int(sample.get("bpm", 120))

    # Decide has_score: respect metadata, but override when force_lyrics_only
    has_score = bool(
        sample.get("pitch") and sample.get("note") and sample.get("has_score", True)
    ) and not force_lyrics_only

    if has_score:
        pitches = [int(p) for p in sample["pitch"]]
        notes = [str(n) for n in sample["note"]]
        pitch2word = [int(p) for p in sample.get("pitch2word", list(range(len(pitches))))]
        syllables = convert_annotation_to_syllables(words, pitches, notes, pitch2word)
    else:
        syllables = [{"char": w} for w in words]

    text_tensor = build_text_tensor(syllables, bpm, tokenizer, has_score=has_score)
    if text_tensor.shape[0] == 0:
        return ""

    if preprocessor is None:
        preprocessor = create_lightweight_preprocessor(tokenizer)

    svs_seq = preprocessor.build_svs_sequence(
        text_tensor, is_prompt=False, has_score=has_score,
    )

    # Strip trailing audio_start token (model adds it during generation)
    seq_ids = svs_seq.tolist()
    if seq_ids:
        seq_ids = seq_ids[:-1]

    return tokenizer.decode(seq_ids, skip_special_tokens=False)
