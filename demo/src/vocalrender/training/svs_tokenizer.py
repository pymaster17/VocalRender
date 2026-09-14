"""
SVS tokenizer extension and mask-token collection for VoxCPM training.

Centralises the logic for:
1. Extending a base tokenizer with SVS-specific tokens (pitch, note, BPM,
   special tokens).
2. Resizing model embeddings with physically-motivated initialisation.
3. Collecting token-ID sets needed by the SVS masking strategies.

Training and inference scripts call into this module instead of
duplicating the tokenizer setup inline.
"""

from dataclasses import dataclass, field
from typing import Set

from vocalrender.model.svs_utils import get_svs_token_maps, resize_token_embeddings_with_svs_init


@dataclass
class SVSTokenArtifacts:
    """Artifacts produced by :func:`extend_tokenizer_for_svs`."""

    base_tokenizer: object
    """The tokenizer after SVS tokens have been added."""

    num_added: int
    """Number of new tokens added to the tokenizer."""

    pitch_tokens: list
    """List of pitch token strings (e.g. ``<P_60>``)."""

    note_tokens: list
    """List of note-duration token strings (e.g. ``<NOTE_4>``)."""

    bpm_tokens: list
    """List of BPM token strings (e.g. ``<BPM_120>``)."""

    special_tokens: list
    """List of SVS special token strings (e.g. ``<SVS_MASK>``)."""


@dataclass
class SVSMaskTokenIDs:
    """Token-ID sets needed by the SVS masking strategies."""

    pitch_token_ids: Set[int] = field(default_factory=set)
    note_token_ids: Set[int] = field(default_factory=set)
    bpm_token_ids: Set[int] = field(default_factory=set)
    mask_token_id: int = 0


def extend_tokenizer_for_svs(
    model,
    tokenizer_wrapper,
) -> SVSTokenArtifacts:
    """Extend *tokenizer_wrapper* with SVS tokens and resize model embeddings.

    Parameters
    ----------
    model
        A ``VoxCPMModel`` or ``VoxCPM2Model`` instance.  Its embedding
        layer will be resized in-place.
    tokenizer_wrapper
        Either a ``CharTokenizerWrapper`` (which wraps a HuggingFace
        tokenizer) or a direct ``PreTrainedTokenizer``.

    Returns
    -------
    SVSTokenArtifacts
        Contains the unwrapped base tokenizer, token lists, and the
        count of newly added tokens.
    """
    pitch_tokens, note_tokens, bpm_tokens, _, special_tokens = get_svs_token_maps()
    new_tokens = pitch_tokens + note_tokens + bpm_tokens + special_tokens

    # Handle CharTokenizerWrapper: access underlying tokenizer
    if hasattr(tokenizer_wrapper, "tokenizer"):
        base_tokenizer = tokenizer_wrapper.tokenizer
    else:
        base_tokenizer = tokenizer_wrapper

    num_added = base_tokenizer.add_tokens(new_tokens)

    # Resize embeddings and initialise SVS tokens with physical priors
    resize_token_embeddings_with_svs_init(model, base_tokenizer)

    return SVSTokenArtifacts(
        base_tokenizer=base_tokenizer,
        num_added=num_added,
        pitch_tokens=pitch_tokens,
        note_tokens=note_tokens,
        bpm_tokens=bpm_tokens,
        special_tokens=special_tokens,
    )


def collect_svs_mask_token_ids(
    base_tokenizer,
    artifacts: SVSTokenArtifacts,
) -> SVSMaskTokenIDs:
    """Build token-ID sets for the SVS masking strategies.

    Should only be called when masking is actually enabled (i.e. the
    ``svs_mask_strategy`` is not ``"none"``).

    Parameters
    ----------
    base_tokenizer
        The tokenizer (after SVS extension).
    artifacts : SVSTokenArtifacts
        Token lists returned by :func:`extend_tokenizer_for_svs`.

    Returns
    -------
    SVSMaskTokenIDs
        Dataclass with ``pitch_token_ids``, ``note_token_ids``,
        ``bpm_token_ids``, and ``mask_token_id``.

    Raises
    ------
    ValueError
        If the ``<SVS_MASK>`` token is not found in the tokenizer.
    """
    unk_id = base_tokenizer.unk_token_id

    pitch_token_ids: Set[int] = set()
    for pt in artifacts.pitch_tokens:
        pid = base_tokenizer.convert_tokens_to_ids(pt)
        if pid != unk_id:
            pitch_token_ids.add(pid)

    note_token_ids: Set[int] = set()
    for nt in artifacts.note_tokens:
        nid = base_tokenizer.convert_tokens_to_ids(nt)
        if nid != unk_id:
            note_token_ids.add(nid)

    bpm_token_ids: Set[int] = set()
    for bt in artifacts.bpm_tokens:
        bid = base_tokenizer.convert_tokens_to_ids(bt)
        if bid != unk_id:
            bpm_token_ids.add(bid)

    mask_token_id = base_tokenizer.convert_tokens_to_ids("<SVS_MASK>")
    if mask_token_id == unk_id:
        raise ValueError(
            "<SVS_MASK> token not found in tokenizer. "
            "Make sure svs_utils.py includes it."
        )

    return SVSMaskTokenIDs(
        pitch_token_ids=pitch_token_ids,
        note_token_ids=note_token_ids,
        bpm_token_ids=bpm_token_ids,
        mask_token_id=mask_token_id,
    )
