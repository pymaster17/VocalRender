"""
SVS (Singing Voice Synthesis) data loading and processing utilities.

This module contains the **preprocessed Arrow-based** SVS data pipeline used
by active training (``load_preprocessed_svs_datasets``,
``HFPreprocessedSVSDataset``, ``build_preprocessed_svs_dataloader``).

The raw folder-based pipeline and shared annotation helpers live in
:mod:`vocalrender.training.svs_raw_data` and are re-exported here for backward
compatibility.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset

from .svs_loading import load_preprocessed_svs_datasets


class HFPreprocessedSVSDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for preprocessed SVS data (from Arrow).

    This dataset loads pre-computed data directly without any on-the-fly processing.
    Optionally applies random pitch/note token masking during training.

    Masking — multi-strategy probabilistic selection:
        ``mask_strategy`` can be ``"none"`` (no masking) or a **list** of
        strategy names.  Each strategy has a per-sample selection
        probability defined in ``mask_params``; the remaining probability
        is assigned to "none".  Per sample exactly one strategy fires.

        Available strategies and their probability / rate keys:

        =========== ================ =============== ==========================
        Strategy    Selection prob   Internal rate   Effect
        =========== ================ =============== ==========================
        global      global_mask_prob (n/a — always   Mask ALL pitch/note/BPM +
                                      100 %)         melisma collapse
        paired      pair_mask_prob   pair_mask_rate  Mask (P,N) pairs together
        independent indep_mask_prob  pitch_mask_rate Mask pitch / note tokens
                                     note_mask_rate  independently
        =========== ================ =============== ==========================

    Prompt audio / voice conditioning:
        When ``prompt_audio_prob > 0`` a same-song audio segment is
        prepended at the sequence front — matching the V2 TTS
        pretraining layout.
    """
    
    # Special token IDs for prompt audio boundaries (pre-SVS: inherited
    # from VoxCPM2 pretraining as `<|audio_prompt_start|>` /
    # `<|audio_prompt_end|>` — semantically compatible with SVS prompt-audio
    # usage, so we keep the static assignment).
    PROMPT_AUDIO_START_ID = 103
    PROMPT_AUDIO_END_ID = 104
    # Token IDs for `<audio_start>` / `<audio_end>` — inherited from VoxCPM
    # pretraining (both V1 and V2).
    AUDIO_START_ID = 101
    AUDIO_END_ID = 102

    # Map from strategy name → mask_params key for selection probability
    _STRATEGY_PROB_KEYS = {
        "global": "global_mask_prob",
        "paired": "pair_mask_prob",
        "independent": "indep_mask_prob",
    }
    
    def __init__(
        self,
        dataset: Dataset,
        mask_strategy = "none",
        mask_params: Optional[Dict] = None,
        pitch_token_ids: Optional[set] = None,
        note_token_ids: Optional[set] = None,
        bpm_token_ids: Optional[set] = None,
        mask_token_id: int = 0,
        # Prompt audio configuration
        prompt_audio_prob: float = 0.0,
        prompt_max_frames: int = 50,
        song_index: Optional[Dict[str, List[int]]] = None,
        prompt_audio_seed: Optional[int] = None,
        text_tokenizer=None,
    ):
        self.dataset = dataset
        self.mask_params = mask_params or {}
        self.mask_token_id = mask_token_id
        
        # --- Build strategy probability table --------------------------
        # Accept str ("none" / legacy single strategy) or list of names.
        if isinstance(mask_strategy, list):
            active_strategies = mask_strategy
        elif mask_strategy and mask_strategy != "none":
            # Legacy single-strategy string
            active_strategies = [mask_strategy]
        else:
            active_strategies = []
        
        self._strategy_probs: List[Tuple[str, float]] = []
        total_prob = 0.0
        for name in active_strategies:
            key = self._STRATEGY_PROB_KEYS.get(name)
            if key is None:
                raise ValueError(f"Unknown mask strategy: {name!r}. "
                                 f"Choose from {list(self._STRATEGY_PROB_KEYS)}")
            prob = float(self.mask_params.get(key, 0.0))
            if prob > 0:
                self._strategy_probs.append((name, prob))
                total_prob += prob
        if total_prob > 1.0 + 1e-6:
            raise ValueError(f"Sum of mask strategy probabilities ({total_prob:.4f}) > 1.0")
        
        self._masking_enabled = len(self._strategy_probs) > 0
        
        # Prompt audio config
        self.prompt_audio_prob = prompt_audio_prob
        self.prompt_max_frames = prompt_max_frames
        self.song_index = song_index or {}
        self.prompt_audio_seed = prompt_audio_seed

        # text_tokenizer kept for forward compatibility — currently unused;
        # threaded in by the runner.
        self.text_tokenizer = text_tokenizer

        if not 0.0 <= self.prompt_audio_prob <= 1.0:
            raise ValueError(
                f"prompt_audio_prob must be in [0, 1], got {prompt_audio_prob}"
            )
        
        # Build reverse lookup: sample_idx -> song_name (for same-song prompt lookup)
        self._idx_to_song: Dict[int, str] = {}
        if self.song_index:
            for song_name, indices in self.song_index.items():
                for idx in indices:
                    self._idx_to_song[idx] = song_name
        
        # Pre-compute frozen sets for fast O(1) lookup via torch.isin
        if pitch_token_ids:
            self._pitch_ids_tensor = torch.tensor(
                sorted(pitch_token_ids), dtype=torch.int32
            )
        else:
            self._pitch_ids_tensor = torch.tensor([], dtype=torch.int32)

        if note_token_ids:
            self._note_ids_tensor = torch.tensor(
                sorted(note_token_ids), dtype=torch.int32
            )
        else:
            self._note_ids_tensor = torch.tensor([], dtype=torch.int32)

        if bpm_token_ids:
            self._bpm_ids_tensor = torch.tensor(
                sorted(bpm_token_ids), dtype=torch.int32
            )
        else:
            self._bpm_ids_tensor = torch.tensor([], dtype=torch.int32)

        # Union of all SVS-conditioning token IDs — pitch + note + BPM +
        # <SVS_MASK>.
        svs_cond_ids = set()
        svs_cond_ids.update(pitch_token_ids or set())
        svs_cond_ids.update(note_token_ids or set())
        svs_cond_ids.update(bpm_token_ids or set())
        if mask_token_id:
            svs_cond_ids.add(mask_token_id)
        if svs_cond_ids:
            self._svs_cond_ids_tensor = torch.tensor(
                sorted(svs_cond_ids), dtype=torch.int32
            )
        else:
            self._svs_cond_ids_tensor = torch.tensor([], dtype=torch.int32)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        
        # Convert lists back to tensors
        packed_text_tokens = torch.tensor(item["packed_text_tokens"], dtype=torch.int32)
        audio_feats = torch.tensor(item["audio_feats"], dtype=torch.float32)
        text_mask = torch.tensor(item["text_mask"], dtype=torch.int32)
        audio_mask = torch.tensor(item["audio_mask"], dtype=torch.int32)
        loss_mask = torch.tensor(item["loss_mask"], dtype=torch.int32)
        labels = torch.tensor(item["labels"], dtype=torch.int32)
        
        # Apply pitch/note masking (training-time data augmentation)
        # Skip for weak-label samples — their BPM/pitch/note are already <SVS_MASK>
        has_score = item.get("has_score", True)
        if has_score and self._masking_enabled:
            selected = self._select_mask_strategy()
            if selected == "global":
                # Global mask with melisma collapse — modifies all tensors
                packed_text_tokens, audio_feats, text_mask, audio_mask, loss_mask, labels = \
                    self._apply_global_mask_collapse(
                        packed_text_tokens, audio_feats, text_mask, audio_mask, loss_mask, labels)
            elif selected == "paired":
                packed_text_tokens = self._mask_paired(packed_text_tokens.clone())
            elif selected == "independent":
                packed_text_tokens = self._mask_independent(packed_text_tokens.clone())

        if self.prompt_audio_prob > 0:
            if self.prompt_audio_seed is not None:
                rng = random.Random(self.prompt_audio_seed + idx)
            else:
                rng = random

            if rng.random() < self.prompt_audio_prob:
                # Baseline SVS: prepend same-song prompt at the front.
                packed_text_tokens, audio_feats, text_mask, audio_mask, loss_mask, labels = \
                    self._prepend_prompt_audio(
                        idx, packed_text_tokens, audio_feats,
                        text_mask, audio_mask, loss_mask, labels,
                        rng=rng,
                    )

        total_length = packed_text_tokens.shape[0]

        sample: Dict = {
            "packed_text_tokens": packed_text_tokens,
            "audio_feats": audio_feats,
            "text_mask": text_mask,
            "audio_mask": audio_mask,
            "loss_mask": loss_mask,
            "labels": labels,
            "audio_duration": item["audio_duration"],
            "text_token_count": item["text_token_count"],
            "total_length": total_length,
        }
        return sample

    # ------------------------------------------------------------------
    # Prompt audio prepending
    # ------------------------------------------------------------------
    
    def _prepend_audio_prefix(
        self,
        prompt_audio_region: torch.Tensor,
        packed_text_tokens: torch.Tensor,
        audio_feats: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple:
        """Prepend an already-resolved audio conditioning region."""
        T_prompt = prompt_audio_region.shape[0]
        if T_prompt <= 0:
            return packed_text_tokens, audio_feats, text_mask, audio_mask, loss_mask, labels

        P, D = prompt_audio_region.shape[1], prompt_audio_region.shape[2]

        prompt_text = torch.tensor(
            [self.PROMPT_AUDIO_START_ID] + [0] * T_prompt + [self.PROMPT_AUDIO_END_ID],
            dtype=torch.int32,
        )
        prompt_text_mask = torch.tensor([1] + [0] * T_prompt + [1], dtype=torch.int32)
        prompt_audio_mask = torch.tensor([0] + [1] * T_prompt + [0], dtype=torch.int32)
        prompt_loss_mask = torch.zeros(T_prompt + 2, dtype=torch.int32)
        prompt_labels = torch.zeros(T_prompt + 2, dtype=torch.int32)
        prompt_audio_padded = torch.cat([
            torch.zeros(1, P, D, dtype=torch.float32),
            prompt_audio_region.to(torch.float32),
            torch.zeros(1, P, D, dtype=torch.float32),
        ], dim=0)

        packed_text_tokens = torch.cat([prompt_text, packed_text_tokens])
        audio_feats = torch.cat([prompt_audio_padded, audio_feats], dim=0)
        text_mask = torch.cat([prompt_text_mask, text_mask])
        audio_mask = torch.cat([prompt_audio_mask, audio_mask])
        loss_mask = torch.cat([prompt_loss_mask, loss_mask])
        labels = torch.cat([prompt_labels, labels])

        return packed_text_tokens, audio_feats, text_mask, audio_mask, loss_mask, labels

    def _extract_same_song_prompt(
        self,
        idx: int,
        rng=None,
    ) -> Optional[torch.Tensor]:
        """Pick a random non-self segment from the same song and return its
        audio latent region ``[T_prompt, P, D]``.

        Returns ``None`` when the song lookup fails, the song has no other
        segments, or the candidate's ``audio_mask`` contains no audio
        positions. Cropping respects ``prompt_max_frames``.
        """
        if rng is None:
            rng = random

        song_name = self._idx_to_song.get(idx)
        if not song_name:
            return None

        candidates = self.song_index.get(song_name, [])
        other_candidates = [c for c in candidates if c != idx]
        if not other_candidates:
            return None

        prompt_idx = rng.choice(other_candidates)
        prompt_item = self.dataset[prompt_idx]
        prompt_audio = torch.tensor(prompt_item["audio_feats"], dtype=torch.float32)
        prompt_audio_mask_raw = torch.tensor(prompt_item["audio_mask"], dtype=torch.int32)
        audio_positions = torch.where(prompt_audio_mask_raw == 1)[0]
        if len(audio_positions) == 0:
            return None

        start_pos = int(audio_positions[0].item())
        end_pos = int(audio_positions[-1].item()) + 1
        prompt_audio_region = prompt_audio[start_pos:end_pos]

        T_prompt = int(prompt_audio_region.shape[0])
        if T_prompt > self.prompt_max_frames:
            crop_start = rng.randint(0, T_prompt - self.prompt_max_frames)
            prompt_audio_region = prompt_audio_region[
                crop_start:crop_start + self.prompt_max_frames
            ]
        return prompt_audio_region

    def _prepend_prompt_audio(
        self,
        idx: int,
        packed_text_tokens: torch.Tensor,
        audio_feats: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        labels: torch.Tensor,
        rng=None,
    ) -> Tuple:
        """Prepend a same-song audio segment as prompt prefix."""
        prompt_audio_region = self._extract_same_song_prompt(idx, rng=rng)
        if prompt_audio_region is None:
            return packed_text_tokens, audio_feats, text_mask, audio_mask, loss_mask, labels

        return self._prepend_audio_prefix(
            prompt_audio_region,
            packed_text_tokens,
            audio_feats,
            text_mask,
            audio_mask,
            loss_mask,
            labels,
        )

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_mask_strategy(self) -> str:
        """Randomly select a masking strategy based on configured probabilities.
        
        Returns one of the strategy names or ``"none"``.
        """
        roll = random.random()
        cumulative = 0.0
        for name, prob in self._strategy_probs:
            cumulative += prob
            if roll < cumulative:
                return name
        return "none"
    
    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------
    
    def _mask_independent(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mask each pitch / note token independently.
        
        Uses ``pitch_mask_rate`` and ``note_mask_rate`` from mask_params
        (default 1.0 = mask all when strategy is selected).
        """
        pitch_rate = self.mask_params.get("pitch_mask_rate", 1.0)
        note_rate = self.mask_params.get("note_mask_rate", 1.0)
        
        if pitch_rate > 0 and len(self._pitch_ids_tensor) > 0:
            is_pitch = torch.isin(tokens, self._pitch_ids_tensor)
            if pitch_rate >= 1.0:
                tokens[is_pitch] = self.mask_token_id
            else:
                mask_draw = torch.rand(tokens.shape) < pitch_rate
                tokens[is_pitch & mask_draw] = self.mask_token_id
        
        if note_rate > 0 and len(self._note_ids_tensor) > 0:
            is_note = torch.isin(tokens, self._note_ids_tensor)
            if note_rate >= 1.0:
                tokens[is_note] = self.mask_token_id
            else:
                mask_draw = torch.rand(tokens.shape) < note_rate
                tokens[is_note & mask_draw] = self.mask_token_id
        
        return tokens
    
    def _mask_paired(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mask adjacent (pitch, note) pairs together.
        
        In the aggregated prompt layout, pitch and note tokens always
        appear as adjacent pairs: [..., <P_x>, <NOTE_y>, ...].
        
        Uses ``pair_mask_rate`` from mask_params (default 1.0 = mask
        all pairs when strategy is selected).
        """
        pair_rate = self.mask_params.get("pair_mask_rate", 1.0)
        if pair_rate <= 0:
            return tokens
        
        is_pitch = torch.isin(tokens, self._pitch_ids_tensor)
        is_note = torch.isin(tokens, self._note_ids_tensor)
        
        # Find positions where tokens[i] is pitch and tokens[i+1] is note
        n = tokens.shape[0]
        if n < 2:
            return tokens
        
        pair_starts = is_pitch[:-1] & is_note[1:]
        pair_indices = torch.where(pair_starts)[0]
        
        if len(pair_indices) == 0:
            return tokens
        
        if pair_rate >= 1.0:
            # Mask all pairs
            for idx in pair_indices:
                tokens[idx] = self.mask_token_id
                tokens[idx + 1] = self.mask_token_id
        else:
            mask_draw = torch.rand(len(pair_indices)) < pair_rate
            for idx, do_mask in zip(pair_indices, mask_draw):
                if do_mask:
                    tokens[idx] = self.mask_token_id
                    tokens[idx + 1] = self.mask_token_id
        
        return tokens
    
    def _apply_global_mask_collapse(
        self,
        packed_text_tokens: torch.Tensor,
        audio_feats: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Global mask with melisma collapse.
        
        Masks ALL pitch, note & BPM tokens AND collapses melisma so
        each word has exactly one <SVS_MASK> pair, exactly matching
        the weak-label token structure.
        
        Since removing tokens changes sequence length, this method
        operates on all tensors together (not just tokens).
        
        Note: Strategy selection (coin flip) is handled by
        ``_select_mask_strategy``; this method always applies.
        """
        tokens = packed_text_tokens.clone()
        n = len(tokens)
        
        # Classify each token
        is_pitch = torch.isin(tokens, self._pitch_ids_tensor)
        is_note = torch.isin(tokens, self._note_ids_tensor)
        is_bpm = torch.isin(tokens, self._bpm_ids_tensor)
        is_score = is_pitch | is_note  # pitch or note token
        
        # Mask all BPM tokens
        tokens[is_bpm] = self.mask_token_id
        
        # Walk through the sequence, find consecutive runs of score tokens
        # (pitch/note groups).  Within each group, keep only the first
        # pitch-note pair and mark the rest for removal.
        remove_mask = torch.zeros(n, dtype=torch.bool)
        i = 0
        while i < n:
            if not is_score[i]:
                i += 1
                continue
            
            # Found start of a score-token group
            group_start = i
            while i < n and is_score[i]:
                i += 1
            group_end = i  # exclusive
            
            # Count pitch tokens → each pitch starts one pair
            pair_count = 0
            first_pair_end = group_start  # will be updated
            for k in range(group_start, group_end):
                if is_pitch[k]:
                    pair_count += 1
                    if pair_count == 1:
                        # Replace first pair with <SVS_MASK>
                        tokens[k] = self.mask_token_id
                        # The note following this pitch
                        if k + 1 < group_end and is_note[k + 1]:
                            tokens[k + 1] = self.mask_token_id
                            first_pair_end = k + 2
                        else:
                            first_pair_end = k + 1
            
            # Mark everything after the first pair for removal
            if pair_count > 1:
                for k in range(first_pair_end, group_end):
                    remove_mask[k] = True
        
        # Remove extra melisma tokens from all tensors
        if remove_mask.any():
            keep = ~remove_mask
            tokens = tokens[keep]
            audio_feats = audio_feats[keep]
            text_mask = text_mask[keep]
            audio_mask = audio_mask[keep]
            loss_mask = loss_mask[keep]
            labels = labels[keep]
        
        return tokens, audio_feats, text_mask, audio_mask, loss_mask, labels
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate preprocessed SVS batch.
        
        Pads all sequences to the maximum length in the batch.
        """
        # Find max length
        max_len = max(sample["total_length"] for sample in batch)
        
        # Pad and stack tensors
        packed_text_list = []
        audio_feats_list = []
        text_mask_list = []
        audio_mask_list = []
        loss_mask_list = []
        labels_list = []

        for sample in batch:
            cur_len = sample["total_length"]
            pad_len = max_len - cur_len

            # Pad packed_text_tokens
            if pad_len > 0:
                packed_text = torch.nn.functional.pad(
                    sample["packed_text_tokens"], (0, pad_len), value=0
                )
            else:
                packed_text = sample["packed_text_tokens"][:max_len]
            packed_text_list.append(packed_text)

            # Pad audio_feats [T, P, D]
            audio_feats = sample["audio_feats"]
            if pad_len > 0:
                pad_feats = torch.zeros(
                    (pad_len, audio_feats.shape[1], audio_feats.shape[2]),
                    dtype=audio_feats.dtype
                )
                audio_feats = torch.cat([audio_feats, pad_feats], dim=0)
            else:
                audio_feats = audio_feats[:max_len]
            audio_feats_list.append(audio_feats)

            # Pad masks
            for mask_list, mask_tensor in [
                (text_mask_list, sample["text_mask"]),
                (audio_mask_list, sample["audio_mask"]),
                (loss_mask_list, sample["loss_mask"]),
                (labels_list, sample["labels"]),
            ]:
                if pad_len > 0:
                    padded = torch.nn.functional.pad(mask_tensor, (0, pad_len), value=0)
                else:
                    padded = mask_tensor[:max_len]
                mask_list.append(padded)

        # Stack into batches
        text_tokens_batch = torch.stack(packed_text_list, dim=0)
        audio_feats_batch = torch.stack(audio_feats_list, dim=0)
        text_mask_batch = torch.stack(text_mask_list, dim=0)
        audio_mask_batch = torch.stack(audio_mask_list, dim=0)
        loss_mask_batch = torch.stack(loss_mask_list, dim=0)
        labels_batch = torch.stack(labels_list, dim=0)

        # Return contiguous tensors to avoid pin_memory issues
        out: Dict[str, torch.Tensor] = {
            "text_tokens": text_tokens_batch.contiguous(),
            "audio_feats": audio_feats_batch.contiguous(),
            "text_mask": text_mask_batch.contiguous(),
            "audio_mask": audio_mask_batch.contiguous(),
            "loss_mask": loss_mask_batch.contiguous(),
            "labels": labels_batch.contiguous(),
        }
        return out


# DynamicBatchSampler lives in its own module; re-export here for
# backward compatibility with existing imports.
from .dynamic_batch import DynamicBatchSampler  # noqa: F401

# Raw folder-based pipeline and shared annotation helpers live in
# svs_raw_data; re-export here for backward compatibility.
from .svs_raw_data import (  # noqa: F401
    convert_annotation_to_syllables,
    convert_words_to_syllables,
    expand_syllables,
    get_note_duration_map,
    build_svs_text_tensor,
    tokenize_svs_batch,
    HFSVSDataset,
    build_svs_dataloader,
    load_svs_datasets,
    DEFAULT_AUDIO_COLUMN,
    DEFAULT_ID_COLUMN,
)


def build_preprocessed_svs_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
    max_batch_tokens: int = 0,
    mask_strategy: str = "none",
    mask_params: Optional[Dict] = None,
    pitch_token_ids: Optional[set] = None,
    note_token_ids: Optional[set] = None,
    bpm_token_ids: Optional[set] = None,
    mask_token_id: int = 0,
    # Prompt audio configuration
    prompt_audio_prob: float = 0.0,
    prompt_max_frames: int = 50,
    song_index: Optional[Dict[str, List[int]]] = None,
    prompt_audio_seed: Optional[int] = None,
    text_tokenizer=None,
) -> torch.utils.data.DataLoader:
    """
    Build DataLoader for preprocessed SVS dataset.
    
    Args:
        hf_dataset: HuggingFace dataset with preprocessed data
        accelerator: Training accelerator
        batch_size: Fixed batch size (used when max_batch_tokens=0)
        num_workers: Number of data loading workers
        drop_last: Whether to drop the last incomplete batch
        max_batch_tokens: Maximum tokens per batch. If > 0, uses dynamic batching
                          instead of fixed batch_size. Recommended: 8192-16384.
        mask_strategy: Masking strategy: ``\"none\"`` or list of strategies (e.g. ``[\"global\", \"paired\"]``)
        mask_params: Strategy-specific parameters dict
        pitch_token_ids: Set of pitch token IDs for masking
        note_token_ids: Set of note token IDs for masking
        bpm_token_ids: Set of BPM token IDs for masking (used by global strategy)
        mask_token_id: ID of the <SVS_MASK> token
        prompt_audio_prob: Probability of prepending prompt audio
        prompt_max_frames: Maximum number of audio frames for same-song prompt cropping
        song_index: Mapping of song_name -> list of sample indices for same-song prompt selection
        prompt_audio_seed: If set, use deterministic RNG per sample (for validation)

    Returns:
        DataLoader with either fixed or dynamic batching
    """
    torch_dataset = HFPreprocessedSVSDataset(
        hf_dataset,
        mask_strategy=mask_strategy,
        mask_params=mask_params,
        pitch_token_ids=pitch_token_ids,
        note_token_ids=note_token_ids,
        bpm_token_ids=bpm_token_ids,
        mask_token_id=mask_token_id,
        prompt_audio_prob=prompt_audio_prob,
        prompt_max_frames=prompt_max_frames,
        song_index=song_index,
        prompt_audio_seed=prompt_audio_seed,
        text_tokenizer=text_tokenizer,
    )
    
    if max_batch_tokens > 0:
        # Extract lengths directly from HuggingFace dataset column (instant!)
        # This avoids iterating through the dataset which would be slow
        lengths = hf_dataset["total_length"]  # Direct column access
        
        # When prompt audio is enabled, increase estimated lengths to
        # account for the same-song prompt prepended at the sequence front
        # at runtime.
        if prompt_audio_prob > 0 and prompt_max_frames > 0:
            avg_prompt_overhead = int((prompt_max_frames * 0.75 + 2) * prompt_audio_prob + 0.5)
            lengths = [l + avg_prompt_overhead for l in lengths]
        
        # Use dynamic batching with distributed support
        # For HYBRID_SHARD: all ranks inside the same shard group process the
        # same micro-batch (params sharded, same data).  Partition data across
        # replica groups only, using dp_rank / dp_world_size.
        batch_sampler = DynamicBatchSampler(
            lengths=lengths,
            max_batch_tokens=max_batch_tokens,
            max_batch_size=batch_size,  # Use batch_size as upper cap
            shuffle=True,
            drop_last=drop_last,
            rank=accelerator.dp_rank,
            world_size=accelerator.dp_world_size,
        )
        
        # Note: When using batch_sampler, we can't use accelerator.prepare_dataloader
        # directly because it conflicts with batch_sampler. Create manually.
        loader = torch.utils.data.DataLoader(
            torch_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=HFPreprocessedSVSDataset.collate_fn,
            pin_memory=True,
        )
        # Store batch_sampler reference for epoch management in training loop
        # Use object.__setattr__ to bypass PyTorch DataLoader's attribute restrictions
        object.__setattr__(loader, '_dynamic_sampler', batch_sampler)
        return loader
    else:
        # Use fixed batch size
        return accelerator.prepare_dataloader(
            torch_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=HFPreprocessedSVSDataset.collate_fn,
            drop_last=drop_last,
        )
