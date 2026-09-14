"""
SVS preprocessor: converts raw SVS data to training-ready format.

Contains :class:`SVSPreprocessor` (AudioVAE encoding + SVS token sequence
construction) and :func:`create_lightweight_preprocessor` (token-maps-only
variant for inference scripts).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class SVSPreprocessor:
    """
    Preprocessor that converts raw SVS data to training-ready format.

    This mirrors the logic in BatchProcessor + AudioFeatureProcessingPacker,
    but processes all data at once and saves to disk.

    Optimized to only load AudioVAE and tokenizer (not the full 800M+ model).
    """

    def __init__(
        self,
        pretrained_path: str,
        sample_rate: int = 44100,
        device: str = "cuda",
    ):
        self.sample_rate = sample_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load config
        config_path = Path(pretrained_path) / "config.json"
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            import json as json_lib
            config_dict = json_lib.load(f)

        self.patch_size = config_dict.get("patch_size", 4)
        self.feat_dim = config_dict.get("feat_dim", 64)

        # Load AudioVAE only (not the full model!)
        # Auto-detect architecture: V1 uses AudioVAE, V2 uses AudioVAEV2
        architecture = config_dict.get("architecture", "voxcpm").lower()
        print(f"Loading AudioVAE from {pretrained_path} (architecture: {architecture})...")

        if architecture == "voxcpm2":
            from vocalrender.modules.audiovae.audio_vae_v2 import AudioVAE as AudioVAEV2, AudioVAEConfig as AudioVAEConfigV2
            audio_vae_config_dict = config_dict.get("audio_vae_config", None)
            if audio_vae_config_dict:
                audio_vae_config = AudioVAEConfigV2(**audio_vae_config_dict)
                self.audio_vae = AudioVAEV2(config=audio_vae_config)
            else:
                self.audio_vae = AudioVAEV2()
        else:
            from vocalrender.modules.audiovae.audio_vae import AudioVAE, AudioVAEConfig
            audio_vae_config_dict = config_dict.get("audio_vae_config", None)
            if audio_vae_config_dict:
                audio_vae_config = AudioVAEConfig(**audio_vae_config_dict)
                self.audio_vae = AudioVAE(config=audio_vae_config)
            else:
                self.audio_vae = AudioVAE()

        # Load VAE weights - support both safetensors and pytorch formats
        vae_safetensors_path = Path(pretrained_path) / "audiovae.safetensors"
        vae_pth_path = Path(pretrained_path) / "audiovae.pth"

        if vae_safetensors_path.exists():
            try:
                from safetensors.torch import load_file
                vae_state_dict = load_file(str(vae_safetensors_path), device="cpu")
                print(f"  Loaded AudioVAE from safetensors: {vae_safetensors_path}")
            except ImportError:
                vae_state_dict = torch.load(vae_pth_path, map_location="cpu", weights_only=True)["state_dict"]
                print(f"  safetensors not available, loaded from: {vae_pth_path}")
        elif vae_pth_path.exists():
            checkpoint = torch.load(vae_pth_path, map_location="cpu", weights_only=True)
            vae_state_dict = checkpoint.get("state_dict", checkpoint)
            print(f"  Loaded AudioVAE from: {vae_pth_path}")
        else:
            raise FileNotFoundError(f"AudioVAE checkpoint not found at {pretrained_path}")

        self.audio_vae.load_state_dict(vae_state_dict)
        self.audio_vae.to(self.device).to(torch.float32)
        self.audio_vae.eval()

        self.patch_len = self.audio_vae.hop_length * self.patch_size

        # Load tokenizer only
        print(f"Loading tokenizer from {pretrained_path}...")
        from transformers import LlamaTokenizerFast
        self.tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_path)

        # Special token IDs
        self.audio_start_id = 101
        self.audio_end_id = 102
        self.audio_prompt_start_id = 103
        self.audio_prompt_end_id = 104

        # Add SVS tokens to tokenizer
        self._setup_svs_tokens()

        print(f"Preprocessor initialized on {self.device}")
        print(f"  Patch size: {self.patch_size}, Feat dim: {self.feat_dim}")

    def _setup_svs_tokens(self):
        """Add SVS tokens to tokenizer and build lookup maps."""
        from vocalrender.model.svs_utils import get_svs_token_maps

        pitch_tokens, note_tokens, bpm_tokens, dur_units, special_tokens = get_svs_token_maps()
        new_tokens = pitch_tokens + note_tokens + bpm_tokens + special_tokens
        num_added = self.tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added} SVS tokens to tokenizer")

        # Build lookup maps
        self.pitch_to_id = {}
        for pt in pitch_tokens:
            pid = self.tokenizer.convert_tokens_to_ids(pt)
            if pid != self.tokenizer.unk_token_id:
                try:
                    val = int(pt.split('_')[1][:-1])
                    self.pitch_to_id[val] = pid
                except Exception:
                    pass

        # Build note_to_id mapping: note token string -> token id
        # Also build note_str_to_idx for encoding in build_text_tensor
        self.note_to_id = {}  # note_token_str -> tokenizer id
        self.note_idx_to_id = {}  # note_idx -> tokenizer id
        self.note_str_to_idx = {}  # note_token_str -> note_idx
        for idx, nt in enumerate(note_tokens):
            nid = self.tokenizer.convert_tokens_to_ids(nt)
            if nid != self.tokenizer.unk_token_id:
                self.note_to_id[nt] = nid
                self.note_idx_to_id[idx] = nid
                self.note_str_to_idx[nt] = idx

        self.bpm_to_id = {}
        for bt in bpm_tokens:
            bid = self.tokenizer.convert_tokens_to_ids(bt)
            if bid != self.tokenizer.unk_token_id:
                try:
                    val = int(bt.split('_')[1][:-1])
                    self.bpm_to_id[val] = bid
                except Exception:
                    pass

        self.svs_mask_token_id = None
        if "<SVS_MASK>" in special_tokens:
            mask_id = self.tokenizer.convert_tokens_to_ids("<SVS_MASK>")
            if mask_id != self.tokenizer.unk_token_id:
                self.svs_mask_token_id = mask_id

    def encode_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to VAE latent features."""
        wav = wav.to(self.device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        elif wav.dim() == 2:
            wav = wav.unsqueeze(1)  # [B, 1, T]

        wav_len = wav.size(-1)
        if wav_len % self.patch_len != 0:
            padding_size = self.patch_len - wav_len % self.patch_len
            wav = torch.nn.functional.pad(wav, (0, padding_size))

        with torch.no_grad():
            z = self.audio_vae.encode(wav, self.audio_vae.in_sample_rate)  # [B, D, T']
            feat = z.transpose(1, 2)  # [B, T', D]

        return feat.cpu()

    def extract_audio_feats(self, audio_waveform: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Extract and reshape audio features for training."""
        audio_feats = self.encode_audio(audio_waveform)  # [1, T', D]
        return self._reshape_audio_feats(audio_feats)

    def _reshape_audio_feats(self, audio_feats: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Reshape VAE latent features into patch format for training.

        Args:
            audio_feats: [1, T', D] or [T', D] raw VAE latent features
        Returns:
            (audio_feats, audio_duration): [T, P, D] reshaped features and duration in seconds
        """
        if audio_feats.dim() == 2:
            audio_feats = audio_feats.unsqueeze(0)  # [1, T', D]

        if audio_feats.size(1) % self.patch_size != 0:
            audio_feats_ = audio_feats.transpose(1, 2)
            padding = nn.functional.pad(
                audio_feats_,
                (0, self.patch_size - audio_feats.size(1) % self.patch_size)
            )
            audio_feats = padding.transpose(1, 2)

        audio_duration = audio_feats.size(1) / 25.0
        audio_feats = rearrange(audio_feats, "b (t p) c -> b t p c", p=self.patch_size)

        return audio_feats.squeeze(0), audio_duration  # [T, P, D], float

    def encode_audio_batch(self, wavs: list) -> list:
        """Batch encode variable-length audio waveforms through AudioVAE.

        Args:
            wavs: List of 1D tensors [T_i] with different lengths
        Returns:
            List of (audio_feats, audio_duration) tuples, same format as extract_audio_feats
        """
        if not wavs:
            return []

        # 1. Compute per-sample padded lengths (aligned to patch_len)
        padded_lens = []
        for wav in wavs:
            wav_len = wav.size(-1)
            if wav_len % self.patch_len != 0:
                padded_len = wav_len + (self.patch_len - wav_len % self.patch_len)
            else:
                padded_len = wav_len
            padded_lens.append(padded_len)

        max_len = max(padded_lens)

        # 2. Pad all waveforms to max_len and stack into batch [B, 1, max_len]
        batch = torch.zeros(len(wavs), 1, max_len)
        for i, wav in enumerate(wavs):
            batch[i, 0, :wav.size(-1)] = wav

        batch = batch.to(self.device)

        # 3. VAE encode entire batch
        with torch.no_grad():
            z = self.audio_vae.encode(batch, self.audio_vae.in_sample_rate)  # [B, D, T_max']
            feats_all = z.transpose(1, 2).cpu()  # [B, T_max', D]

        # 4. Extract per-sample features and reshape
        results = []
        for i, padded_len in enumerate(padded_lens):
            feat_len = padded_len // self.audio_vae.hop_length
            sample_feats = feats_all[i, :feat_len, :]  # [T'_i, D]
            results.append(self._reshape_audio_feats(sample_feats))

        return results

    def build_svs_sequence(
        self,
        text_tensor: torch.Tensor,
        is_prompt: bool = False,
        has_score: bool = True,
    ) -> torch.Tensor:
        """
        Build SVS token sequence from text_tensor.

        Args:
            text_tensor: [L, N] where columns are [text_ids..., pitch, note, bpm]
            is_prompt: Whether this is a prompt sample
            has_score: Whether this sample has full score annotations

        Returns:
            Token sequence tensor [S] containing all tokens + audio_start
        """
        L = text_tensor.shape[0]
        num_cols = text_tensor.shape[1]

        # Layout: [text_ids..., pitch, note, bpm, word_idx]
        num_text_cols = num_cols - 4
        if num_text_cols < 1:
            num_text_cols = 1

        pitch_col = num_text_cols
        note_col = num_text_cols + 1
        bpm_col = num_text_cols + 2
        word_idx_col = num_text_cols + 3

        full_seq_ids = []

        # BPM token (global, from first row)
        bpm_val = 120
        if bpm_col < num_cols:
            bpm_val = int(text_tensor[0, bpm_col].item())

        if not has_score and self.svs_mask_token_id is not None:
            # Weak label: use <SVS_MASK> for BPM
            full_seq_ids.append(self.svs_mask_token_id)
        elif bpm_val in self.bpm_to_id:
            full_seq_ids.append(self.bpm_to_id[bpm_val])
        elif 120 in self.bpm_to_id:
            full_seq_ids.append(self.bpm_to_id[120])

        prev_word_idx = None

        for i in range(L):
            text_ids = []
            for tc in range(num_text_cols):
                tid = int(text_tensor[i, tc].item())
                if tid != 0:
                    text_ids.append(tid)

            pitch_val = int(text_tensor[i, pitch_col].item())
            note_idx = int(text_tensor[i, note_col].item())
            cur_word_idx = int(text_tensor[i, word_idx_col].item())

            # Melisma = same original word (word_idx) across consecutive rows
            is_melisma = prev_word_idx is not None and cur_word_idx == prev_word_idx

            pitch_id = self.pitch_to_id.get(pitch_val, None)
            note_id = self.note_idx_to_id.get(note_idx, None)

            # Weak label: replace pitch/note with <SVS_MASK>
            if not has_score and self.svs_mask_token_id is not None:
                pitch_id = self.svs_mask_token_id
                note_id = self.svs_mask_token_id

            # Aggregated layout: word_text_tokens + (pitch, note); melisma
            # rows reuse the previous word's text tokens.
            if not is_melisma:
                full_seq_ids.extend(text_ids)
            if pitch_id is not None:
                full_seq_ids.append(pitch_id)
            if note_id is not None:
                full_seq_ids.append(note_id)

            prev_word_idx = cur_word_idx

        # Add audio start token
        audio_start = self.audio_prompt_start_id if is_prompt else self.audio_start_id
        full_seq_ids.append(audio_start)

        return torch.tensor(full_seq_ids, dtype=torch.int32)

    def process_sample(
        self,
        text_tensor: torch.Tensor,
        audio_waveform: torch.Tensor = None,
        is_prompt: bool = False,
        precomputed_audio: Tuple[torch.Tensor, float] = None,
        has_score: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single sample into training-ready format.

        Args:
            text_tensor: [L, N] text tensor
            audio_waveform: Raw audio waveform (used if precomputed_audio is None)
            is_prompt: Whether this is a prompt sample
            precomputed_audio: Optional (audio_feats, audio_duration) tuple from
                               encode_audio_batch, skips VAE encoding if provided
            has_score: Whether this sample has full score annotations

        Returns dict with:
            - packed_text_tokens: [T_total] int32
            - audio_feats: [T_audio, P, D] float32
            - text_mask: [T_total] int32
            - audio_mask: [T_total] int32
            - loss_mask: [T_total] int32
            - labels: [T_total] int32
            - audio_duration: float
            - text_token_count: int
        """
        # 1. Build SVS token sequence
        svs_seq = self.build_svs_sequence(text_tensor, is_prompt=is_prompt, has_score=has_score)
        text_length = svs_seq.shape[0]

        # 2. Extract audio features (use precomputed if available)
        if precomputed_audio is not None:
            audio_feats, audio_duration = precomputed_audio
        else:
            audio_feats, audio_duration = self.extract_audio_feats(audio_waveform)
        audio_length = audio_feats.shape[0]

        # 3. Build packed text tokens
        text_pad = torch.zeros(audio_length, dtype=torch.int32)
        audio_end = self.audio_prompt_end_id if is_prompt else self.audio_end_id
        packed_text = torch.cat([
            svs_seq,
            text_pad,
            torch.tensor([audio_end], dtype=torch.int32),
        ])

        # 4. Pad audio features
        audio_pad_before = torch.zeros(
            (text_length, self.patch_size, audio_feats.size(-1)),
            dtype=torch.float32,
        )
        audio_pad_after = torch.zeros(
            (1, self.patch_size, audio_feats.size(-1)),
            dtype=torch.float32,
        )
        padded_audio_feats = torch.cat([audio_pad_before, audio_feats, audio_pad_after], dim=0)

        # 5. Build masks
        text_mask = torch.cat([
            torch.ones(text_length, dtype=torch.int32),
            torch.zeros(audio_length, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        ])

        audio_mask = torch.cat([
            torch.zeros(text_length, dtype=torch.int32),
            torch.ones(audio_length, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        ])

        loss_mask = torch.cat([
            torch.zeros(text_length, dtype=torch.int32),
            torch.zeros(audio_length, dtype=torch.int32) if is_prompt else torch.ones(audio_length, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        ])

        # 6. Build labels
        labels = torch.zeros(text_length + audio_length + 1, dtype=torch.int32)
        labels[-2] = 1  # Stop token position

        return {
            "packed_text_tokens": packed_text,
            "audio_feats": padded_audio_feats,
            "text_mask": text_mask,
            "audio_mask": audio_mask,
            "loss_mask": loss_mask,
            "labels": labels,
            "audio_duration": audio_duration,
            "text_token_count": text_tensor.shape[0],
            "total_length": packed_text.shape[0],
        }


def create_lightweight_preprocessor(tokenizer):
    """Create an SVSPreprocessor with only token maps (no VAE/model).

    Useful for inference scripts that need to rebuild SVS prompts
    from metadata without loading the full preprocessing pipeline.
    """
    p = SVSPreprocessor.__new__(SVSPreprocessor)
    p.tokenizer = tokenizer
    p.audio_start_id = 101
    p.audio_end_id = 102
    p.audio_prompt_start_id = 103
    p.audio_prompt_end_id = 104
    p._setup_svs_tokens()
    return p
