"""
VoxCPM: A Tokenizer-free speech generation model

This module contains the main VoxCPM model implementation, including configuration classes
and the core VoxCPMModel for text-to-speech generation.

Copyright 2026 OpenBMB
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
from typing import Dict, Tuple, Union, Generator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import librosa
import numpy as np
from einops import rearrange
from pydantic import BaseModel

try:
    from safetensors.torch import load_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
from tqdm import tqdm
from transformers import LlamaTokenizerFast

from ..modules.audiovae import AudioVAEV2, AudioVAEConfigV2
from ..modules.layers import ScalarQuantizationLayer
from ..modules.layers.lora import apply_lora_to_named_linear_modules
from ..modules.locdit import CfmConfig, UnifiedCFM, VoxCPMLocDiTV2
from ..modules.locenc import VoxCPMLocEnc
from ..modules.minicpm4 import MiniCPM4Config, MiniCPMModel
from .utils import build_prefill_and_persistent_kpm, get_dtype, mask_multichar_chinese_tokens, next_and_close


def _trim_audio_silence_vad(
    audio: torch.Tensor,
    sample_rate: int,
    max_silence_ms: float = 200.0,
    top_db: float = 35.0,
) -> torch.Tensor:
    """使用能量阈值（VAD 方式）截取首尾静音及尾部长段伪静音，首尾各最多保留 max_silence_ms 毫秒静音。

    会同时截掉末尾的长段伪静音（低能量但非完全静音的段落，如长时间底噪）。

    Args:
        audio: (1, T) 的音频 tensor
        sample_rate: 采样率
        max_silence_ms: 首尾允许保留的最大静音长度（毫秒）
        top_db: 低于参考电平多少 dB 视为静音

    Returns:
        截取后的 (1, T') tensor
    """
    if audio.numel() == 0:
        return audio
    y = audio.squeeze(0).numpy()
    n = len(y)
    frame_length = 2048
    hop_length = 512
    ref = np.max(np.abs(y))
    if ref <= 0:
        return audio
    threshold = ref * (10.0 ** (-top_db / 20.0))

    try:
        _, (start, end) = librosa.effects.trim(
            y, top_db=top_db, ref=np.max, frame_length=frame_length, hop_length=hop_length
        )
    except Exception:
        start, end = 0, n

    # 用逐帧 RMS 找「最后一段有持续能量的位置」，截掉末尾长伪静音（低能量底噪等）
    n_frames = max(0, (n - frame_length) // hop_length + 1)
    last_voice_frame = -1
    for j in range(n_frames):
        idx = j * hop_length
        if idx + frame_length > n:
            break
        rms = np.sqrt(np.mean(y[idx : idx + frame_length] ** 2))
        if rms >= threshold:
            last_voice_frame = j
    if last_voice_frame >= 0:
        end_by_vad = min(n, (last_voice_frame + 1) * hop_length + (frame_length - hop_length))
        end = min(end, end_by_vad)

    max_silence_samples = int(max_silence_ms * sample_rate / 1000.0)
    new_start = max(0, start - max_silence_samples)
    new_end = min(n, end + max_silence_samples)
    return audio[:, new_start:new_end]


class VoxCPMEncoderConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None


class VoxCPMDitConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None
    dit_mean_mode: bool = False

    cfm_config: CfmConfig


class VoxCPMConfig(BaseModel):
    lm_config: MiniCPM4Config
    patch_size: int = 4
    feat_dim: int = 64
    residual_lm_num_layers: int = 8
    residual_lm_no_rope: bool = False
    scalar_quantization_latent_dim: int = 512
    scalar_quantization_scale: int = 9

    encoder_config: VoxCPMEncoderConfig
    dit_config: VoxCPMDitConfig
    audio_vae_config: Optional[AudioVAEConfigV2] = None

    max_length: int = 8192
    device: str = "cuda"
    dtype: str = "bfloat16"


class LoRAConfig(BaseModel):
    enable_lm: bool = False  # Apply LoRA to base_lm + residual_lm
    enable_dit: bool = False  # Apply LoRA to VoxCPMLocDiT
    enable_proj: bool = False  # Apply LoRA to projection Linear layers

    r: int = 8
    alpha: int = 16
    dropout: float = 0.0

    # Target linear layer names for LM & DiT (matched by attribute name)
    target_modules_lm: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    target_modules_dit: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    # Projection layer attribute names to find on VoxCPM2Model
    target_proj_modules: list[str] = ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj", "fusion_concat_proj"]


VoxCPMConfig.model_rebuild()


class VoxCPM2Model(nn.Module):
    def __init__(
        self,
        config: VoxCPMConfig,
        tokenizer: LlamaTokenizerFast,
        audio_vae: AudioVAEV2,
        lora_config: LoRAConfig = None,
    ):
        super().__init__()
        self.config = config
        self.lora_config = lora_config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        self.device = config.device
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        print(f"Running on device: {self.device}, dtype: {self.config.dtype}", file=sys.stderr)

        # Text-Semantic LM
        self.base_lm = MiniCPMModel(config.lm_config)
        self.base_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        self.text_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        self.audio_start_token = 101
        self.audio_end_token = 102
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104

        # Residual Acoustic LM
        residual_lm_config = config.lm_config.model_copy(deep=True)
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        residual_lm_config.vocab_size = 0
        residual_lm_config.no_rope = config.residual_lm_no_rope
        self.residual_lm = MiniCPMModel(residual_lm_config)
        self.residual_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        # Local Encoder
        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=config.feat_dim)

        # Local DiT
        decoder_config = config.lm_config.model_copy(deep=True)
        decoder_config.hidden_size = config.dit_config.hidden_dim
        decoder_config.intermediate_size = config.dit_config.ffn_dim
        decoder_config.num_attention_heads = config.dit_config.num_heads
        decoder_config.num_hidden_layers = config.dit_config.num_layers
        decoder_config.kv_channels = config.dit_config.kv_channels
        decoder_config.vocab_size = 0

        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            cfm_params=config.dit_config.cfm_config,
            estimator=VoxCPMLocDiTV2(decoder_config, in_channels=config.feat_dim),
            mean_mode=config.dit_config.dit_mean_mode,
        )

        # Projection layers
        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            config.scalar_quantization_latent_dim,
            config.scalar_quantization_scale,
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.fusion_concat_proj = nn.Linear(config.lm_config.hidden_size * 2, config.lm_config.hidden_size)

        # Stop Predictor
        self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)
        self.stop_loss = nn.CrossEntropyLoss(reduction="none")

        # Audio VAE
        self.audio_vae = audio_vae
        self.chunk_size = audio_vae.chunk_size
        self.in_sample_rate = audio_vae.in_sample_rate
        self.out_sample_rate = audio_vae.out_sample_rate
        self._encode_sample_rate = self.in_sample_rate
        self.sample_rate = self.out_sample_rate

        if self.lora_config is not None:
            self._apply_lora()

    def _apply_lora(self):
        """注入 LoRA 到 LM / DiT / 投影层"""
        cfg = self.lora_config
        lora_kwargs = dict(r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)

        # LM: base_lm + residual_lm
        if cfg.enable_lm:
            for lm in [self.base_lm, self.residual_lm]:
                apply_lora_to_named_linear_modules(lm, target_submodule_names=cfg.target_modules_lm, **lora_kwargs)

        # DiT: feat_decoder.estimator
        if cfg.enable_dit:
            apply_lora_to_named_linear_modules(
                self.feat_decoder.estimator, target_submodule_names=cfg.target_modules_dit, **lora_kwargs
            )

        # 投影层
        if cfg.enable_proj:
            from ..modules.layers.lora import LoRALinear

            for attr_name in cfg.target_proj_modules:
                module = getattr(self, attr_name, None)
                if isinstance(module, nn.Linear):
                    setattr(self, attr_name, LoRALinear(base=module, **lora_kwargs))

    def set_activation_checkpointing(self, enabled: bool = True):
        modules = [
            self.base_lm,
            self.residual_lm,
            self.feat_encoder.encoder,
            self.feat_decoder.estimator.decoder,
        ]
        for module in modules:
            if hasattr(module, "set_gradient_checkpointing"):
                module.set_gradient_checkpointing(enabled)
        return self

    def forward(
        self,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        audio_feats: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        *,
        progress: float = 0.0,
        sample_generate: bool = False,
    ):
        text_tokens = text_tokens.to(self.device, dtype=torch.long)
        text_mask = text_mask.to(self.device, dtype=self._dtype())
        audio_feats = audio_feats.to(self.device, dtype=self._dtype())
        audio_mask = audio_mask.to(self.device, dtype=self._dtype())
        loss_mask = loss_mask.to(self.device, dtype=self._dtype())
        labels = labels.to(self.device, dtype=torch.long)

        B, T, P, D = audio_feats.shape
        feat_embed = self.feat_encoder(audio_feats)
        feat_embed = self.enc_to_lm_proj(feat_embed)

        scale_emb = getattr(self.config.lm_config, "scale_emb", 1.0)
        if not getattr(self.config.lm_config, "use_mup", False):
            scale_emb = 1.0
        text_embed = self.base_lm.embed_tokens(text_tokens) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed

        enc_outputs, _ = self.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )
        enc_outputs = enc_outputs.to(self._dtype())
        enc_outputs = self.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = torch.cat((torch.zeros_like(enc_outputs[:, 0:1, :]), enc_outputs[:, :-1, :]), dim=1)

        residual_inputs = self.fusion_concat_proj(
            torch.cat((enc_outputs, audio_mask.unsqueeze(-1) * feat_embed), dim=-1)
        )
        residual_outputs, _ = self.residual_lm(
            inputs_embeds=residual_inputs,
            is_causal=True,
        )
        residual_outputs = residual_outputs.to(self._dtype())
        residual_hidden = torch.cat(
            (torch.zeros_like(residual_outputs[:, 0:1, :]), residual_outputs[:, :-1, :]),
            dim=1,
        )

        dit_hidden = torch.cat((self.lm_to_dit_proj(lm_hidden), self.res_to_dit_proj(residual_hidden)), dim=-1)
        dit_hidden = rearrange(dit_hidden, "b t c -> (b t) c")

        # Keep diffusion inputs in the same dtype as the model (e.g., bfloat16)
        target_dtype = self._dtype()

        feat_gt = rearrange(audio_feats.to(target_dtype), "b t p d -> (b t) p d")
        feat_cond = torch.cat(
            (torch.zeros_like(audio_feats[:, 0:1, ...]), audio_feats[:, :-1, ...]),
            dim=1,
        )
        feat_cond = rearrange(feat_cond.to(target_dtype), "b t p d -> (b t) p d")

        loss_seq_mask = loss_mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        loss_seq_mask = rearrange(loss_seq_mask, "b t p -> (b t) p 1").to(target_dtype)

        feat_gt_t = feat_gt.transpose(1, 2).contiguous()
        feat_cond_t = feat_cond.transpose(1, 2).contiguous()

        def _safe_compute_loss_full(head, mask_seq):
            # Single-head path: forward over the full (B*T) flattened batch
            # and let ``compute_loss`` mask the reduction. ``UnifiedCFM`` has
            # no zero guard on ``mask.sum()`` so we emit a graph-connected
            # zero when the mask is all zero (e.g., validation batches that
            # carry no relevant frames).
            if mask_seq.sum().item() <= 0:
                return dit_hidden.sum() * 0.0
            return head.compute_loss(
                feat_gt_t,
                dit_hidden,
                cond=feat_cond_t,
                tgt_mask=mask_seq.transpose(1, 2).contiguous(),
                progress=progress,
            )

        diff_loss = _safe_compute_loss_full(self.feat_decoder, loss_seq_mask)

        stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
        stop_losses = self.stop_loss(stop_logits.transpose(1, 2), labels)
        denom = torch.clamp(loss_mask.sum(), min=1.0)
        stop_loss = (stop_losses * loss_mask).sum() / denom

        feat_pred = None
        if sample_generate:
            feat_cond_for_sample = feat_cond.transpose(1, 2).contiguous()
            feat_pred_seq = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=feat_cond_for_sample,
                n_timesteps=(
                    self.config.dit_config.cfm_config.inference_cfg_rate
                    if hasattr(self.config.dit_config.cfm_config, "inference_cfg_rate")
                    else 10
                ),
            )
            feat_pred = rearrange(feat_pred_seq.transpose(1, 2), "(b t) d p -> b d (t p)", b=B, p=self.patch_size)

        feat_gt_tensor = rearrange(feat_gt, "(b t) p d -> b d (t p)", b=B, p=self.patch_size)

        result = {
            "loss/stop": stop_loss,
            "feat_gt": feat_gt_tensor,
            "feat_pred": feat_pred,
        }
        if diff_loss is not None:
            result["loss/diff"] = diff_loss
        return result

    def _dtype(self):
        return get_dtype(self.config.dtype)

    def _decoded_samples_per_patch(self) -> int:
        """Return decoded waveform samples produced by one LM patch.

        VoxCPM2 encodes audio at ``_encode_sample_rate`` (16 kHz) but decodes at
        ``sample_rate`` / ``out_sample_rate`` (48 kHz). Output-side trimming must
        therefore scale the latent patch length by the decode/encode SR ratio.
        """
        return int(round(self.patch_size * self.chunk_size * (self.sample_rate / self._encode_sample_rate)))

    def _encode_wav(self, wav_path: str, padding_mode: str = "right") -> torch.Tensor:
        """Load, trim, pad and VAE-encode an audio file.

        Args:
            wav_path: path to the audio file.
            padding_mode: "right" (default) or "left" padding for alignment.

        Returns:
            audio_feat: (T, P, D) tensor of latent patches.
        """
        audio, _ = librosa.load(wav_path, sr=self._encode_sample_rate, mono=True)
        audio = torch.from_numpy(audio).unsqueeze(0)
        audio = _trim_audio_silence_vad(audio, self._encode_sample_rate, max_silence_ms=200.0)
        patch_len = self.patch_size * self.chunk_size
        if audio.size(1) % patch_len != 0:
            padding_size = patch_len - audio.size(1) % patch_len
            pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
            audio = torch.nn.functional.pad(audio, pad)
        feat = self.audio_vae.encode(audio.to(self.device), self._encode_sample_rate).cpu()
        return feat.view(self.audio_vae.latent_dim, -1, self.patch_size).permute(1, 2, 0)

    def _make_ref_prefix(self, ref_feat: torch.Tensor, device: torch.device):
        """Build the [ref_start ref_audio ref_end] prefix segments.

        Returns:
            tokens, feats, text_mask, audio_mask  (all 1-D / 2-D tensors)
        """
        ref_len = ref_feat.size(0)
        z1 = torch.zeros((1, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32, device=device)
        tokens = torch.cat(
            [
                torch.tensor([self.ref_audio_start_token], dtype=torch.int32, device=device),
                torch.zeros(ref_len, dtype=torch.int32, device=device),
                torch.tensor([self.ref_audio_end_token], dtype=torch.int32, device=device),
            ]
        )
        feats = torch.cat([z1, ref_feat, z1], dim=0)
        t_mask = torch.cat(
            [
                torch.tensor([1], dtype=torch.int32),
                torch.zeros(ref_len, dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
            ]
        ).to(device)
        a_mask = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                torch.ones(ref_len, dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            ]
        ).to(device)
        return tokens, feats, t_mask, a_mask

    def generate(self, *args, **kwargs) -> torch.Tensor:
        return next_and_close(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[torch.Tensor, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        reference_wav_path: str = "",
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
    ) -> Generator[torch.Tensor, None, None]:
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False

        if reference_wav_path and prompt_wav_path:
            # Combined mode: reference isolation prefix + continuation suffix
            text = prompt_text + target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            ref_feat = self._encode_wav(reference_wav_path, padding_mode="right")
            prompt_feat = self._encode_wav(prompt_wav_path, padding_mode="left")
            prompt_audio_length = prompt_feat.size(0)

            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_feat, text_token.device)

            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )

            text_token = torch.cat([ref_tokens, text_token, prompt_pad_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat, prompt_feat], dim=0)
            text_mask = torch.cat(
                [
                    ref_t_mask,
                    torch.ones(text_length, dtype=torch.int32).to(text_token.device),
                    torch.zeros(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask,
                    torch.zeros(text_length, dtype=torch.int32).to(text_token.device),
                    torch.ones(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )

        elif reference_wav_path:
            # Reference-only mode (prompt isolation)
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            ref_feat = self._encode_wav(reference_wav_path, padding_mode="right")
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_feat, text_token.device)

            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([ref_tokens, text_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat], dim=0)
            text_mask = torch.cat(
                [
                    ref_t_mask,
                    torch.ones(text_length, dtype=torch.int32).to(text_token.device),
                ]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask,
                    torch.zeros(text_length, dtype=torch.int32).to(text_token.device),
                ]
            )

        elif len(prompt_wav_path) == 0:
            # Zero-shot mode
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            audio_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_mask = torch.ones(text_length, dtype=torch.int32).to(text_token.device)
            audio_mask = torch.zeros(text_length, dtype=torch.int32).to(text_token.device)

        else:
            # Continuation-only mode
            text = prompt_text + target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            prompt_feat = self._encode_wav(prompt_wav_path, padding_mode="left")
            prompt_audio_length = prompt_feat.size(0)
            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([text_token, prompt_pad_token])
            audio_feat = torch.cat([text_pad_feat, prompt_feat], dim=0)
            text_mask = torch.cat(
                [
                    torch.ones(text_length, dtype=torch.int32),
                    torch.zeros(prompt_audio_length, dtype=torch.int32),
                ]
            ).to(text_token.device)
            audio_mask = torch.cat(
                [
                    torch.zeros(text_length, dtype=torch.int32),
                    torch.ones(prompt_audio_length, dtype=torch.int32),
                ]
            ).to(text_token.device)

        text_token = text_token.unsqueeze(0).to(self.device)
        text_mask = text_mask.unsqueeze(0).to(self.device)
        audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
        audio_mask = audio_mask.unsqueeze(0).to(self.device)

        target_text_length = len(self.text_tokenizer(target_text))

        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len),
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )
            if streaming:
                patch_len = self._decoded_samples_per_patch()
                for latent_pred, _ in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                    yield decode_audio
                break
            else:
                latent_pred, pred_audio_feat = next_and_close(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(
                            f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...",
                            file=sys.stderr,
                        )
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break

        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
            patch_len = self._decoded_samples_per_patch()
            has_continuation = bool(prompt_wav_path)
            if has_continuation:
                decode_audio = decode_audio[..., patch_len * (streaming_prefix_len - 1):].squeeze(1).cpu()
            else:
                decode_audio = decode_audio.squeeze(1).cpu()
            yield decode_audio

    @torch.inference_mode()
    def build_prompt_cache(
        self,
        prompt_text: str = None,
        prompt_wav_path: str = None,
        reference_wav_path: str = None,
    ):
        """
        Build prompt cache for subsequent generation.

        Supports the same parameter combinations as ``generate()``:
        - ``reference_wav_path`` only -> reference mode (voice cloning, isolated)
        - ``prompt_text`` + ``prompt_wav_path`` -> continuation mode
        - all three -> combined ref + continuation mode

        Args:
            prompt_text: prompt text for continuation mode.
                Must be paired with ``prompt_wav_path``.
            prompt_wav_path: prompt audio path for continuation mode.
                Must be paired with ``prompt_text``.
            reference_wav_path: reference audio path for voice cloning
                (structurally isolated via ref_audio tokens).

        Returns:
            prompt_cache: dict used by ``_generate_with_prompt_cache``.
        """
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        if prompt_wav_path is None and reference_wav_path is None:
            raise ValueError("At least one of prompt_wav_path or reference_wav_path must be provided")

        cache = {}

        if reference_wav_path:
            cache["ref_audio_feat"] = self._encode_wav(reference_wav_path, padding_mode="right")

        if prompt_wav_path and prompt_text is not None:
            cache["prompt_text"] = prompt_text
            cache["audio_feat"] = self._encode_wav(prompt_wav_path, padding_mode="left")

        has_ref = "ref_audio_feat" in cache
        has_prompt = "audio_feat" in cache
        if has_ref and has_prompt:
            cache["mode"] = "ref_continuation"
        elif has_ref:
            cache["mode"] = "reference"
        else:
            cache["mode"] = "continuation"

        return cache

    def merge_prompt_cache(
        self,
        original_cache: dict,
        new_text: str,
        new_audio_feat: torch.Tensor,
    ):
        """
        Merge original prompt cache with newly generated content to stabilize voice.

        Args:
            original_cache: original prompt cache (any mode)
            new_text: newly generated text
            new_audio_feat: newly generated audio features

        Returns:
            merged_cache: merged cache with prompt_text and audio_feat
        """
        if original_cache is None:
            return {
                "prompt_text": new_text,
                "audio_feat": new_audio_feat,
                "mode": "continuation",
            }
        merged = {}
        if "ref_audio_feat" in original_cache:
            merged["ref_audio_feat"] = original_cache["ref_audio_feat"]
        merged["prompt_text"] = original_cache.get("prompt_text", "") + new_text
        old_feat = original_cache.get("audio_feat", new_audio_feat.new_empty(0, *new_audio_feat.shape[1:]))
        merged["audio_feat"] = torch.cat([old_feat, new_audio_feat], dim=0)
        merged["mode"] = "ref_continuation" if "ref_audio_feat" in merged else "continuation"
        return merged

    def generate_with_prompt_cache(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return next_and_close(self._generate_with_prompt_cache(*args, streaming=False, **kwargs))

    def generate_with_prompt_cache_streaming(
        self, *args, **kwargs
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]], None, None]:
        return self._generate_with_prompt_cache(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate_with_prompt_cache(
        self,
        target_text: str,
        prompt_cache: dict,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """
        Generate audio using pre-built prompt cache.

        Args:
            target_text: Text to convert to speech
            prompt_cache: Cache built by ``build_prompt_cache()``. Can be None
                for zero-shot generation.
            min_len: Minimum audio length to avoid very short audio
            max_len: Maximum audio length
            inference_timesteps: Number of diffusion sampling steps
            cfg_value: Classifier-free guidance value
            retry_badcase: Whether to retry on bad cases
            retry_badcase_max_times: Maximum retry attempts
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio
            streaming: Whether to return a generator of audio chunks
            streaming_prefix_len: Number of prefix audio patches to use for streaming mode

        Returns:
            Generator of Tuple containing:
                - Decoded audio tensor for the current step if ``streaming=True``, else final decoded audio tensor
                - Tensor of new text tokens
                - New audio features up to the current step as a List if ``streaming=True``, else as a concatenated Tensor
        """
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False

        # Determine mode from cache
        if prompt_cache is None:
            mode = "zero_shot"
            text = target_text
        else:
            mode = prompt_cache.get("mode", "continuation")
            if mode in ("continuation", "ref_continuation"):
                prompt_text = prompt_cache.get("prompt_text", "")
                text = prompt_text + target_text
            else:
                text = target_text

        text_token = torch.LongTensor(self.text_tokenizer(text))
        text_token = torch.cat(
            [
                text_token,
                torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
            ],
            dim=-1,
        )

        target_text_token = torch.LongTensor(self.text_tokenizer(target_text))
        text_length = text_token.shape[0]

        if mode in ("zero_shot", "continuation"):
            prompt_audio_feat = (
                prompt_cache["audio_feat"]
                if prompt_cache
                else torch.empty((0, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32)
            )
            audio_length = prompt_audio_feat.size(0)
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([text_token, text_pad_token])
            audio_feat = torch.cat([text_pad_feat, prompt_audio_feat], dim=0)
            text_mask = torch.cat(
                [torch.ones(text_length, dtype=torch.int32), torch.zeros(audio_length, dtype=torch.int32)]
            ).to(text_token.device)
            audio_mask = torch.cat(
                [torch.zeros(text_length, dtype=torch.int32), torch.ones(audio_length, dtype=torch.int32)]
            ).to(text_token.device)

        elif mode == "reference":
            ref_audio_feat = prompt_cache["ref_audio_feat"]
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_audio_feat, text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([ref_tokens, text_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat], dim=0)
            text_mask = torch.cat([ref_t_mask, torch.ones(text_length, dtype=torch.int32).to(text_token.device)])
            audio_mask = torch.cat([ref_a_mask, torch.zeros(text_length, dtype=torch.int32).to(text_token.device)])

        else:
            # ref_continuation mode
            ref_audio_feat = prompt_cache["ref_audio_feat"]
            prompt_audio_feat = prompt_cache["audio_feat"]
            prompt_audio_length = prompt_audio_feat.size(0)

            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_audio_feat, text_token.device)

            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )

            text_token = torch.cat([ref_tokens, text_token, prompt_pad_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat, prompt_audio_feat], dim=0)
            text_mask = torch.cat(
                [
                    ref_t_mask,
                    torch.ones(text_length, dtype=torch.int32).to(text_token.device),
                    torch.zeros(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask,
                    torch.zeros(text_length, dtype=torch.int32).to(text_token.device),
                    torch.ones(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )

        text_token = text_token.unsqueeze(0).to(self.device)
        text_mask = text_mask.unsqueeze(0).to(self.device)
        audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
        audio_mask = audio_mask.unsqueeze(0).to(self.device)

        # run inference
        target_text_length = len(self.text_tokenizer(target_text))
        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len),
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )
            if streaming:
                patch_len = self._decoded_samples_per_patch()
                for latent_pred, pred_audio_feat in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                    yield (decode_audio, target_text_token, pred_audio_feat)
                break
            else:
                latent_pred, pred_audio_feat = next_and_close(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(
                            f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...",
                            file=sys.stderr,
                        )
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
            patch_len = self._decoded_samples_per_patch()
            if mode in ("continuation", "ref_continuation"):
                decode_audio = decode_audio[..., patch_len * (streaming_prefix_len - 1) :].squeeze(1).cpu()
            else:
                decode_audio = decode_audio[..., :].squeeze(1).cpu()
            yield (decode_audio, target_text_token, pred_audio_feat)

    def inference(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return next_and_close(self._inference(*args, streaming=False, **kwargs))

    def inference_streaming(self, *args, **kwargs) -> Generator[Tuple[torch.Tensor, List[torch.Tensor]], None, None]:
        return self._inference(*args, streaming=True, **kwargs)

    # ------------------------------------------------------------------
    # Batch generation (for SVS validation / inference)
    # ------------------------------------------------------------------

    # Token IDs matching training-time prompt audio boundary tokens
    PROMPT_AUDIO_START_ID = 103
    PROMPT_AUDIO_END_ID = 104

    def generate_batch(
        self,
        target_texts: List[str],
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        verbose: bool = True,
        temperature: float = 1.0,
        temperature_mode: str = "scale",
        fsq_temperature: float = 0.0,
        prompt_audio_feats: Optional[List[Optional[torch.Tensor]]] = None,
        return_latents: bool = False,
        return_latent_dtype: str = "bfloat16",
    ):
        """
        Generate audio for multiple texts in a single batch.

        Args:
            target_texts: List of text prompts to generate audio for
            min_len: Minimum generation length (patches)
            max_len: Maximum generation length (patches)
            inference_timesteps: Number of diffusion steps
            cfg_value: Classifier-free guidance value
            verbose: Whether to show progress bar
            temperature: Temperature value for sampling diversity
            temperature_mode: "scale" or "ditar"
            prompt_audio_feats: Optional list of per-sample prompt audio features.
                Each element is either a [T_i, P, D] tensor (VAE latent patches)
                or None (no prompt for that sample).  When provided, the prompt
                is prepended in the training format:
                [<prompt_audio_start>(103) | feats | <prompt_audio_end>(104)]
            return_latent_dtype: dtype used for the optional returned
                `[T, P, D]` latents. Keep the default `bfloat16` for
                storage-efficient dumps; request `float32` when the
                latents will be decoded or re-used as training targets.

        Returns:
            List of decoded audio tensors, one per input text
        """
        return self._generate_batch(
            target_texts=target_texts,
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            verbose=verbose,
            temperature=temperature,
            temperature_mode=temperature_mode,
            fsq_temperature=fsq_temperature,
            prompt_audio_feats=prompt_audio_feats,
            return_latents=return_latents,
            return_latent_dtype=return_latent_dtype,
        )

    @torch.inference_mode()
    def _generate_batch(
        self,
        target_texts: List[str],
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        verbose: bool = True,
        temperature: float = 1.0,
        temperature_mode: str = "scale",
        fsq_temperature: float = 0.0,
        prompt_audio_feats: Optional[List[Optional[torch.Tensor]]] = None,
        return_latents: bool = False,
        return_latent_dtype: str = "bfloat16",
    ):
        """
        Internal batch generation method for VoxCPM2.

        Prepares batch inputs with padding and calls _inference_batch.
        When *prompt_audio_feats* is provided, the prompt audio is prepended
        to each sample in the same format used during training:
            [<prompt_audio_start>(103) | audio_feats | <prompt_audio_end>(104) | SVS_tokens | audio_start(101)]
        """
        batch_size = len(target_texts)
        if batch_size == 0:
            return []

        # Setup KV cache for current batch size
        self.base_lm.setup_cache(batch_size, self.config.max_length, self.device, get_dtype(self.config.dtype))
        self.residual_lm.setup_cache(batch_size, self.config.max_length, self.device, get_dtype(self.config.dtype))

        P = self.patch_size
        D = self.audio_vae.latent_dim

        # Tokenize all texts and optionally prepend prompt audio prefix
        text_tokens_list = []
        text_mask_list = []
        audio_mask_list = []
        audio_feat_list = []

        for i, text in enumerate(target_texts):
            # Tokenize SVS prompt + audio_start
            svs_tokens = torch.LongTensor(self.text_tokenizer(text))
            svs_tokens = torch.cat([
                svs_tokens,
                torch.tensor([self.audio_start_token], dtype=torch.int32),
            ])
            svs_len = svs_tokens.shape[0]

            # -- Check for prompt audio --
            pa = None
            if prompt_audio_feats is not None and i < len(prompt_audio_feats):
                pa = prompt_audio_feats[i]

            if pa is not None and pa.shape[0] > 0:
                T_prompt = pa.shape[0]
                # Text tokens: [103, 0...0(T_prompt), 104, ...svs_tokens...]
                prompt_text = torch.tensor(
                    [self.PROMPT_AUDIO_START_ID] + [0] * T_prompt + [self.PROMPT_AUDIO_END_ID],
                    dtype=torch.int32,
                )
                full_tokens = torch.cat([prompt_text, svs_tokens])

                # Text mask: boundary=1, audio_positions=0, svs_region=1
                prompt_text_mask = torch.tensor(
                    [1] + [0] * T_prompt + [1], dtype=torch.int32,
                )
                svs_text_mask = torch.ones(svs_len, dtype=torch.int32)
                full_text_mask = torch.cat([prompt_text_mask, svs_text_mask])

                # Audio mask: boundary=0, audio_positions=1, svs_region=0
                prompt_audio_mask = torch.tensor(
                    [0] + [1] * T_prompt + [0], dtype=torch.int32,
                )
                svs_audio_mask = torch.zeros(svs_len, dtype=torch.int32)
                full_audio_mask = torch.cat([prompt_audio_mask, svs_audio_mask])

                # Audio features: zero-pad for boundary tokens, prompt audio between
                prompt_audio_padded = torch.cat([
                    torch.zeros(1, P, D, dtype=torch.float32),   # <prompt_audio_start>
                    pa.to(torch.float32),                        # [T_prompt, P, D]
                    torch.zeros(1, P, D, dtype=torch.float32),   # <prompt_audio_end>
                ], dim=0)
                svs_audio = torch.zeros(svs_len, P, D, dtype=torch.float32)
                full_audio = torch.cat([prompt_audio_padded, svs_audio], dim=0)
            else:
                # No prompt audio — original path
                full_tokens = svs_tokens
                full_text_mask = torch.ones(svs_len, dtype=torch.int32)
                full_audio_mask = torch.zeros(svs_len, dtype=torch.int32)
                full_audio = torch.zeros(svs_len, P, D, dtype=torch.float32)

            text_tokens_list.append(full_tokens)
            text_mask_list.append(full_text_mask)
            audio_mask_list.append(full_audio_mask)
            audio_feat_list.append(full_audio)

        # Find max length for padding
        max_seq_len = max(t.shape[0] for t in text_tokens_list)

        # Left-pad and stack
        padded_tokens = []
        padded_text_masks = []
        padded_audio_masks = []
        padded_audio_feats = []

        for tokens, tmask, amask, afeat in zip(
            text_tokens_list, text_mask_list, audio_mask_list, audio_feat_list
        ):
            seq_len = tokens.shape[0]
            pad_len = max_seq_len - seq_len

            padded_tokens.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.int32), tokens
            ]))
            padded_text_masks.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.int32), tmask
            ]))
            padded_audio_masks.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.int32), amask
            ]))
            padded_audio_feats.append(torch.cat([
                torch.zeros(pad_len, P, D, dtype=torch.float32), afeat
            ], dim=0))

        # Stack into batched tensors
        text_tokens = torch.stack(padded_tokens, dim=0).to(self.device)
        text_mask = torch.stack(padded_text_masks, dim=0).to(self.device)
        audio_mask = torch.stack(padded_audio_masks, dim=0).to(self.device)
        audio_feat = torch.stack(padded_audio_feats, dim=0).to(
            device=self.device, dtype=get_dtype(self.config.dtype),
        )

        # Run batch inference
        latent_preds, pred_lengths = self._inference_batch(
            text=text_tokens,
            text_mask=text_mask,
            feat=audio_feat,
            feat_mask=audio_mask,
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            verbose=verbose,
        )

        # Decode in small batches to avoid VAE peak-memory spikes while still
        # amortizing Python overhead better than pure per-sample decoding.
        decoded_audios = []
        P = self.patch_size
        latents_per_sample: List[Optional[torch.Tensor]] = [] if return_latents else None
        latent_return_dtype = get_dtype(return_latent_dtype)
        if latent_preds.shape[-1] == 0:
            decoded_audios = [
                torch.empty((1, 0), dtype=torch.float32)
                for _ in range(batch_size)
            ]
            if return_latents:
                latents_per_sample = [
                    torch.empty((0, P, latent_preds.shape[1]), dtype=latent_return_dtype)
                    for _ in range(batch_size)
                ]
        else:
            samples_per_patch = self._decoded_samples_per_patch()
            decode_batch_size = min(batch_size, 4)
            for start in range(0, batch_size, decode_batch_size):
                end = min(start + decode_batch_size, batch_size)
                audio_chunk = self.audio_vae.decode(
                    latent_preds[start:end].to(torch.float32)
                ).squeeze(1).cpu()
                for local_idx, pred_len in enumerate(pred_lengths[start:end]):
                    sample_num_samples = pred_len * samples_per_patch
                    decoded_audios.append(
                        audio_chunk[local_idx:local_idx + 1, :sample_num_samples]
                    )
                    if return_latents:
                        abs_idx = start + local_idx
                        flat = latent_preds[abs_idx, :, :pred_len * P].to(latent_return_dtype).cpu()
                        lat = flat.transpose(0, 1).contiguous().view(pred_len, P, -1)
                        latents_per_sample.append(lat)

        # Reset cache to batch_size=1
        self.base_lm.setup_cache(1, self.config.max_length, self.device, get_dtype(self.config.dtype))
        self.residual_lm.setup_cache(1, self.config.max_length, self.device, get_dtype(self.config.dtype))

        if return_latents:
            return decoded_audios, latents_per_sample
        return decoded_audios

    @torch.inference_mode()
    def _inference_batch(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Core batch inference method for VoxCPM2 audio generation.

        Processes multiple samples in parallel with per-sample stop conditions.
        Uses V2's concat fusion for the residual LM.

        Args:
            text: Batched input text tokens [B, T]
            text_mask: Mask for text tokens [B, T]
            feat: Batched input audio features [B, T, P, D]
            feat_mask: Mask for audio features [B, T]

        Returns:
            Tuple of:
                - Predicted latent features [B, D, max_pred_len * P]
                - List of actual prediction lengths per sample
        """
        B, T, P, D = feat.shape
        target_dtype = get_dtype(self.config.dtype)
        batch_indices = torch.arange(B, device=self.device)

        feat_embed = self.feat_encoder(feat)
        feat_embed = self.enc_to_lm_proj(feat_embed)

        if self.config.lm_config.use_mup:
            scale_emb = self.config.lm_config.scale_emb
        else:
            scale_emb = 1.0

        text_embed = self.base_lm.embed_tokens(text) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

        prefill_kpm, persistent_kpm = build_prefill_and_persistent_kpm(
            text_mask, feat_mask, self.base_lm.kv_cache.max_length,
        )

        # Pre-allocate output buffer and keep stop bookkeeping on-device. The
        # older Python-list path forced a host sync (`cpu().tolist()`) every
        # AR step, which scales badly under the multi-GPU thread pool.
        pred_feat_buf = torch.zeros(B, max_len, P, D, device=self.device, dtype=target_dtype)
        gen_lengths = torch.zeros(B, dtype=torch.long, device=self.device)
        prefix_feat_cond = feat[:, -1, ...]  # [B, P, D]
        stopped = torch.zeros(B, dtype=torch.bool, device=self.device)
        stop_lengths = torch.zeros(B, dtype=torch.long, device=self.device)
        base_step = torch.empty(1, dtype=torch.long, device=self.device)
        residual_step = torch.empty(1, dtype=torch.long, device=self.device)

        enc_outputs, kv_cache_tuple = self.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
            key_padding_mask=prefill_kpm,
        )
        self.base_lm.kv_cache.fill_caches(kv_cache_tuple)

        enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]  # [B, H]

        # V2 uses concat fusion for residual LM input
        residual_enc_inputs = self.fusion_concat_proj(
            torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed), dim=-1)
        )
        residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
            inputs_embeds=residual_enc_inputs,
            is_causal=True,
            key_padding_mask=prefill_kpm,
        )
        self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        residual_hidden = residual_enc_outputs[:, -1, :]  # [B, H]

        pbar = tqdm(range(max_len), disable=not verbose, desc=f"Batch inference (B={B})")
        for i in pbar:
            active_mask = ~stopped
            active_count = int(active_mask.sum().item())
            if active_count == 0:
                break

            # V2 uses concatenated dit_hidden (not additive)
            dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)
            dit_hidden_2 = self.res_to_dit_proj(residual_hidden)
            dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)

            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            ).transpose(1, 2)  # [B, P, D]

            curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))
            curr_embed = self.enc_to_lm_proj(curr_embed)

            # Write predictions directly into the pre-allocated buffer.
            # Audio eval under FSDP summons params in fp32, so ``feat_decoder``
            # may return fp32 while ``pred_feat_buf`` is allocated in
            # ``target_dtype`` (bf16 under amp_bf16). Cast to the buf dtype
            # explicitly to avoid "Index put requires dtypes match" at eval.
            active_indices = batch_indices[active_mask]
            write_steps = gen_lengths[active_indices]
            pred_feat_buf[active_indices, write_steps] = pred_feat[active_indices].to(pred_feat_buf.dtype)
            gen_lengths[active_indices] = write_steps + 1

            prefix_feat_cond = pred_feat

            # Check stop condition per sample
            stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
            stop_flags = stop_logits.argmax(dim=-1)
            newly_stopped = active_mask & (i > min_len) & (stop_flags == 1)
            stop_lengths = torch.where(newly_stopped, gen_lengths, stop_lengths)
            stopped = stopped | newly_stopped

            if verbose:
                pbar.set_postfix(active=int((~stopped).sum().item()))

            # Update LM states for next step
            base_step.fill_(self.base_lm.kv_cache.step())
            lm_hidden = self.base_lm.forward_step(
                curr_embed[:, 0, :],
                base_step,
                key_padding_mask=persistent_kpm,
            ).clone()

            lm_hidden = self.fsq_layer(lm_hidden)
            # V2: concat fusion for residual step
            curr_residual_input = self.fusion_concat_proj(
                torch.cat((lm_hidden, curr_embed[:, 0, :]), dim=-1)
            )
            residual_step.fill_(self.residual_lm.kv_cache.step())
            residual_hidden = self.residual_lm.forward_step(
                curr_residual_input,
                residual_step,
                key_padding_mask=persistent_kpm,
            ).clone()

        # Finalize results — assign lengths for samples that never stopped
        stop_lengths = torch.where(stopped, stop_lengths, gen_lengths)

        max_pred_len = int(stop_lengths.max().item()) if stop_lengths.numel() > 0 else 0

        if max_pred_len == 0:
            return torch.zeros(B, D, 0, device=self.device, dtype=target_dtype), stop_lengths.cpu().tolist()

        # Trim buffer to actual max generated length and reshape
        pred_feat_all = pred_feat_buf[:, :max_pred_len]
        feat_pred = rearrange(pred_feat_all, "b t p d -> b d (t p)", b=B, p=self.patch_size)

        return feat_pred, stop_lengths.cpu().tolist()

    @torch.inference_mode()
    def _inference(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """Core inference method for audio generation.

        This is the main inference loop that generates audio features
        using the language model and diffusion transformer.

        Args:
            text: Input text tokens
            text_mask: Mask for text tokens
            feat: Input audio features
            feat_mask: Mask for audio features
            min_len: Minimum generation length
            max_len: Maximum generation length
            inference_timesteps: Number of diffusion steps
            cfg_value: Classifier-free guidance value
            streaming: Whether to yield each step latent feature or just the final result

        Returns:
            Generator of Tuple containing:
                - Predicted latent feature at the current step if ``streaming=True``, else final latent features
                - Predicted audio feature sequence so far as a List if ``streaming=True``, else as a concatenated Tensor
        """
        B, T, P, D = feat.shape

        feat_embed = self.feat_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)

        if self.config.lm_config.use_mup:
            scale_emb = self.config.lm_config.scale_emb
        else:
            scale_emb = 1.0

        text_embed = self.base_lm.embed_tokens(text) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

        prefix_feat_cond = feat[:, -1, ...]  # b, p, d
        pred_feat_seq = []  # b, t, p, d
        curr_embed = None

        # Prepare prompt context patches for streaming mode
        # - Continuation modes (feat_mask ends with 1): use the last (streaming_prefix_len - 1)
        #   trailing audio patches as initial context so the VAE can decode smoothly.
        # - Reference-only / zero-shot (feat_mask ends with 0): start from scratch.
        has_continuation_audio = feat_mask[0, -1].item() == 1
        if has_continuation_audio:
            audio_indices = feat_mask.squeeze(0).nonzero(as_tuple=True)[0]
            context_len = min(streaming_prefix_len - 1, len(audio_indices))
            last_audio_indices = audio_indices[-context_len:]
            pred_feat_seq = list(feat[:, last_audio_indices, :, :].split(1, dim=1))
        else:
            pred_feat_seq = []

        enc_outputs, kv_cache_tuple = self.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )
        self.base_lm.kv_cache.fill_caches(kv_cache_tuple)

        enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

        residual_enc_inputs = self.fusion_concat_proj(
            torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed), dim=-1)
        )
        residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
            inputs_embeds=residual_enc_inputs,
            is_causal=True,
        )
        self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        residual_hidden = residual_enc_outputs[:, -1, :]

        for i in tqdm(range(max_len)):
            dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
            dit_hidden_2 = self.res_to_dit_proj(residual_hidden)  # [b, h_dit]
            dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)

            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            ).transpose(
                1, 2
            )  # [b, p, d]

            curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, c
            curr_embed = self.enc_to_lm_proj(curr_embed)

            pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
            prefix_feat_cond = pred_feat

            if streaming:
                # return the last three predicted latent features to provide enough context for smooth decoding
                pred_feat_chunk = torch.cat(pred_feat_seq[-streaming_prefix_len:], dim=1)
                feat_pred = rearrange(pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size)

                yield feat_pred, pred_feat_seq

            stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
            if i > min_len and stop_flag == 1:
                break

            lm_hidden = self.base_lm.forward_step(
                curr_embed[:, 0, :], torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device)
            ).clone()

            lm_hidden = self.fsq_layer(lm_hidden)
            curr_residual_input = self.fusion_concat_proj(torch.cat((lm_hidden, curr_embed[:, 0, :]), dim=-1))
            residual_hidden = self.residual_lm.forward_step(
                curr_residual_input, torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device)
            ).clone()

        if not streaming:
            pred_feat_seq = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d
            feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)
            yield feat_pred, pred_feat_seq.squeeze(0).cpu()

    @classmethod
    def from_local(
        cls,
        path: str,
        training: bool = False,
        lora_config: LoRAConfig = None,
    ):
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as _cfg_f:
            config = VoxCPMConfig.model_validate_json(_cfg_f.read())
        tokenizer = LlamaTokenizerFast.from_pretrained(path)
        audio_vae_config = getattr(config, "audio_vae_config", None)
        audio_vae = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()
        # Try to load AudioVAE from safetensors first, fallback to pytorch
        audiovae_safetensors_path = os.path.join(path, "audiovae.safetensors")
        audiovae_pth_path = os.path.join(path, "audiovae.pth")
        if os.path.exists(audiovae_safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"Loading AudioVAE from safetensors: {audiovae_safetensors_path}", file=sys.stderr)
            vae_state_dict = load_file(audiovae_safetensors_path, device="cpu")
        elif os.path.exists(audiovae_pth_path):
            print(f"Loading AudioVAE from pytorch: {audiovae_pth_path}", file=sys.stderr)
            checkpoint = torch.load(
                audiovae_pth_path,
                map_location="cpu",
                weights_only=True,
            )
            vae_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(
                f"AudioVAE checkpoint not found. Expected either {audiovae_safetensors_path} or {audiovae_pth_path}"
            )
        model = cls(
            config,
            tokenizer,
            audio_vae,
            lora_config,
        )
        if not training:
            lm_dtype = get_dtype(model.config.dtype)
            model = model.to(lm_dtype)
        else:  # training mode
            for name, param in model.named_parameters():
                if "audio_vae" in name:  # freeze VAE weights
                    param.requires_grad = False
                    continue
                if lora_config is not None:
                    if "lora" not in name:  # freeze non-LoRA weights
                        param.requires_grad = False
            # KV cache is only needed for autoregressive inference. Releasing it
            # in training mode avoids a fixed per-rank GPU allocation.
            model.base_lm.kv_cache = None
            model.residual_lm.kv_cache = None
        model.audio_vae = model.audio_vae.to(torch.float32)

        # Try to load from safetensors first, fallback to pytorch_model.bin
        safetensors_path = os.path.join(path, "model.safetensors")
        pytorch_model_path = os.path.join(path, "pytorch_model.bin")

        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"Loading model from safetensors: {safetensors_path}", file=sys.stderr)
            model_state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_model_path):
            print(f"Loading model from pytorch_model.bin: {pytorch_model_path}", file=sys.stderr)
            checkpoint = torch.load(
                pytorch_model_path,
                map_location="cpu",
                weights_only=True,
            )
            model_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(f"Model file not found. Expected either {safetensors_path} or {pytorch_model_path}")

        for kw, val in vae_state_dict.items():
            model_state_dict[f"audio_vae.{kw}"] = val

        # Auto-resize embedding if checkpoint has a larger vocab (e.g., SVS-
        # extended checkpoint saved with resized embeddings but stale
        # config.lm_config.vocab_size).
        embed_key = "base_lm.embed_tokens.weight"
        if embed_key in model_state_dict:
            ckpt_vocab_size = model_state_dict[embed_key].shape[0]
            model_vocab_size = model.base_lm.embed_tokens.weight.shape[0]
            if ckpt_vocab_size != model_vocab_size:
                print(
                    f"Auto-resizing embedding: {model_vocab_size} -> "
                    f"{ckpt_vocab_size} to match checkpoint",
                    file=sys.stderr,
                )
                old_emb = model.base_lm.embed_tokens
                _, emb_dim = old_emb.weight.shape
                new_emb = torch.nn.Embedding(
                    ckpt_vocab_size,
                    emb_dim,
                    dtype=old_emb.weight.dtype,
                    device=old_emb.weight.device,
                )
                new_emb.weight.data[:model_vocab_size] = old_emb.weight.data
                model.base_lm.embed_tokens = new_emb

        # LoRALinear keeps weight/bias compatible with nn.Linear but adds
        # lora_A/lora_B, which are absent from base pretrained checkpoints.
        model.load_state_dict(model_state_dict, strict=False)
        if training:
            return model
        return model.to(model.device).eval()

    # ------------------------------------------------------------------ #
    # LoRA Weight Management
    # ------------------------------------------------------------------ #
    def _iter_lora_modules(self):
        """Iterate over all LoRA modules."""
        from ..modules.layers.lora import LoRALinear

        for module in self.modules():
            if isinstance(module, LoRALinear):
                yield module

    def load_lora_weights(self, lora_path: str, device: str = None):
        """Load LoRA weights from a safetensors or pytorch checkpoint.

        Args:
            lora_path: Checkpoint path (directory or .safetensors/.ckpt file)
            device: Target device, defaults to model's current device
        Returns:
            tuple: (loaded_keys, skipped_keys)
        """
        from pathlib import Path

        device = device or self.device
        lora_p = Path(lora_path)

        # Try safetensors first, then fallback to .ckpt
        if lora_p.is_dir():
            safetensors_file = lora_p / "lora_weights.safetensors"
            ckpt_file = lora_p / "lora_weights.ckpt"
        else:
            safetensors_file = lora_p if lora_p.suffix == ".safetensors" else None
            ckpt_file = lora_p if lora_p.suffix in [".ckpt", ".pth"] else None

        # Load from safetensors if available
        if safetensors_file and safetensors_file.exists() and SAFETENSORS_AVAILABLE:
            state_dict = load_file(str(safetensors_file), device=device)
        elif ckpt_file and ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
        else:
            raise FileNotFoundError(f"LoRA checkpoint not found. Expected either {safetensors_file} or {ckpt_file}")

        model_params = dict(self.named_parameters())

        loaded_keys, skipped_keys = [], []
        for key, value in state_dict.items():
            if key in model_params:
                model_params[key].data.copy_(value.to(device))
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)

        return loaded_keys, skipped_keys

    def set_lora_enabled(self, enabled: bool):
        """Enable/disable all LoRA layers."""
        for module in self._iter_lora_modules():
            module.set_enabled(enabled)

    def reset_lora_weights(self):
        """Reset all LoRA weights (A: kaiming, B: zeros), effectively unloading LoRA."""
        for module in self._iter_lora_modules():
            module.reset_lora_parameters()

    def get_lora_state_dict(self) -> dict:
        """Get all LoRA parameters (lora_A/lora_B)."""
        return {name: param.data.clone() for name, param in self.named_parameters() if "lora_" in name}
