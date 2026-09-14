"""
Audio utility functions shared between test and validation.
"""

import numpy as np
import torch


def normalize_audio(audio_np: np.ndarray) -> np.ndarray:
    """Normalize audio to [-0.9, 0.9] range."""
    max_val = np.abs(audio_np).max()
    if max_val > 0:
        return audio_np / max_val * 0.9
    return audio_np


def decode_reference_audio(sample: dict, audio_vae, device: str, sample_rate: int = 44100) -> np.ndarray | None:
    """
    Decode reference audio from a preprocessed sample.

    Preprocessed data stores audio as VAE latent features in 'audio_feats' column.
    We use the 'audio_mask' to extract only valid audio frames (skipping text-position padding).

    Args:
        sample: A single dataset sample dict containing 'audio_feats' and optionally 'audio_mask'.
        audio_vae: The AudioVAE model instance.
        device: Device string (e.g. "cuda").
        sample_rate: Audio sample rate (unused in decoding, kept for API consistency).

    Returns:
        1-D numpy array of decoded audio, or None on failure.
    """
    from einops import rearrange

    if "audio_feats" not in sample:
        return None

    audio_feats = sample["audio_feats"]
    audio_mask = sample.get("audio_mask", None)

    # Convert to tensor if needed (Arrow datasets return lists)
    if isinstance(audio_feats, list):
        audio_feats = torch.tensor(audio_feats, dtype=torch.float32)
    if audio_mask is not None and isinstance(audio_mask, list):
        audio_mask = torch.tensor(audio_mask, dtype=torch.int32)

    if not isinstance(audio_feats, torch.Tensor):
        return None

    # Extract only valid audio frames using audio_mask
    if audio_mask is not None and isinstance(audio_mask, torch.Tensor):
        audio_indices = (audio_mask == 1).nonzero(as_tuple=True)[0]
        if len(audio_indices) > 0:
            audio_feats = audio_feats[audio_indices]
        else:
            return None

    # Reverse preprocessing: [T, P, D] -> [1, T*P, D] -> [1, D, T*P] -> decode
    audio_feats = rearrange(audio_feats, "t p d -> 1 (t p) d")
    audio_feats = audio_feats.transpose(1, 2)  # [1, D, T*P]
    audio_feats = audio_feats.to(device).float()

    with torch.no_grad():
        ref_audio = audio_vae.decode(audio_feats)  # [1, 1, samples]

    return ref_audio.cpu().float().numpy().flatten()


def decode_latents_batched(
    latents,
    audio_vae,
    device: str,
    sample_rate: int = 44100,
    max_batch_tokens: int = 0,
) -> list:
    """Decode a list of ``(T, P, D)`` VAE latents to 1-D fp32 mono waveforms in
    batches.

    Variable-length latents are right-padded to the per-batch max along the
    ``(T*P)`` time axis, decoded as one ``[B, D, L_max]`` tensor, then each
    decoded waveform is trimmed back to its true length ``L_i * hop_length``.
    The AudioVAE decoder is a *causal* conv stack (output sample ``t`` depends
    only on inputs ``<= t``), so right-padding + trimming is numerically
    identical to a ``B=1`` decode up to fp reduction-order jitter (~1e-6).

    Args:
        latents: list of ``(T, P, D)`` tensors (bf16/fp32) or lists; ``None``
            entries pass through as ``None``.
        audio_vae: AudioVAE (v1) or AudioVAEV2 instance.
        device: device string for the decode.
        sample_rate: fallback output sample rate (only used if the VAE does not
            expose ``out_sample_rate``).
        max_batch_tokens: in-batch ``T*P`` frame budget; ``<=0`` packs all
            latents into a single batch.

    Returns:
        list aligned to ``latents``; each entry is a 1-D fp32 numpy waveform or
        ``None`` (for ``None`` / invalid inputs).
    """
    from einops import rearrange

    results = [None] * len(latents)

    # Flatten valid latents to [D, L] and record original index + length.
    items = []  # (orig_idx, z[D, L], L)
    for i, lat in enumerate(latents):
        if lat is None:
            continue
        t = lat
        if isinstance(t, list):
            t = torch.tensor(t, dtype=torch.float32)
        if not isinstance(t, torch.Tensor):
            continue
        z = rearrange(t, "tt p d -> (tt p) d").transpose(0, 1).contiguous()  # [D, T*P]
        items.append((i, z, int(z.shape[-1])))
    if not items:
        return results

    is_v2 = getattr(audio_vae, "sr_bin_boundaries", None) is not None
    out_sr = int(getattr(audio_vae, "out_sample_rate", sample_rate))

    # Sort by length so each padded batch packs similar-length latents tightly.
    items.sort(key=lambda x: x[2])

    # Group into batches by *padded* volume. Because items are length-sorted
    # ascending, the newest item's length is the batch max, so
    # ``L * (len(cur)+1)`` is exactly the padded ``[B, D, L_max]`` token count —
    # budgeting on that (not the sum of true lengths) bounds the compute wasted
    # on zero-padding when a batch mixes short and long latents.
    batches, cur = [], []
    for it in items:
        L = it[2]
        if cur and max_batch_tokens > 0 and L * (len(cur) + 1) > max_batch_tokens:
            batches.append(cur)
            cur = []
        cur.append(it)
    if cur:
        batches.append(cur)

    for batch in batches:
        L_max = max(it[2] for it in batch)
        B = len(batch)
        D = batch[0][1].shape[0]
        z = torch.zeros(B, D, L_max, dtype=torch.float32)
        for bi, (_, zz, L) in enumerate(batch):
            z[bi, :, :L] = zz.float()
        z = z.to(device)
        with torch.no_grad():
            if is_v2:
                sr_cond = torch.full((B,), out_sr, device=device, dtype=torch.int32)
                out = audio_vae.decode(z, sr_cond)  # [B, 1, S_max]
            else:
                out = audio_vae.decode(z)  # [B, 1, S_max]
        out = out.cpu().float()
        S_max = out.shape[-1]
        # The decoder upsamples every latent frame by a fixed integer factor, so
        # the per-frame sample count is exactly S_max // L_max (architecture- and
        # sample-rate-agnostic; works for v1 where in_sr==out_sr and for v2 where
        # the output is resampled to out_sample_rate). Trimming each row to its
        # true length is what keeps batching lossless vs the B=1 decode.
        ratio = S_max // L_max
        for bi, (orig_idx, _, L) in enumerate(batch):
            n_samp = min(L * ratio, S_max)
            results[orig_idx] = out[bi, 0, :n_samp].numpy().flatten()

    return results
