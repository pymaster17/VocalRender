"""
Evaluation metrics for SVS audio quality assessment.

Includes SingMOS (Singing Voice Mean Opinion Score) and Audiobox
Aesthetics (AES) evaluation.
"""

import torch
import numpy as np
import sys


# ============================================================
# SingMOS Evaluation
# ============================================================

def load_singmos_predictor(device: torch.device, model_name: str = "singmos_pro",
                           local_model_path: str = None):
    """Load SingMOS model for evaluation.

    Args:
        device: Torch device to load the model on.
        model_name: Name of the SingMOS model variant.
        local_model_path: Optional local path to model weights (skips download).

    Returns:
        SingMOS predictor model on the specified device.
    """
    # s3prl (SingMOS dependency) still calls the deprecated
    # ``torchaudio.set_audio_backend`` at import time, which was removed in
    # torchaudio 2.1+. Shim it to a no-op before loading so s3prl's upstream
    # modules import cleanly. The call was advisory (picked ``sox_io`` /
    # ``soundfile``); modern torchaudio auto-dispatches, so skipping it is
    # functionally equivalent.
    import torchaudio
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None

    # torchaudio pip wheels built without libsox no longer ship the
    # ``torchaudio.sox_effects`` submodule, which s3prl still imports at
    # module load time. Inject a minimal shim that maps the handful of
    # effects s3prl actually uses onto ``torchaudio.functional``.
    import sys as _sys
    if "torchaudio.sox_effects" not in _sys.modules:
        try:
            import torchaudio.sox_effects  # noqa: F401
        except ModuleNotFoundError:
            import types as _types
            import torchaudio.functional as _F

            def _apply_effects_tensor(tensor, sample_rate, effects, channels_first=True):
                wav = tensor if channels_first else tensor.transpose(0, 1)
                sr = int(sample_rate)
                for eff in effects:
                    name = eff[0]
                    args = eff[1:]
                    if name == "rate":
                        new_sr = int(float(args[0]))
                        wav = _F.resample(wav, sr, new_sr)
                        sr = new_sr
                    elif name == "channels":
                        tgt = int(args[0])
                        if wav.shape[0] != tgt:
                            if tgt == 1:
                                wav = wav.mean(dim=0, keepdim=True)
                            else:
                                wav = wav.expand(tgt, -1).contiguous()
                    elif name in ("gain", "norm"):
                        pass  # best-effort: skip normalization / gain
                    # silently ignore other effects
                if not channels_first:
                    wav = wav.transpose(0, 1)
                return wav, sr

            shim = _types.ModuleType("torchaudio.sox_effects")
            shim.apply_effects_tensor = _apply_effects_tensor
            shim.apply_effects_file = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("torchaudio.sox_effects.apply_effects_file unavailable "
                             "(libsox not installed); load audio via soundfile instead")
            )
            shim.effect_names = lambda: []
            shim.init_sox_effects = lambda: None
            shim.shutdown_sox_effects = lambda: None
            _sys.modules["torchaudio.sox_effects"] = shim
            torchaudio.sox_effects = shim

    if local_model_path:
        return torch.hub.load(
            "South-Twilight/SingMOS:v1.1.2", model_name,
            pretrained=False, model_path=local_model_path, trust_repo=True,
        ).to(device)
    return torch.hub.load(
        "South-Twilight/SingMOS:v1.1.2", model_name, trust_repo=True,
    ).to(device)


class SingMOSFrameCapture:
    """Hook class to capture frame-level scores from SingMOS decoder."""

    def __init__(self):
        self.frame_scores = None
        self.handle = None

    def hook_fn(self, module, input, output):
        self.frame_scores = output.squeeze(-1)

    def register(self, decoder):
        self.handle = decoder.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()


def compute_singmos_masked_mean(frame_scores: torch.Tensor, lengths: torch.Tensor,
                                hop_size: int = 320) -> torch.Tensor:
    """Compute masked mean of frame scores based on audio lengths.

    Args:
        frame_scores: [B, T] tensor of per-frame SingMOS scores.
        lengths: [B] tensor of sample counts per audio.
        hop_size: Hop size for frame-to-sample conversion.

    Returns:
        [B] tensor of masked-mean scores.
    """
    batch_size, max_frames = frame_scores.shape
    valid_frames = torch.ceil(lengths.float() / hop_size).long()
    indices = torch.arange(max_frames, device=frame_scores.device).unsqueeze(0).expand(batch_size, -1)
    mask = indices < valid_frames.unsqueeze(1)
    return (frame_scores * mask.float()).sum(1) / valid_frames.float().clamp(min=1)


def compute_singmos_score(audio_waveform, sample_rate: int,
                          predictor, frame_capture: SingMOSFrameCapture,
                          device: torch.device, target_sr: int = 16000) -> float:
    """Compute SingMOS score for a single audio waveform.

    Args:
        audio_waveform: 1-D numpy array or torch Tensor of audio samples.
        sample_rate: Source sample rate.
        predictor: Loaded SingMOS model.
        frame_capture: Hook for capturing frame scores.
        device: Device to run inference on.
        target_sr: Target sample rate for SingMOS (default 16 kHz).

    Returns:
        SingMOS score (float).
    """
    import librosa

    audio_np = audio_waveform.cpu().numpy() if isinstance(audio_waveform, torch.Tensor) else audio_waveform
    if sample_rate != target_sr:
        audio_np = librosa.resample(audio_np.astype("float32"), orig_sr=sample_rate, target_sr=target_sr)

    wave = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)  # [1, T]
    length = torch.tensor([wave.shape[1]], dtype=torch.long, device=device)

    with torch.no_grad():
        _ = predictor(wave, length)
        score = compute_singmos_masked_mean(frame_capture.frame_scores, length).item()

    return score


def compute_batch_singmos_scores(audio_list, sample_rate: int, predictor,
                                 frame_capture: SingMOSFrameCapture, device: torch.device,
                                 target_sr: int = 16000, max_duration: float = 30.0,
                                 max_batch_seconds: float = 96.0,
                                 max_batch_size: int = 16) -> list:
    """Compute SingMOS scores for a list of audio waveforms.

    Uses duration-sorted greedy batching to minimize padding waste: samples are
    sorted by length, grouped into sub-batches whose total duration does not
    exceed *max_batch_seconds* and whose size does not exceed *max_batch_size*,
    then each sub-batch is padded only to its own maximum length and scored in
    a single forward pass.  Results are returned in the original input order.

    Args:
        audio_list: List of 1-D numpy arrays (audio samples).
        sample_rate: Source sample rate.
        predictor: Loaded SingMOS model.
        frame_capture: Hook for capturing frame scores.
        device: Device to run inference on.
        target_sr: Target sample rate for SingMOS (default 16 kHz).
        max_duration: Maximum evaluated duration per sample (seconds).
        max_batch_seconds: Maximum total duration (in seconds at *target_sr*)
            allowed in a single sub-batch.  Larger values use more GPU memory
            but require fewer forward passes.
        max_batch_size: Maximum number of samples in a single sub-batch.

    Returns:
        List of SingMOS scores (floats), same length and order as *audio_list*.
    """
    import librosa

    if not audio_list:
        return []

    max_samples = int(max_duration * target_sr)

    # ------------------------------------------------------------------
    # 1. Pre-process: resample & truncate, keep original indices
    # ------------------------------------------------------------------
    indexed_waves: list[tuple[int, torch.Tensor]] = []
    for idx, audio_np in enumerate(audio_list):
        if sample_rate != target_sr:
            audio_np = librosa.resample(audio_np.astype("float32"), orig_sr=sample_rate, target_sr=target_sr)
        if len(audio_np) > max_samples:
            audio_np = audio_np[:max_samples]
        indexed_waves.append((idx, torch.from_numpy(audio_np).float()))

    # ------------------------------------------------------------------
    # 2. Sort by duration (descending) for greedy packing
    # ------------------------------------------------------------------
    indexed_waves.sort(key=lambda x: x[1].shape[0], reverse=True)

    # ------------------------------------------------------------------
    # 3. Greedy batching: group similar-length samples together
    # ------------------------------------------------------------------
    max_batch_samples = int(max_batch_seconds * target_sr)
    batches: list[list[tuple[int, torch.Tensor]]] = []
    current_batch: list[tuple[int, torch.Tensor]] = []
    current_samples = 0

    for item in indexed_waves:
        duration = item[1].shape[0]
        if not current_batch:
            current_batch = [item]
            current_samples = duration
        elif current_samples + duration <= max_batch_samples and len(current_batch) < max_batch_size:
            current_batch.append(item)
            current_samples += duration
        else:
            batches.append(current_batch)
            current_batch = [item]
            current_samples = duration
    if current_batch:
        batches.append(current_batch)

    # ------------------------------------------------------------------
    # 4. Run inference per sub-batch, collect scores
    # ------------------------------------------------------------------
    all_scores = [0.0] * len(audio_list)

    pending_batches = list(batches)

    while pending_batches:
        batch = pending_batches.pop(0)
        indices = [b[0] for b in batch]
        waves = [b[1] for b in batch]

        padded = None
        lengths = None
        try:
            lengths = torch.tensor([w.shape[0] for w in waves], dtype=torch.long)
            max_len = lengths.max().item()
            padded = torch.zeros(len(waves), max_len, dtype=torch.float32)
            for i, w in enumerate(waves):
                padded[i, : w.shape[0]] = w

            padded = padded.to(device)
            lengths = lengths.to(device)

            with torch.no_grad():
                _ = predictor(padded, lengths)
                scores = compute_singmos_masked_mean(
                    frame_capture.frame_scores, lengths
                ).cpu().numpy().tolist()
        except RuntimeError as e:
            is_cuda_oom = "out of memory" in str(e).lower()
            del padded, lengths
            if is_cuda_oom and len(batch) > 1:
                split = len(batch) // 2
                print(
                    f"[SingMOS] CUDA OOM on sub-batch size {len(batch)}; "
                    f"retrying as {split}+{len(batch) - split}",
                    file=sys.stderr,
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pending_batches.insert(0, batch[split:])
                pending_batches.insert(0, batch[:split])
                continue
            raise

        for orig_idx, score in zip(indices, scores):
            all_scores[orig_idx] = score

        # Free GPU memory between sub-batches
        del padded, lengths

    return all_scores


# ============================================================
# Audiobox Aesthetics Evaluation
# ============================================================

def load_aes_predictor(device: torch.device = None, ckpt_path: str = None):
    """Load Audiobox Aesthetics predictor model.

    The predictor evaluates audio on four axes:
      - CE (Content Enjoyment)
      - CU (Content Usefulness)
      - PC (Production Complexity)
      - PQ (Production Quality)

    Args:
        device: Torch device (auto-detected if None).
        ckpt_path: Optional local checkpoint path. When None the library
            downloads the checkpoint automatically or uses the HuggingFace hub.

    Returns:
        An AesPredictor instance ready for inference.
    """
    from audiobox_aesthetics.infer import AesPredictor

    predictor = AesPredictor(checkpoint_pth=ckpt_path, data_col="path")
    # Override device if explicitly provided
    if device is not None:
        predictor.device = device
        predictor.model.to(device)
    return predictor


def compute_aes_scores(audio_waveform, sample_rate: int,
                       predictor) -> dict:
    """Compute Audiobox Aesthetics scores for a single audio waveform.

    Args:
        audio_waveform: 1-D numpy array or torch Tensor of audio samples.
        sample_rate: Source sample rate.
        predictor: Loaded AesPredictor.

    Returns:
        Dict with keys 'CE', 'CU', 'PC', 'PQ' (float values).
    """
    if isinstance(audio_waveform, np.ndarray):
        wav_tensor = torch.from_numpy(audio_waveform).float()
    else:
        wav_tensor = audio_waveform.float()

    # Ensure shape is [1, T] (mono, channels-first)
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)

    batch = [{"path": wav_tensor, "sample_rate": sample_rate}]
    results = predictor.forward(batch)
    return results[0]  # dict with CE, CU, PC, PQ


def compute_batch_aes_scores(audio_list, sample_rate: int,
                             predictor, batch_size: int = 16) -> list:
    """Compute Audiobox Aesthetics scores for a list of audio waveforms.

    Processes the list in sub-batches of *batch_size* to control GPU memory.

    Args:
        audio_list: List of 1-D numpy arrays (audio samples).
        sample_rate: Source sample rate.
        predictor: Loaded AesPredictor.
        batch_size: Number of samples per sub-batch.

    Returns:
        List of dicts (each with keys 'CE', 'CU', 'PC', 'PQ'), same length
        and order as *audio_list*.
    """
    if not audio_list:
        return []

    all_results = []
    for start in range(0, len(audio_list), batch_size):
        chunk = audio_list[start : start + batch_size]
        batch = []
        for audio_np in chunk:
            if isinstance(audio_np, np.ndarray):
                wav_tensor = torch.from_numpy(audio_np).float()
            else:
                wav_tensor = audio_np.float()
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            batch.append({"path": wav_tensor, "sample_rate": sample_rate})
        results = predictor.forward(batch)
        all_results.extend(results)

    return all_results
