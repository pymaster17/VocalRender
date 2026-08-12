import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # MUST come before any torch / CUDA-touching import
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
import gradio as gr
import soundfile as sf
import torchaudio
from einops import rearrange

# Add the bundled src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

MODEL_ID = "pymaster/VocalRender"
CKPT_VARIANT = "VocalRender"  # base variant (not Pro)

# ---------------------------------------------------------------------------
# Model loading (module scope, as required by ZeroGPU)
# ---------------------------------------------------------------------------

def _load_model(ckpt_dir: str, device: str = "cuda"):
    """Load the VocalRender model from a local checkpoint directory."""
    from transformers import LlamaTokenizerFast
    from vocalrender.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from vocalrender.model.utils import get_dtype
    from vocalrender.modules.audiovae import AudioVAEV2 as AudioVAEClass

    ckpt_path = Path(ckpt_dir)

    with open(ckpt_path / "config.json") as f:
        config_dict = json.load(f)

    architecture = str(config_dict.get("architecture", "voxcpm")).lower()
    config = VoxCPMConfig(**config_dict)

    tokenizer = LlamaTokenizerFast.from_pretrained(str(ckpt_path))
    if len(tokenizer) != config.lm_config.vocab_size:
        config.lm_config.vocab_size = len(tokenizer)

    audio_vae_config = getattr(config, "audio_vae_config", None)
    audio_vae = AudioVAEClass(config=audio_vae_config) if audio_vae_config else AudioVAEClass()

    vae_pt_path = ckpt_path / "audiovae.pth"
    vae_ckpt = torch.load(str(vae_pt_path), map_location="cpu", weights_only=True)
    vae_sd = vae_ckpt.get("state_dict", vae_ckpt)
    audio_vae.load_state_dict(vae_sd)

    model = VoxCPM2Model(config, tokenizer, audio_vae, lora_config=None)

    from safetensors.torch import load_file as load_safetensors
    sf_path = ckpt_path / "model.safetensors"
    model_sd = load_safetensors(str(sf_path))

    for k, v in vae_sd.items():
        model_sd[f"audio_vae.{k}"] = v
    model.load_state_dict(model_sd, strict=False)

    model = model.to(get_dtype(config.dtype)).to(device).eval()
    model.device = device
    model.audio_vae = model.audio_vae.to(torch.float32)
    return model


def _download_model():
    """Download model weights from HF Hub to a local directory."""
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=[f"{CKPT_VARIANT}/*"],
        repo_type="model",
    )
    return os.path.join(local_dir, CKPT_VARIANT)


print("[VocalRender] Downloading model weights...", file=sys.stderr)
_CKPT_DIR = _download_model()
print(f"[VocalRender] Model downloaded to: {_CKPT_DIR}", file=sys.stderr)

print("[VocalRender] Loading model...", file=sys.stderr)
model = _load_model(_CKPT_DIR, device="cuda")
print("[VocalRender] Model loaded on cuda", file=sys.stderr)


# ---------------------------------------------------------------------------
# SVS prompt building (mirrors infer_vocalrender_svs_single.py)
# ---------------------------------------------------------------------------

_cached_preprocessor = None

def _get_preprocessor(model):
    global _cached_preprocessor
    if _cached_preprocessor is not None:
        return _cached_preprocessor
    from vocalrender.preprocessing import create_lightweight_preprocessor
    _cached_preprocessor = create_lightweight_preprocessor(
        model.text_tokenizer.tokenizer,
    )
    return _cached_preprocessor


def _build_svs_prompt(entry: Dict, model) -> str:
    from vocalrender.preprocessing import rebuild_svs_prompt
    tokenizer = model.text_tokenizer.tokenizer
    preprocessor = _get_preprocessor(model)
    prompt = rebuild_svs_prompt(
        sample=entry,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        force_lyrics_only=False,
    )
    if not prompt:
        raise ValueError("Empty SVS prompt")
    return prompt


def _encode_prompt_audio(wav_path: str, model, max_frames: int = 50):
    """Load audio, encode through VAE, return [T, P, D] latent."""
    from vocalrender.model.utils import get_in_sample_rate

    target_sr = get_in_sample_rate(model)
    patch_size = model.patch_size

    audio_data, sr = sf.read(wav_path)
    wav = torch.from_numpy(audio_data).float()
    if wav.dim() == 2:
        wav = wav.mean(dim=1)

    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav.unsqueeze(0)).squeeze(0)

    wav_input = wav.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    hop = model.audio_vae.hop_length
    wav_len = wav_input.size(-1)
    patch_len = hop * patch_size
    if wav_len % patch_len != 0:
        pad = patch_len - wav_len % patch_len
        wav_input = nn.functional.pad(wav_input, (0, pad))

    wav_input = wav_input.to(model.device)
    with torch.no_grad():
        z = model.audio_vae.encode(wav_input, target_sr)
        feats = z.transpose(1, 2).cpu()

    T_prime = feats.size(1)
    if T_prime % patch_size != 0:
        pad_len = patch_size - T_prime % patch_size
        feats = nn.functional.pad(feats.transpose(1, 2), (0, pad_len)).transpose(1, 2)

    feats = rearrange(feats, "b (t p) c -> b t p c", p=patch_size).squeeze(0)

    if feats.shape[0] > max_frames:
        import random
        start = random.Random(42).randint(0, feats.shape[0] - max_frames)
        feats = feats[start: start + max_frames]

    return feats


# ---------------------------------------------------------------------------
# Gradio inference function
# ---------------------------------------------------------------------------

def _parse_input(lyrics_str, pitches_str, notes_str, pitch2word_str, bpm):
    """Parse text-area inputs into the JSON entry format."""
    words = [w.strip() for w in lyrics_str.split("|") if w.strip()]
    pitches = [int(p.strip()) for p in pitches_str.split(",") if p.strip()]
    notes = [n.strip() for n in notes_str.split(",") if n.strip()]
    pitch2word = [int(p.strip()) for p in pitch2word_str.split(",") if p.strip()]

    # Auto-generate pitch2word if not provided
    if not pitch2word:
        pitch2word = list(range(len(pitches)))

    return {
        "item_name": "user_input",
        "word": words,
        "pitch": pitches,
        "note": notes,
        "pitch2word": pitch2word,
        "bpm": int(bpm),
    }


@spaces.GPU(duration=60)
def generate(
    voice_preset: str,
    prompt_audio,
    lyrics_str: str,
    pitches_str: str,
    notes_str: str,
    pitch2word_str: str,
    bpm: int,
    cfg_value: float,
    inference_timesteps: int,
    temperature: float,
    max_len: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate singing voice from lyrics, MIDI pitches, and a prompt audio clip.

    Args:
        voice_preset: An included example voice, or the option to upload a custom voice.
        prompt_audio: A clean 2-8 second singing audio clip (wav) providing the target timbre.
        lyrics_str: Pipe-separated lyrics syllables (e.g. "我|的|孤|独").
        pitches_str: Comma-separated MIDI pitch numbers (e.g. "65,64,64,65,67,65").
        notes_str: Comma-separated note duration tokens (e.g. "<NOTE_8>,<NOTE_32>").
        pitch2word_str: Comma-separated mapping from pitch index to word index.
        bpm: Beats per minute.
        cfg_value: Classifier-free guidance value.
        inference_timesteps: Number of diffusion inference steps.
        temperature: Sampling temperature.
        max_len: Maximum generation length in patches.
    """
    t0 = time.perf_counter()

    preset_path = VOICE_PRESETS.get(voice_preset)
    if preset_path:
        prompt_audio = preset_path
    elif prompt_audio is None:
        return None, gr.Markdown(
            "❌ Choose one of the included voices, or upload a 2–8 second singing clip."
        ), ""

    entry = _parse_input(lyrics_str, pitches_str, notes_str, pitch2word_str, bpm)

    # Build SVS prompt
    svs_prompt = _build_svs_prompt(entry, model)

    # Encode prompt audio
    prompt_audio_feats = _encode_prompt_audio(prompt_audio, model, max_frames=50)
    if prompt_audio_feats is None or prompt_audio_feats.numel() == 0:
        return None, gr.Markdown("❌ Prompt audio encoding failed."), ""

    # Generate
    gen_kwargs = dict(
        target_texts=[svs_prompt],
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        max_len=max_len,
        verbose=True,
        temperature=temperature,
        fsq_temperature=0.0,
        prompt_audio_feats=[prompt_audio_feats],
    )

    with torch.no_grad():
        audio_tensors = model.generate_batch(**gen_kwargs)

    if not audio_tensors or audio_tensors[0] is None or audio_tensors[0].numel() == 0:
        return None, gr.Markdown("❌ Audio generation failed."), ""

    # Normalize and save
    from vocalrender.evaluation.audio_utils import normalize_audio
    from vocalrender.model.utils import get_out_sample_rate

    sample_rate = get_out_sample_rate(model)
    audio_np = normalize_audio(audio_tensors[0].float().cpu().numpy().flatten())
    duration = len(audio_np) / sample_rate
    elapsed = time.perf_counter() - t0

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_np, sample_rate)
        out_path = tmp.name

    info = f"✅ Generated {duration:.2f}s of audio in {elapsed:.1f}s"
    return out_path, gr.Markdown(info), svs_prompt


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
"""

NOTE_OPTIONS = [
    "<NOTE_1>", "<NOTE_2>", "<NOTE_4>", "<NOTE_8>", "<NOTE_16>", "<NOTE_32>",
    "<NOTE_DOT_1>", "<NOTE_DOT_2>", "<NOTE_DOT_4>", "<NOTE_DOT_8>", "<NOTE_DOT_16>", "<NOTE_DOT_32>",
]

VOICE_PRESETS = {
    "Included voice 1": "assets/2003000081.wav",
    "Included voice 2": "assets/2017000644.wav",
    "Included voice 3": "assets/2044001666.wav",
    "Upload my own voice": None,
}

# Predefined examples from the official repo
EXAMPLES = [
    # Demo example from inference_input.json
    {
        "lyrics": "我|的|孤|独|是|真|的",
        "pitches": "65,64,64,65,67,65,67,69,63",
        "notes": "<NOTE_8>,<NOTE_32>,<NOTE_16>,<NOTE_16>,<NOTE_16>,<NOTE_8>,<NOTE_16>,<NOTE_16>,<NOTE_8>",
        "pitch2word": "0,1,2,2,2,3,4,5,6",
        "bpm": 64,
        "prompt_audio": "assets/2003000081.wav",
        "voice_preset": "Included voice 1",
    },
    # Opencpop demo entry 2003000087 (prompt audio 2003000081)
    {
        "lyrics": "假|如|迷|路|了|一|SP|定|SP|把|思|念|装|进|漂|流|瓶|SP",
        "pitches": "69,70,69,70,69,62,0,62,60,58,0,58,69,70,69,70,69,62,62,60,0",
        "notes": "<NOTE_16>,<NOTE_16>,<NOTE_16>,<NOTE_16>,<NOTE_16>,<NOTE_16>,<NOTE_32>,<NOTE_16>,<NOTE_DOT_32>,<NOTE_DOT_8>,<NOTE_DOT_16>,<NOTE_32>,<NOTE_16>,<NOTE_16>,<NOTE_16>,<NOTE_DOT_32>,<NOTE_DOT_16>,<NOTE_16>,<NOTE_DOT_16>,<NOTE_8>,<NOTE_16>",
        "pitch2word": "0,1,2,3,4,5,6,7,7,7,8,9,10,11,12,13,14,15,16,16,17",
        "bpm": 58,
        "prompt_audio": "assets/2003000081.wav",
        "voice_preset": "Included voice 1",
    },
    # Opencpop demo entry 2017000646 (prompt audio 2017000644)
    {
        "lyrics": "在|一|瞬|间|温|热|了|SP|双|眼",
        "pitches": "58,61,63,61,63,68,70,68,0,61,63,61,63,63",
        "notes": "<NOTE_DOT_8>,<NOTE_DOT_16>,<NOTE_4>,<NOTE_DOT_16>,<NOTE_8>,<NOTE_16>,<NOTE_16>,<NOTE_DOT_8>,<NOTE_16>,<NOTE_8>,<NOTE_16>,<NOTE_16>,<NOTE_DOT_16>,<NOTE_DOT_2>",
        "pitch2word": "0,1,2,3,4,5,5,6,7,8,8,8,8,9",
        "bpm": 70,
        "prompt_audio": "assets/2017000644.wav",
        "voice_preset": "Included voice 2",
    },
]

with gr.Blocks(elem_id="col-container") as demo:
    gr.Markdown(
        "# 🎵 VocalRender Demo — Turn a score into a singing voice\n"
        "VocalRender sings a melody you describe with **lyrics, notes, and tempo**. "
        "A short singing recording supplies the vocal color; it does not need to contain the same song.\n\n"
        "[Model](https://huggingface.co/pymaster/VocalRender) · "
        "[Paper](https://arxiv.org/abs/2607.27768) · "
        "[Code](https://github.com/pymaster17/VocalRender)"
    )

    with gr.Accordion("New here? Read this musician-friendly guide", open=True):
        gr.Markdown(
            "### What you need\n"
            "1. **A voice reference:** choose an included voice, or upload 2–8 seconds of clean, unaccompanied singing.\n"
            "2. **Lyrics:** separate sung syllables with `|`. Use `SP` for a rest or breath.\n"
            "3. **Melody:** enter MIDI note numbers. For reference, middle C is 60; use 0 for a rest.\n"
            "4. **Rhythm and tempo:** choose a duration for each note and enter the song's BPM.\n\n"
            "To hear it immediately, select one of the prepared examples at the bottom and press "
            "**Generate Singing**. The first run may wait in a shared GPU queue.\n\n"
            "> This research demo currently works best with Chinese lyrics. Only upload a voice "
            "recording that you own or have permission to use."
        )

    with gr.Column(elem_id="col-container"):
        with gr.Accordion("Preview the three included voices", open=False):
            with gr.Row():
                for preset_name, preset_path in list(VOICE_PRESETS.items())[:3]:
                    gr.Audio(
                        value=preset_path,
                        label=preset_name,
                        interactive=False,
                    )

        voice_preset = gr.Radio(
            choices=list(VOICE_PRESETS),
            value="Included voice 1",
            label="1. Choose a voice reference",
            info="Use an included singing voice, or choose Upload my own voice below.",
        )

        with gr.Row():
            prompt_audio = gr.Audio(
                label="Optional upload (2–8 seconds of clean singing)",
                type="filepath",
                format="wav",
            )

        with gr.Row():
            lyrics_str = gr.Textbox(
                label="2. Lyrics (separate each sung syllable with |)",
                value="我|的|孤|独|是|真|的",
                info="Example: 我|的|孤|独. Write SP where the singer rests or breathes.",
            )

        with gr.Row():
            pitches_str = gr.Textbox(
                label="3. Melody (MIDI note numbers)",
                value="65,64,64,65,67,65,67,69,63",
                info="One number per note, separated by commas. Middle C is 60; use 0 for a rest.",
            )
            notes_str = gr.Textbox(
                label="4. Rhythm (note durations)",
                value="<NOTE_8>,<NOTE_32>,<NOTE_16>,<NOTE_16>,<NOTE_16>,<NOTE_8>,<NOTE_16>,<NOTE_16>,<NOTE_8>",
                info="Use NOTE_4 for a quarter note, NOTE_8 for an eighth note, and DOT for dotted notes.",
            )

        with gr.Row():
            pitch2word_str = gr.Textbox(
                label="Which lyric syllable belongs to each note? (advanced)",
                value="0,1,2,2,2,3,4,5,6",
                info="Starting from 0, map every note to a lyric syllable. Leave empty when there is exactly one note per syllable.",
            )
            bpm = gr.Number(label="5. Tempo (BPM)", value=64, precision=0)

        with gr.Accordion("Advanced settings", open=False):
            with gr.Row():
                cfg_value = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="CFG value")
                inference_timesteps = gr.Slider(1, 50, value=10, step=1, label="Inference timesteps")
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                max_len = gr.Slider(100, 3000, value=2000, step=100, label="Max length (patches)")

        run_btn = gr.Button("Generate Singing", variant="primary")
        status_out = gr.Markdown("")
        prompt_out = gr.Textbox(label="Generated SVS prompt (debug)", visible=False)
        audio_out = gr.Audio(label="Synthesized Singing", type="filepath", format="wav")

    # Wire the button
    run_btn.click(
        fn=generate,
        inputs=[voice_preset, prompt_audio, lyrics_str, pitches_str, notes_str, pitch2word_str, bpm,
                cfg_value, inference_timesteps, temperature, max_len],
        outputs=[audio_out, status_out, prompt_out],
    )

    # Examples
    gr.Examples(
        examples=[
            [e["voice_preset"], None, e["lyrics"], e["pitches"], e["notes"], e["pitch2word"], e["bpm"]]
            for e in EXAMPLES
        ],
        inputs=[voice_preset, prompt_audio, lyrics_str, pitches_str, notes_str, pitch2word_str, bpm],
        label="Ready-to-use score examples",
    )

demo.launch(mcp_server=True, theme=gr.themes.Citrus(), css=CSS)
