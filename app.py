import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # MUST come before any torch / CUDA-touching import
import sys
import json
import re
import base64
import random
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
    words = _split_lyrics(lyrics_str)
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


def _generate_impl(
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

BRAVURA_FONT_DATA = base64.b64encode(
    (Path(__file__).parent / "assets/fonts/Bravura.woff2").read_bytes()
).decode("ascii")

CSS = """
@font-face {
  font-family: "BravuraVocalRender";
  src: url("data:font/woff2;base64,__BRAVURA_FONT_DATA__") format("woff2");
  font-weight: normal;
  font-style: normal;
}
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
.note-legend { display: flex; flex-wrap: wrap; gap: .45rem; margin: .4rem 0 1rem; }
.note-choice { display: flex; min-width: 74px; flex-direction: column; align-items: center; padding: .3rem; border: 1px solid #ddd; border-radius: .45rem; }
.note-choice small { font-size: .68rem; text-align: center; }
.selected-note { display: flex; align-items: center; gap: .5rem; min-height: 66px; padding: .4rem .6rem; border: 1px solid #ddd; border-radius: .5rem; }
.bravura-note { display: inline-flex; align-items: center; min-width: 46px; height: 48px; font-family: "BravuraVocalRender"; font-size: 3rem; line-height: 1; }
.bravura-dot { margin-left: -.3rem; font-family: "BravuraVocalRender"; font-size: 2.2rem; }
.dark .note-choice, .dark .selected-note { border-color: #444; }
""".replace("__BRAVURA_FONT_DATA__", BRAVURA_FONT_DATA)

NOTE_OPTIONS = [
    "<NOTE_1>", "<NOTE_2>", "<NOTE_4>", "<NOTE_8>", "<NOTE_16>", "<NOTE_32>",
    "<NOTE_DOT_1>", "<NOTE_DOT_2>", "<NOTE_DOT_4>", "<NOTE_DOT_8>", "<NOTE_DOT_16>", "<NOTE_DOT_32>",
]

# Longest to shortest, so the duration editor behaves naturally as a slider.
NOTE_DURATION_OPTIONS = [
    ("Dotted whole", "<NOTE_DOT_1>"),
    ("Whole", "<NOTE_1>"),
    ("Dotted half", "<NOTE_DOT_2>"),
    ("Half", "<NOTE_2>"),
    ("Dotted quarter", "<NOTE_DOT_4>"),
    ("Quarter", "<NOTE_4>"),
    ("Dotted eighth", "<NOTE_DOT_8>"),
    ("Eighth", "<NOTE_8>"),
    ("Dotted sixteenth", "<NOTE_DOT_16>"),
    ("Sixteenth", "<NOTE_16>"),
    ("Dotted thirty-second", "<NOTE_DOT_32>"),
    ("Thirty-second", "<NOTE_32>"),
]
NOTE_TO_DURATION_INDEX = {
    token: index for index, (_, token) in enumerate(NOTE_DURATION_OPTIONS)
}


def _note_icon_html(token: str) -> str:
    """Render a professional notation glyph from the bundled Bravura font."""
    dotted = "_DOT_" in token
    denominator = int(re.search(r"(\d+)>$", token).group(1))
    codepoint = {
        1: "1D15D",
        2: "1D15E",
        4: "1D15F",
        8: "1D160",
        16: "1D161",
        32: "1D162",
    }[denominator]
    dot = '<span class="bravura-dot">&#xE1E7;</span>' if dotted else ""
    return f'<span class="bravura-note" aria-hidden="true">&#x{codepoint};{dot}</span>'


def _duration_html(index: int, compact: bool = False) -> str:
    label, token = NOTE_DURATION_OPTIONS[int(index)]
    icon = _note_icon_html(token)
    if compact:
        return f'<div class="selected-note">{icon}<span>{label}</span></div>'
    return f'<div class="note-choice"><strong>{index}</strong>{icon}<small>{label}</small></div>'


NOTE_DURATION_LEGEND_HTML = (
    '<div class="note-legend">'
    + "".join(_duration_html(index) for index in range(len(NOTE_DURATION_OPTIONS)))
    + "</div>"
)

VOICE_PRESETS = {
    "Alto-1": "assets/alto-1.wav",
    "Alto-2": "assets/alto-2.wav",
    "Alto-3": "assets/alto-3.wav",
    "Tenor-1": "assets/tenor-1.wav",
    "Tenor-2": "assets/tenor-2.wav",
    "Tenor-3": "assets/tenor-3.wav",
    "Upload my own voice": None,
}

SCORE_PRESETS = json.loads(
    (Path(__file__).parent / "assets/score_presets.json").read_text(encoding="utf-8")
)

MAX_SCORE_WORDS = 64
CHINESE_PUNCTUATION = "，。！？、；：‘’“”（）《》〈〉【】…—·,.!?;:()[]-"


def _validate_chinese_lyrics(lyrics: str) -> None:
    """Reject unsupported languages before a ZeroGPU request is made."""
    text = (lyrics or "").strip()
    if not text:
        raise gr.Error("Enter some Chinese lyrics before creating the score editor.")

    # Remove supported structural tokens before checking the remaining text.
    check_text = re.sub(r"(?i)(?<![A-Za-z])SP(?![A-Za-z])", "", text)
    check_text = check_text.replace("|", "")
    check_text = re.sub(r"\s+", "", check_text)
    check_text = check_text.translate(str.maketrans("", "", CHINESE_PUNCTUATION))

    unsupported = sorted({char for char in check_text if not ("\u3400" <= char <= "\u9fff")})
    if unsupported:
        preview = " ".join(unsupported[:8])
        raise gr.Error(
            "VocalRender was trained only on Chinese lyrics. "
            f"Please remove unsupported characters or languages: {preview}"
        )

    if not any("\u3400" <= char <= "\u9fff" for char in check_text):
        raise gr.Error("Please enter Chinese lyrics. This checkpoint does not support other languages.")


def _split_lyrics(lyrics: str) -> List[str]:
    """Validate Chinese lyrics and split them into sung units."""
    lyrics = (lyrics or "").strip()
    _validate_chinese_lyrics(lyrics)
    if "|" in lyrics:
        words = [
            word.strip().strip(CHINESE_PUNCTUATION)
            for word in lyrics.split("|")
            if word.strip().strip(CHINESE_PUNCTUATION)
        ]
    else:
        # Keep SP as a rest token; ignore punctuation when splitting Chinese characters.
        normalized = re.sub(r"(?i)(?<![A-Za-z])SP(?![A-Za-z])", "|SP|", lyrics)
        words = re.findall(r"SP|[\u3400-\u9fff]", normalized, flags=re.IGNORECASE)
    if len(words) > MAX_SCORE_WORDS:
        raise gr.Error(f"Please use at most {MAX_SCORE_WORDS} lyric units per generation.")
    return words


def _score_rows_from_lyrics(lyrics: str) -> List[Dict]:
    words = _split_lyrics(lyrics)
    if not words:
        raise gr.Error("Enter some lyrics before creating the score editor.")
    return [
        {
            "word": word,
            "word_index": word_index,
            "uid": f"{word_index}-0",
            "pitch": 0 if word.upper() == "SP" else 60,
            "duration_index": NOTE_TO_DURATION_INDEX["<NOTE_4>"],
        }
        for word_index, word in enumerate(words)
    ]


def _create_manual_score(lyrics: str):
    return _score_rows_from_lyrics(lyrics), gr.Markdown("")


def _preset_to_rows(preset: Dict) -> List[Dict]:
    """Build editable rows while preserving every melisma note in a preset."""
    words = preset["words"]
    pitches = preset["pitches"]
    notes = preset["notes"]
    mapping = preset["pitch2word"]
    rows = []
    occurrence = {}
    for note_index, word_index in enumerate(mapping):
        occurrence[word_index] = occurrence.get(word_index, 0) + 1
        token = notes[note_index]
        rows.append({
            "word": words[word_index],
            "word_index": word_index,
            "uid": f"{word_index}-{occurrence[word_index] - 1}",
            "pitch": pitches[note_index],
            "duration_index": NOTE_TO_DURATION_INDEX[token],
        })
    return rows


def load_random_preset():
    """Choose a dataset preset and load it for user editing without using a GPU."""
    preset = random.choice(SCORE_PRESETS)
    lyrics = "|".join(preset["words"])
    rows = _preset_to_rows(preset)
    message = gr.Markdown(
        "🎲 **Preset loaded.** Adjust any pitch, duration, melisma note, or tempo before generating."
    )
    return lyrics, rows, preset["bpm"], message


def _sync_score_rows(rows: List[Dict], pitches, durations) -> List[Dict]:
    synced = [dict(row) for row in rows]
    for row, pitch, duration in zip(synced, pitches, durations):
        row["pitch"] = int(pitch)
        row["duration_index"] = int(round(duration))
    return synced


def _add_melisma_note(rows: List[Dict], row_index: int, pitches, durations) -> List[Dict]:
    rows = _sync_score_rows(rows, pitches, durations)
    source = rows[row_index]
    insert_at = max(
        index for index, row in enumerate(rows)
        if row["word_index"] == source["word_index"]
    ) + 1
    existing_ids = {row["uid"] for row in rows}
    suffix = 1
    while f"{source['word_index']}-{suffix}" in existing_ids:
        suffix += 1
    rows.insert(insert_at, {
        **source,
        "uid": f"{source['word_index']}-{suffix}",
    })
    return rows


def _delete_melisma_note(rows: List[Dict], row_index: int, pitches, durations) -> List[Dict]:
    rows = _sync_score_rows(rows, pitches, durations)
    word_index = rows[row_index]["word_index"]
    if sum(row["word_index"] == word_index for row in rows) <= 1:
        raise gr.Error("Each lyric unit must keep at least one note.")
    return rows[:row_index] + rows[row_index + 1:]


@spaces.GPU(duration=60)
def generate_from_word_score(
    rows,
    voice,
    uploaded_audio,
    tempo,
    cfg,
    steps,
    temperature,
    max_len,
    *score_values,
):
    """Top-level ZeroGPU endpoint for the dynamic word-by-word score editor."""
    row_count = len(rows)
    pitches = [int(value) for value in score_values[:row_count]]
    duration_indices = [int(round(value)) for value in score_values[row_count:]]
    notes = [NOTE_DURATION_OPTIONS[index][1] for index in duration_indices]
    words_by_index = {}
    for row in rows:
        words_by_index[row["word_index"]] = row["word"]
    words = [words_by_index[index] for index in sorted(words_by_index)]
    pitch2word = [row["word_index"] for row in rows]
    return _generate_impl(
        voice,
        uploaded_audio,
        "|".join(words),
        ",".join(map(str, pitches)),
        ",".join(notes),
        ",".join(map(str, pitch2word)),
        tempo,
        cfg,
        steps,
        temperature,
        max_len,
    )

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
            "2. **Lyrics:** enter Chinese lyrics. They are split character by character; "
            "you can also use `|` to control the split. Other languages are not supported by this checkpoint.\n"
            "3. **Melody and rhythm:** press **Create word-by-word score**, then set pitch and duration. "
            "Use **+ Melisma note** when one lyric unit spans multiple notes.\n"
            "4. **Generate:** choose the tempo and press **Generate Singing**. Or press "
            "**🎲 Random score preset** to load a ready-made score, adjust it freely, then generate. "
            "The first run may wait in a shared GPU queue.\n\n"
            "> This checkpoint supports Chinese lyrics only. Other languages are rejected before inference. Only upload a voice "
            "recording that you own or have permission to use."
        )

    with gr.Column(elem_id="col-container"):
        with gr.Accordion("Preview the six included voices", open=False):
            gr.Markdown("**Alto references**")
            with gr.Row():
                for preset_name in ("Alto-1", "Alto-2", "Alto-3"):
                    gr.Audio(
                        value=VOICE_PRESETS[preset_name],
                        label=preset_name,
                        interactive=False,
                    )
            gr.Markdown("**Tenor references**")
            with gr.Row():
                for preset_name in ("Tenor-1", "Tenor-2", "Tenor-3"):
                    gr.Audio(
                        value=VOICE_PRESETS[preset_name],
                        label=preset_name,
                        interactive=False,
                    )

        voice_preset = gr.Radio(
            choices=list(VOICE_PRESETS),
            value="Alto-1",
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
                label="2. Enter lyrics",
                value="",
                placeholder="Enter Chinese lyrics, or click 🎲 Random score preset",
                info="Type normally, or use | to choose the exact split. Write SP for a rest or breath.",
                scale=5,
            )
            random_preset_btn = gr.Button(
                "🎲 Random score preset",
                variant="secondary",
                scale=1,
            )

        preset_info = gr.Markdown("")

        split_btn = gr.Button("Create word-by-word score")
        score_rows = gr.State([])

        @gr.render(inputs=score_rows)
        def render_word_score(rows):
            if not rows:
                return

            gr.Markdown(
                "### 3. Set pitch and duration for each lyric unit\n"
                "Type a MIDI pitch directly (60 = middle C/C4; 0 = rest). "
                "Choose note length with the duration slider."
            )
            gr.HTML(NOTE_DURATION_LEGEND_HTML)
            pitch_controls = []
            duration_controls = []
            add_buttons = []
            delete_buttons = []
            word_note_counts = {
                word_index: sum(row["word_index"] == word_index for row in rows)
                for word_index in {row["word_index"] for row in rows}
            }
            for index, row in enumerate(rows):
                uid = row["uid"]
                with gr.Row(key=f"score-row-{uid}"):
                    gr.Textbox(
                        value=row["word"],
                        label=f"Lyric {row['word_index'] + 1}",
                        interactive=False,
                        scale=1,
                        key=f"score-word-{uid}",
                    )
                    pitch = gr.Number(
                        minimum=0,
                        maximum=127,
                        value=row["pitch"],
                        precision=0,
                        label="MIDI pitch",
                        interactive=True,
                        scale=2,
                        key=f"score-pitch-{uid}",
                    )
                    duration = gr.Slider(
                        minimum=0,
                        maximum=len(NOTE_DURATION_OPTIONS) - 1,
                        value=row["duration_index"],
                        step=1,
                        label="Note duration",
                        interactive=True,
                        scale=3,
                        key=f"score-duration-{uid}",
                    )
                    duration_name = gr.HTML(
                        value=_duration_html(row["duration_index"], compact=True),
                        scale=2,
                        key=f"score-duration-name-{uid}",
                    )
                    duration.change(
                        lambda value: _duration_html(int(round(value)), compact=True),
                        inputs=duration,
                        outputs=duration_name,
                        key=f"show-duration-{uid}",
                    )
                    add_button = gr.Button(
                        "+ Melisma note",
                        scale=1,
                        key=f"score-add-{uid}",
                    )
                    delete_button = gr.Button(
                        "Remove note",
                        variant="stop",
                        visible=word_note_counts[row["word_index"]] > 1,
                        scale=1,
                        key=f"score-delete-{uid}",
                    )
                    pitch_controls.append(pitch)
                    duration_controls.append(duration)
                    add_buttons.append(add_button)
                    delete_buttons.append(delete_button)

            score_editor_inputs = [score_rows, *pitch_controls, *duration_controls]
            for index, (add_button, delete_button) in enumerate(zip(add_buttons, delete_buttons)):
                def add_note(current_rows, *values, row_index=index):
                    count = len(current_rows)
                    return _add_melisma_note(
                        current_rows, row_index, values[:count], values[count:]
                    )

                def delete_note(current_rows, *values, row_index=index):
                    count = len(current_rows)
                    return _delete_melisma_note(
                        current_rows, row_index, values[:count], values[count:]
                    )

                add_button.click(
                    add_note,
                    inputs=score_editor_inputs,
                    outputs=score_rows,
                    key=f"add-melisma-{rows[index]['uid']}",
                )
                delete_button.click(
                    delete_note,
                    inputs=score_editor_inputs,
                    outputs=score_rows,
                    key=f"delete-melisma-{rows[index]['uid']}",
                )

            easy_inputs = [
                score_rows,
                voice_preset,
                prompt_audio,
                bpm,
                cfg_value,
                inference_timesteps,
                temperature,
                max_len,
                *pitch_controls,
                *duration_controls,
            ]

            easy_run_btn.click(
                fn=generate_from_word_score,
                inputs=easy_inputs,
                outputs=[audio_out, status_out, prompt_out],
                key="generate-word-score",
            )

        with gr.Row():
            bpm = gr.Number(label="4. Tempo (BPM)", value=64, precision=0)

        with gr.Accordion("Advanced settings", open=False):
            with gr.Row():
                cfg_value = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="CFG value")
                inference_timesteps = gr.Slider(1, 50, value=10, step=1, label="Inference timesteps")
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                max_len = gr.Slider(100, 3000, value=2000, step=100, label="Max length (patches)")

        easy_run_btn = gr.Button("Generate Singing", variant="primary")

        split_btn.click(
            _create_manual_score,
            inputs=lyrics_str,
            outputs=[score_rows, preset_info],
        )
        lyrics_str.submit(
            _create_manual_score,
            inputs=lyrics_str,
            outputs=[score_rows, preset_info],
        )

        status_out = gr.Markdown("")
        prompt_out = gr.Textbox(label="Generated SVS prompt (debug)", visible=False)
        audio_out = gr.Audio(label="Synthesized Singing", type="filepath", format="wav")

    random_preset_btn.click(
        fn=load_random_preset,
        inputs=None,
        outputs=[
            lyrics_str,
            score_rows,
            bpm,
            preset_info,
        ],
    )

demo.launch(mcp_server=True, theme=gr.themes.Citrus(), css=CSS)
