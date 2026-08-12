import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # MUST come before any torch / CUDA-touching import
import sys
import json
import re
import base64
import hashlib
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

from vocalrender.utils.score_import import (
    ScoreImportError,
    convert_selection as convert_imported_selection,
    get_part as get_imported_part,
    parse_score as parse_imported_score_data,
    part_summary as imported_part_summary,
    recommended_measure_range,
)

MODEL_ID = "pymaster/VocalRender"
CKPT_VARIANTS = ("VocalRender-Pro", "VocalRender")
DEFAULT_CKPT_VARIANT = "VocalRender-Pro"

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


def _download_models():
    """Download both selectable model checkpoints from HF Hub."""
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=[f"{variant}/*" for variant in CKPT_VARIANTS],
        repo_type="model",
    )
    return {
        variant: os.path.join(local_dir, variant)
        for variant in CKPT_VARIANTS
    }


print("[VocalRender] Downloading model checkpoints...", file=sys.stderr)
_CKPT_DIRS = _download_models()
print(f"[VocalRender] Models downloaded to: {_CKPT_DIRS}", file=sys.stderr)

models = {}
for _variant in CKPT_VARIANTS:
    print(f"[VocalRender] Loading {_variant}...", file=sys.stderr)
    models[_variant] = _load_model(_CKPT_DIRS[_variant], device="cuda")
    print(f"[VocalRender] {_variant} loaded on cuda", file=sys.stderr)


# ---------------------------------------------------------------------------
# SVS prompt building (mirrors infer_vocalrender_svs_single.py)
# ---------------------------------------------------------------------------

_cached_preprocessors = {}

def _get_preprocessor(model):
    cache_key = id(model)
    if cache_key in _cached_preprocessors:
        return _cached_preprocessors[cache_key]
    from vocalrender.preprocessing import create_lightweight_preprocessor
    preprocessor = create_lightweight_preprocessor(
        model.text_tokenizer.tokenizer,
    )
    _cached_preprocessors[cache_key] = preprocessor
    return preprocessor


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

    bpm = int(bpm)
    if not 1 <= bpm <= 255:
        raise gr.Error("Tempo must be between 1 and 255 BPM.")
    if not pitches or len(pitches) != len(notes) or len(pitches) != len(pitch2word):
        raise gr.Error("Every score event must have one pitch, duration, and lyric mapping.")
    if any(not 0 <= pitch <= 127 for pitch in pitches):
        raise gr.Error("MIDI pitches must be between 0 and 127.")
    if any(word_index < 0 or word_index >= len(words) for word_index in pitch2word):
        raise gr.Error("The score contains an invalid lyric-to-note mapping.")

    return {
        "item_name": "user_input",
        "word": words,
        "pitch": pitches,
        "note": notes,
        "pitch2word": pitch2word,
        "bpm": bpm,
    }


def _generate_impl(
    checkpoint: str,
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
        checkpoint: Selected VocalRender checkpoint variant.
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

    model = models.get(checkpoint)
    if model is None:
        raise gr.Error(f"Unknown model checkpoint: {checkpoint}")

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
    score_key = hashlib.sha1("|".join(words).encode("utf-8")).hexdigest()[:10]
    return [
        {
            "word": word,
            "word_index": word_index,
            "uid": f"manual-{score_key}-{word_index}-0",
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
            "uid": f"preset-{preset['id']}-{word_index}-{occurrence[word_index] - 1}",
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


def _sync_score_rows(rows: List[Dict], words, pitches, durations) -> List[Dict]:
    synced = [dict(row) for row in rows]
    words_by_index = {}
    for row, word in zip(synced, words):
        words_by_index.setdefault(row["word_index"], str(word).strip())
    for row, pitch, duration in zip(synced, pitches, durations):
        row["word"] = words_by_index[row["word_index"]]
        row["pitch"] = int(pitch)
        row["duration_index"] = int(round(duration))
    return synced


def _add_melisma_note(rows: List[Dict], row_index: int, words, pitches, durations) -> List[Dict]:
    rows = _sync_score_rows(rows, words, pitches, durations)
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


def _delete_melisma_note(rows: List[Dict], row_index: int, words, pitches, durations) -> List[Dict]:
    rows = _sync_score_rows(rows, words, pitches, durations)
    word_index = rows[row_index]["word_index"]
    if sum(row["word_index"] == word_index for row in rows) <= 1:
        raise gr.Error("Each lyric unit must keep at least one note.")
    return rows[:row_index] + rows[row_index + 1:]


@spaces.GPU(duration=60)
def generate_from_word_score(
    rows,
    checkpoint,
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
    words_input = [str(value).strip() for value in score_values[:row_count]]
    pitches = [int(value) for value in score_values[row_count:row_count * 2]]
    duration_indices = [int(round(value)) for value in score_values[row_count * 2:]]
    notes = [NOTE_DURATION_OPTIONS[index][1] for index in duration_indices]
    words_by_index = {}
    for row, word in zip(rows, words_input):
        words_by_index.setdefault(row["word_index"], word)
    words = [words_by_index[index] for index in sorted(words_by_index)]
    pitch2word = [row["word_index"] for row in rows]
    return _generate_impl(
        checkpoint,
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


def _import_part_updates(parsed: Dict, part_key: str):
    """Build Gradio updates for a selected imported part."""
    try:
        part = get_imported_part(parsed, part_key)
    except (ScoreImportError, TypeError) as exc:
        raise gr.Error(str(exc)) from exc
    verse_choices = [
        (f"Embedded lyric line {verse_id}", verse_id)
        for verse_id in part["verses"]
    ]
    verse_choices.append(("Use lyrics textbox", "__external__"))
    verse_value = part["verses"][0] if part["verses"] else "__external__"
    measure_choices = [(f"Measure {measure}", measure) for measure in part["measures"]]
    recommended_start, recommended_end = recommended_measure_range(part)
    recommended_start = recommended_start or part["measures"][0]
    recommended_end = recommended_end or part["measures"][0]
    summary = imported_part_summary(part)
    if parsed.get("warnings"):
        summary += "\n\n**Before loading**\n" + "\n".join(
            f"- {warning}" for warning in parsed["warnings"]
        )
    return (
        gr.Dropdown(choices=verse_choices, value=verse_value),
        gr.Dropdown(choices=measure_choices, value=recommended_start),
        gr.Dropdown(choices=measure_choices, value=recommended_end),
        summary,
    )


def parse_score_for_editor(score_file, abc_text):
    """Parse an upload/paste without requesting a GPU."""
    try:
        parsed = parse_imported_score_data(file_path=score_file, abc_text=abc_text)
        part_choices = [(part["label"], part["key"]) for part in parsed["parts"]]
        part_key = parsed["default_part"]
        verse_update, start_update, end_update, summary = _import_part_updates(parsed, part_key)
        return (
            parsed,
            gr.Dropdown(choices=part_choices, value=part_key),
            verse_update,
            start_update,
            end_update,
            "✅ Score parsed. " + summary,
        )
    except ScoreImportError as exc:
        raise gr.Error(str(exc)) from exc


def change_imported_part(parsed, part_key):
    if not parsed:
        raise gr.Error("Parse a score first.")
    return _import_part_updates(parsed, part_key)


def load_imported_score(parsed, part_key, verse_id, start_measure, end_measure, lyrics):
    """Convert the selected range into the existing editable score rows."""
    if not parsed:
        raise gr.Error("Parse a score first.")
    try:
        result = convert_imported_selection(
            parsed,
            part_key=part_key,
            verse_id=verse_id,
            start_measure=str(start_measure),
            end_measure=str(end_measure),
            external_lyrics=lyrics,
        )
    except ScoreImportError as exc:
        raise gr.Error(str(exc)) from exc
    message = f"✅ **{result['summary']}**"
    if result["warnings"]:
        message += "\n\n**Import notes**\n" + "\n".join(
            f"- {warning}" for warning in result["warnings"]
        )
    return result["lyrics"], result["rows"], result["bpm"], message

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
            "you can also use `|` to control the split, or import ABC/MusicXML below. "
            "Other languages are not supported by this checkpoint.\n"
            "3. **Melody and rhythm:** import a score, or press **Create word-by-word score**, then set pitch and duration. "
            "Use **+ Melisma note** when one lyric unit spans multiple notes.\n"
            "4. **Generate:** choose the tempo and press **Generate Singing**. Or press "
            "**🎲 Random score preset** to load a ready-made score, adjust it freely, then generate. "
            "The first run may wait in a shared GPU queue.\n\n"
            "> This checkpoint supports Chinese lyrics only. Other languages are rejected before inference. Only upload a voice "
            "recording that you own or have permission to use."
        )

    with gr.Column(elem_id="col-container"):
        checkpoint = gr.Radio(
            choices=list(CKPT_VARIANTS),
            value=DEFAULT_CKPT_VARIANT,
            label="Model checkpoint",
            info="VocalRender-Pro is selected by default; switch to VocalRender to compare the base checkpoint.",
            interactive=True,
        )

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
            interactive=True,
        )

        with gr.Row():
            prompt_audio = gr.Audio(
                label="Optional upload (2–8 seconds of clean singing)",
                type="filepath",
                format="wav",
                interactive=True,
            )

        with gr.Accordion("Import ABC notation or MusicXML", open=False):
            gr.Markdown(
                "### Import a score in three steps\n"
                "1. Paste ABC notation or upload `.abc`, `.txt`, `.musicxml`, `.xml`, or `.mxl`, then press **Parse score**.\n"
                "2. Check the suggested vocal part, lyric line, and preselected model-ready measure range.\n"
                "3. Press **Load selected range into editor**, review the notes, then generate.\n\n"
                "ABC `w:` lyrics and MusicXML lyrics are aligned automatically. ABC `W:` words are only page text, "
                "so paste the matching Chinese passage into **Enter lyrics** and choose **Use lyrics textbox**. "
                "Accompaniment chord names such as `\"C\"` are ignored; actual simultaneous notes must be removed "
                "or imported from a melody-only part."
            )
            with gr.Row():
                abc_score_text = gr.Textbox(
                    label="Paste ABC notation",
                    lines=8,
                    placeholder="X:1\nT:My song\nM:4/4\nL:1/4\nQ:1/4=90\nK:C\nC D E F |\nw: 我 爱 唱 歌",
                    scale=3,
                )
                score_file = gr.File(
                    label="Or upload a score file",
                    file_types=[".abc", ".txt", ".musicxml", ".xml", ".mxl"],
                    type="filepath",
                    scale=2,
                )
            parse_score_btn = gr.Button("Parse score", variant="secondary")
            imported_score_state = gr.State(None)
            with gr.Row():
                imported_part = gr.Dropdown(label="Work / vocal part", choices=[])
                imported_verse = gr.Dropdown(label="Lyric line", choices=[])
            with gr.Row():
                imported_start_measure = gr.Dropdown(label="Start measure", choices=[])
                imported_end_measure = gr.Dropdown(label="End measure", choices=[])
            imported_score_info = gr.Markdown("")
            load_imported_btn = gr.Button("Load selected range into editor", variant="primary")

        with gr.Row():
            lyrics_str = gr.Textbox(
                label="2. Enter lyrics",
                value="",
                placeholder="Enter Chinese lyrics, import a score, or click 🎲 Random score preset",
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
            word_controls = []
            add_buttons = []
            delete_buttons = []
            word_note_counts = {
                word_index: sum(row["word_index"] == word_index for row in rows)
                for word_index in {row["word_index"] for row in rows}
            }
            for index, row in enumerate(rows):
                uid = row["uid"]
                with gr.Row(key=f"score-row-{uid}"):
                    word_control = gr.Textbox(
                        value=row["word"],
                        label=f"Lyric {row['word_index'] + 1}",
                        interactive=not any(
                            previous["word_index"] == row["word_index"]
                            for previous in rows[:index]
                        ),
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
                    word_controls.append(word_control)
                    add_buttons.append(add_button)
                    delete_buttons.append(delete_button)

            score_editor_inputs = [score_rows, *word_controls, *pitch_controls, *duration_controls]
            for index, (add_button, delete_button) in enumerate(zip(add_buttons, delete_buttons)):
                def add_note(current_rows, *values, row_index=index):
                    count = len(current_rows)
                    return _add_melisma_note(
                        current_rows,
                        row_index,
                        values[:count],
                        values[count:count * 2],
                        values[count * 2:],
                    )

                def delete_note(current_rows, *values, row_index=index):
                    count = len(current_rows)
                    return _delete_melisma_note(
                        current_rows,
                        row_index,
                        values[:count],
                        values[count:count * 2],
                        values[count * 2:],
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
                checkpoint,
                voice_preset,
                prompt_audio,
                bpm,
                cfg_value,
                inference_timesteps,
                temperature,
                max_len,
                *word_controls,
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

        parse_score_btn.click(
            fn=parse_score_for_editor,
            inputs=[score_file, abc_score_text],
            outputs=[
                imported_score_state,
                imported_part,
                imported_verse,
                imported_start_measure,
                imported_end_measure,
                imported_score_info,
            ],
        )
        imported_part.input(
            fn=change_imported_part,
            inputs=[imported_score_state, imported_part],
            outputs=[
                imported_verse,
                imported_start_measure,
                imported_end_measure,
                imported_score_info,
            ],
        )
        load_imported_btn.click(
            fn=load_imported_score,
            inputs=[
                imported_score_state,
                imported_part,
                imported_verse,
                imported_start_measure,
                imported_end_measure,
                lyrics_str,
            ],
            outputs=[lyrics_str, score_rows, bpm, preset_info],
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
