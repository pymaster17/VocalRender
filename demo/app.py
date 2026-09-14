import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Deployment switches (see README "Deployment"):
#   VOCALRENDER_UI_ONLY=1      run the interface without models (no GPU, no torch)
#   VOCALRENDER_CKPT_DIR=path  load checkpoints from a local directory instead of HF Hub
#   VOCALRENDER_CKPTS=a,b      which checkpoint variants to load (default: all)
#   VOCALRENDER_DEVICE=cuda    torch device for inference (default: cuda)
UI_ONLY = os.environ.get("VOCALRENDER_UI_ONLY") == "1"


class _NoZeroGPU:
    """Stand-in for the `spaces` package outside Hugging Face ZeroGPU Spaces."""

    @staticmethod
    def GPU(*_args, **_kwargs):
        return lambda fn: fn


if UI_ONLY:
    spaces = _NoZeroGPU()
else:
    try:
        import spaces  # MUST come before any torch / CUDA-touching import (ZeroGPU)
    except ImportError:  # self-hosted server: plain GPU, no ZeroGPU scheduling
        spaces = _NoZeroGPU()

import sys
import gc
import json
import re
import threading
import base64
import hashlib
import random
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import gradio as gr

if not UI_ONLY:
    import torch
    import torch.nn as nn
    import numpy as np
    import soundfile as sf
    import torchaudio
    from einops import rearrange

# Locate the vocalrender package when it is not pip-installed:
#   demo/app.py inside the GitHub repo  -> ../src
#   app.py at the root of the HF Space -> ./src (copied there by scripts/sync_space.py)
for _src in (Path(__file__).resolve().parent.parent / "src", Path(__file__).resolve().parent / "src"):
    if (_src / "vocalrender").is_dir():
        sys.path.insert(0, str(_src))
        break

from vocalrender.utils.score_import import (
    ScoreImportError,
    convert_selection as convert_imported_selection,
    get_part as get_imported_part,
    parse_score as parse_imported_score_data,
    part_summary as imported_part_summary,
    recommended_measure_range,
)

MODEL_ID = "pymaster/VocalRender"
ALL_CKPT_VARIANTS = ("VocalRender-Pro", "VocalRender")
CKPT_VARIANTS = tuple(
    variant.strip()
    for variant in os.environ.get("VOCALRENDER_CKPTS", ",".join(ALL_CKPT_VARIANTS)).split(",")
    if variant.strip()
)
_unknown_variants = set(CKPT_VARIANTS) - set(ALL_CKPT_VARIANTS)
if _unknown_variants or not CKPT_VARIANTS:
    raise SystemExit(
        f"VOCALRENDER_CKPTS must be a comma-separated subset of {ALL_CKPT_VARIANTS}, "
        f"got {os.environ.get('VOCALRENDER_CKPTS')!r}"
    )
DEFAULT_CKPT_VARIANT = "VocalRender-Pro" if "VocalRender-Pro" in CKPT_VARIANTS else CKPT_VARIANTS[0]
LOCAL_CKPT_DIR = os.environ.get("VOCALRENDER_CKPT_DIR")
DEVICE = os.environ.get("VOCALRENDER_DEVICE", "cuda")

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


def _resolve_checkpoint_dirs():
    """Locate the selected checkpoint variants: a local directory or HF Hub."""
    if LOCAL_CKPT_DIR:
        base = Path(LOCAL_CKPT_DIR)
        dirs = {variant: base / variant for variant in CKPT_VARIANTS}
        missing = [
            str(path) for path in dirs.values()
            if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file()
        ]
        if missing:
            raise SystemExit(
                "VOCALRENDER_CKPT_DIR must contain one sub-directory per checkpoint variant "
                f"(with config.json and model.safetensors); missing: {missing}"
            )
        return {variant: str(path) for variant, path in dirs.items()}

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


class _ModelCache:
    """Keeps at most one checkpoint variant resident; switching frees the previous one."""

    def __init__(self):
        self.variant = None
        self.model = None
        self.preprocessor = None
        self._lock = threading.Lock()

    def get(self, variant: str):
        if variant not in CKPT_VARIANTS:
            raise gr.Error(f"Unknown model checkpoint: {variant}")
        with self._lock:
            if self.variant == variant and self.model is not None:
                return self.model
            self._unload()
            print(f"[VocalRender] Loading {variant}...", file=sys.stderr)
            t0 = time.perf_counter()
            self.model = _load_model(_CKPT_DIRS[variant], device=DEVICE)
            self.variant = variant
            print(
                f"[VocalRender] {variant} loaded on {DEVICE} in {time.perf_counter() - t0:.1f}s",
                file=sys.stderr,
            )
            return self.model

    def _unload(self):
        if self.model is None:
            return
        print(f"[VocalRender] Unloading {self.variant}...", file=sys.stderr)
        self.model = None
        self.preprocessor = None
        self.variant = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_preprocessor(self, model):
        if self.preprocessor is None:
            from vocalrender.preprocessing import create_lightweight_preprocessor
            self.preprocessor = create_lightweight_preprocessor(model.text_tokenizer.tokenizer)
        return self.preprocessor


models = _ModelCache()
if UI_ONLY:
    print("[VocalRender] UI-only mode: models are not loaded.", file=sys.stderr)
else:
    print("[VocalRender] Resolving model checkpoints...", file=sys.stderr)
    _CKPT_DIRS = _resolve_checkpoint_dirs()
    print(f"[VocalRender] Checkpoints: {_CKPT_DIRS}", file=sys.stderr)
    # Only the default variant is loaded at startup; the other one is loaded on
    # demand when selected in the UI, replacing the resident model.
    models.get(DEFAULT_CKPT_VARIANT)


# ---------------------------------------------------------------------------
# SVS prompt building (mirrors infer_vocalrender_svs_single.py)
# ---------------------------------------------------------------------------

def _get_preprocessor(model):
    return models.get_preprocessor(model)


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

    if checkpoint not in CKPT_VARIANTS:
        raise gr.Error(f"Unknown model checkpoint: {checkpoint}")

    preset_path = VOICE_PRESETS.get(voice_preset)
    if preset_path:
        prompt_audio = preset_path
    elif prompt_audio is None:
        return None, gr.Markdown(
            "❌ 请选择内置音色或上传 2–8 秒歌声 · Choose an included voice, or upload a 2–8 second singing clip."
        ), ""

    entry = _parse_input(lyrics_str, pitches_str, notes_str, pitch2word_str, bpm)

    if UI_ONLY:
        # Development stub: echo the reference clip so the full UI flow can be exercised.
        prompt_preview = " ".join(
            f"{entry['word'][w]}:{p}:{n}" for p, n, w in zip(entry["pitch"], entry["note"], entry["pitch2word"])
        )
        return prompt_audio, gr.Markdown(
            f"🧪 UI-only mode: returned the reference clip instead of running the model ({len(entry['pitch'])} events, {bpm} BPM)."
        ), prompt_preview

    model = models.get(checkpoint)

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

ASSETS_DIR = Path(__file__).parent / "assets"

BRAVURA_FONT_DATA = base64.b64encode(
    (ASSETS_DIR / "fonts/Bravura.woff2").read_bytes()
).decode("ascii")

# The piano roll is a single gr.HTML component: a static shell (html_template),
# scoped styles (css_template) and the editor logic (js_on_load). The score
# model is concatenated in front of the widget so both share one namespace.
PIANO_ROLL_HTML = (ASSETS_DIR / "piano_roll/piano_roll.html").read_text(encoding="utf-8")
PIANO_ROLL_CSS = (ASSETS_DIR / "piano_roll/piano_roll.css").read_text(encoding="utf-8")
PIANO_ROLL_JS = (
    (ASSETS_DIR / "piano_roll/score_model.js").read_text(encoding="utf-8")
    + "\n"
    + (ASSETS_DIR / "piano_roll/piano_roll.js").read_text(encoding="utf-8")
)

CSS = """
@font-face {
  font-family: "BravuraVocalRender";
  src: url("data:font/woff2;base64,__BRAVURA_FONT_DATA__") format("woff2");
  font-weight: normal;
  font-style: normal;
}
.gradio-container { max-width: 1480px !important; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
/* Bravura reserves 2.012em above and below the baseline for ledger lines, so a
   line box puts the baseline at its exact centre whatever the line-height. The
   metronome note glyphs span -0.141em..+0.688em around it, which is 0.27em top
   heavy, so translate by that to centre the ink rather than the baseline. The
   dot carries no side bearing; letter-spacing supplies the gap. */
.bravura-note {
  display: inline-flex; align-items: center; justify-content: center;
  font-family: "BravuraVocalRender"; line-height: 1; letter-spacing: .06em;
  transform: translateY(.27em);
}
#vr-header h1 { margin-bottom: .1rem; }
#vr-header p { margin: 0; }
#vr-sidebar { min-width: 235px; }
#vr-sidebar .block { padding: .55rem .7rem; }
#vr-track-title { font-weight: 600; }
#vr-lyrics-row { align-items: flex-end; }
#vr-generate { font-size: 1.05rem; min-height: 46px; }
#vr-heading { align-items: center; }
.gradio-container { --layout-gap: 8px; }
#vr-heading .wrap { flex-direction: row; }
#vr-heading label { margin: 0; }
#vr-editor-title h3 { margin: 0; }
#vr-links { margin-top: -4px; }
#vr-example-note { font-size: 13px; }
#vr-result { min-height: 100px; }
#vr-language { margin-left: auto; }
#vr-intro p, #vr-links p, #vr-example-note p { margin: 0; }
#vr-links { font-size: 12px; }
#vr-generation { align-items: center; gap: 12px; }
#vr-generation p { font-size: 12px; margin: 0; }
#vr-reference { min-height: 120px; }
#vr-reference .waveform-container { max-height: 60px; }
#vr-alignment:empty, #vr-preset-info:empty { display: none; }
#vr-sidebar h3 { margin: 0; }
@media (max-width: 700px) {
  #vr-heading { flex-direction: column; align-items: stretch; }
  #vr-header { width: 100%; }
  #vr-language { margin-left: 0; width: 100%; }
}
#piano-roll { padding: 0; }
#piano-roll > .html-container, #piano-roll .prose { padding: 0; }
""".replace("__BRAVURA_FONT_DATA__", BRAVURA_FONT_DATA)

# Longest to shortest; the index is the duration_index stored in score rows and
# must stay aligned with DURATIONS in assets/piano_roll/score_model.js.
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

VOICE_PRESETS = {
    "Alto-1": str(ASSETS_DIR / "alto-1.wav"),
    "Alto-2": str(ASSETS_DIR / "alto-2.wav"),
    "Alto-3": str(ASSETS_DIR / "alto-3.wav"),
    "Tenor-1": str(ASSETS_DIR / "tenor-1.wav"),
    "Tenor-2": str(ASSETS_DIR / "tenor-2.wav"),
    "Tenor-3": str(ASSETS_DIR / "tenor-3.wav"),
    "Upload my own voice": None,
}
UPLOAD_VOICE = "Upload my own voice"
VOICE_CHOICES = [
    ("Alto-1 · 女中音 1", "Alto-1"),
    ("Alto-2 · 女中音 2", "Alto-2"),
    ("Alto-3 · 女中音 3", "Alto-3"),
    ("Tenor-1 · 男高音 1", "Tenor-1"),
    ("Tenor-2 · 男高音 2", "Tenor-2"),
    ("Tenor-3 · 男高音 3", "Tenor-3"),
    ("上传我的音色 · Upload my own voice", UPLOAD_VOICE),
]

SCORE_PRESETS = json.loads(
    (ASSETS_DIR / "score_presets.json").read_text(encoding="utf-8")
)

MAX_SCORE_WORDS = 64
MAX_SCORE_EVENTS = 128
DEFAULT_BPM = 64
CHINESE_PUNCTUATION = "，。！？、；：‘’“”（）《》〈〉【】…—·,.!?;:()[]-"


def _validate_chinese_lyrics(lyrics: str) -> None:
    """Reject unsupported languages before a ZeroGPU request is made."""
    text = (lyrics or "").strip()
    if not text:
        raise gr.Error("请先输入中文歌词 · Enter some Chinese lyrics first.")

    # Remove supported structural tokens before checking the remaining text.
    check_text = re.sub(r"(?i)(?<![A-Za-z])SP(?![A-Za-z])", "", text)
    check_text = check_text.replace("|", "")
    check_text = re.sub(r"\s+", "", check_text)
    check_text = check_text.translate(str.maketrans("", "", CHINESE_PUNCTUATION))

    unsupported = sorted({char for char in check_text if not ("㐀" <= char <= "鿿")})
    if unsupported:
        preview = " ".join(unsupported[:8])
        raise gr.Error(
            "VocalRender 仅支持中文歌词 · VocalRender was trained only on Chinese lyrics. "
            f"Please remove unsupported characters or languages: {preview}"
        )

    if not any("㐀" <= char <= "鿿" for char in check_text):
        raise gr.Error("请输入中文歌词 · Please enter Chinese lyrics. This checkpoint does not support other languages.")


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
        words = re.findall(r"SP|[㐀-鿿]", normalized, flags=re.IGNORECASE)
    if len(words) > MAX_SCORE_WORDS:
        raise gr.Error(f"每次最多 {MAX_SCORE_WORDS} 个歌词单元 · Please use at most {MAX_SCORE_WORDS} lyric units per generation.")
    return words


def _score_rows_from_lyrics(lyrics: str) -> List[Dict]:
    words = _split_lyrics(lyrics)
    if not words:
        raise gr.Error("请先输入歌词 · Enter some lyrics before creating the score.")
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


# ---------------------------------------------------------------------------
# Piano-roll value helpers
# ---------------------------------------------------------------------------

def _is_rest(row: Dict) -> bool:
    return str(row.get("word", "")).strip().upper() == "SP"


def _renumber_rows(rows: List[Dict]) -> List[Dict]:
    """Make word_index contiguous from 0 in order of first appearance."""
    mapping: Dict[int, int] = {}
    renumbered = []
    for row in rows:
        mapping.setdefault(row["word_index"], len(mapping))
        renumbered.append({**row, "word_index": mapping[row["word_index"]]})
    return renumbered


def score_value(rows: List[Dict], bpm: int, beats_per_bar: int = 4) -> Dict:
    """The value carried by the piano-roll component."""
    return {"rows": _renumber_rows(rows), "bpm": int(bpm), "beats_per_bar": int(beats_per_bar)}


def rows_from_score_value(value) -> List[Dict]:
    """Validate the editor value coming back from the browser and return clean rows."""
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise gr.Error("乐谱为空 · The score is empty. Draw notes, load a preset, or import a score.")
    rows = []
    for raw in value["rows"]:
        if not isinstance(raw, dict):
            raise gr.Error("乐谱数据无效 · Invalid score data.")
        try:
            pitch = int(raw["pitch"])
            duration_index = int(raw["duration_index"])
            word_index = int(raw["word_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise gr.Error("乐谱数据无效 · Invalid score data.") from exc
        word = str(raw.get("word", "")).strip()
        if _is_rest(raw):
            word = "SP"  # dataset rests may carry a pitch; keep it as-is
        elif not re.fullmatch(r"[㐀-鿿]+", word):
            raise gr.Error(f"歌词“{word}”不是中文 · Lyric {word!r} is not Chinese; this checkpoint supports Chinese only.")
        if not 0 <= pitch <= 127:
            raise gr.Error("MIDI 音高须在 0–127 之间 · MIDI pitches must be between 0 and 127.")
        if not 0 <= duration_index < len(NOTE_DURATION_OPTIONS):
            raise gr.Error("音值无效 · Invalid note duration.")
        rows.append({
            "word": word,
            "word_index": word_index,
            "uid": str(raw.get("uid", f"row-{len(rows)}")),
            "pitch": pitch,
            "duration_index": duration_index,
        })
    if not rows:
        raise gr.Error("乐谱为空 · The score is empty. Draw notes, load a preset, or import a score.")
    if len(rows) > MAX_SCORE_EVENTS:
        raise gr.Error(f"音符事件过多 ({len(rows)}/{MAX_SCORE_EVENTS}) · Too many score events.")
    rows = _renumber_rows(rows)
    if rows[-1]["word_index"] + 1 > MAX_SCORE_WORDS:
        raise gr.Error(f"歌词单元过多 · At most {MAX_SCORE_WORDS} lyric/rest units per generation.")
    return rows


def bpm_from_score_value(value) -> int:
    try:
        bpm = int(round(float(value.get("bpm", DEFAULT_BPM))))
    except (AttributeError, TypeError, ValueError) as exc:
        raise gr.Error("速度无效 · Invalid tempo.") from exc
    if not 1 <= bpm <= 255:
        raise gr.Error("速度须在 1–255 BPM 之间 · Tempo must be between 1 and 255 BPM.")
    return bpm


def _score_message(text: str) -> gr.Markdown:
    return gr.Markdown(text)


def create_or_apply_lyrics(lyrics: str, score):
    """Fill the roll from the lyrics box: create quarter notes when empty, else relabel notes in order."""
    words = _split_lyrics(lyrics)
    existing = score.get("rows") if isinstance(score, dict) else None
    if not existing:
        rows = _score_rows_from_lyrics(lyrics)
        return (
            score_value(rows, DEFAULT_BPM if not isinstance(score, dict) else score.get("bpm", DEFAULT_BPM)),
            _score_message(
                f"✅ 已按字创建 {len(rows)} 个四分音符（C4）· Created {len(rows)} quarter notes at C4. "
                "拖动改音高、拖右缘改时值 · Drag notes to set pitch and drag their right edge for length."
            ),
        )

    rows = rows_from_score_value(score)
    units = [word for word in words if word.upper() != "SP"]
    sung_words = len({row["word_index"] for row in rows if not _is_rest(row)})
    if len(units) != sung_words:
        raise gr.Error(
            f"乐谱有 {sung_words} 个填词位置，输入了 {len(units)} 个字。请调整歌词或音符后再填入；原歌词未改变。 "
            f"The score has {sung_words} lyric slots, but you entered {len(units)} units. "
            "Match the lyrics to the notes first; existing lyrics are unchanged."
        )
    assigned = 0
    current_word = None
    for index, row in enumerate(rows):
        if _is_rest(row):
            continue
        if row["word_index"] != current_word:
            current_word = row["word_index"]
            if assigned < len(units):
                word = units[assigned]
                assigned += 1
                for later in rows[index:]:
                    if later["word_index"] != current_word:
                        break
                    later["word"] = word
    return (
        score_value(rows, bpm_from_score_value(score), score.get("beats_per_bar", 4)),
        _score_message(f"✅ 已将 {assigned} 个字填入音符 · Applied {assigned} lyric unit(s) to the notes."),
    )


def load_random_preset():
    """Choose a dataset preset and load it for user editing without using a GPU."""
    preset = random.choice(SCORE_PRESETS)
    sung_words = [word for word in preset["words"] if word.upper() != "SP"]
    lyrics = ("|" if any(len(word) > 1 for word in sung_words) else "").join(sung_words)
    rows = _preset_to_rows(preset)
    message = _score_message("")
    return lyrics, score_value(rows, preset["bpm"]), message


@spaces.GPU(duration=60)
def generate_from_score(
    score,
    checkpoint,
    voice,
    uploaded_audio,
    cfg,
    steps,
    temperature,
    max_len,
):
    """Top-level ZeroGPU endpoint fed by the piano-roll component value."""
    rows = rows_from_score_value(score)
    bpm = bpm_from_score_value(score)
    words_by_index: Dict[int, str] = {}
    for row in rows:
        words_by_index.setdefault(row["word_index"], row["word"])
    words = [words_by_index[index] for index in sorted(words_by_index)]
    pitches = [row["pitch"] for row in rows]
    notes = [NOTE_DURATION_OPTIONS[row["duration_index"]][1] for row in rows]
    pitch2word = [row["word_index"] for row in rows]
    return _generate_impl(
        checkpoint,
        voice,
        uploaded_audio,
        "|".join(words),
        ",".join(map(str, pitches)),
        ",".join(notes),
        ",".join(map(str, pitch2word)),
        bpm,
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
        (f"内嵌歌词行 Embedded lyric line {verse_id}", verse_id)
        for verse_id in part["verses"]
    ]
    verse_choices.append(("使用歌词框 Use lyrics textbox", "__external__"))
    verse_value = part["verses"][0] if part["verses"] else "__external__"
    measure_choices = [(f"小节 Measure {measure}", measure) for measure in part["measures"]]
    recommended_start, recommended_end = recommended_measure_range(part)
    recommended_start = recommended_start or part["measures"][0]
    recommended_end = recommended_end or part["measures"][0]
    summary = imported_part_summary(part)
    if parsed.get("warnings"):
        summary += "\n\n**载入前请注意 · Before loading**\n" + "\n".join(
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
            "✅ 乐谱已解析 · Score parsed. " + summary,
        )
    except ScoreImportError as exc:
        raise gr.Error(str(exc)) from exc


def change_imported_part(parsed, part_key):
    if not parsed:
        raise gr.Error("请先解析乐谱 · Parse a score first.")
    return _import_part_updates(parsed, part_key)


def load_imported_score(parsed, part_key, verse_id, start_measure, end_measure, lyrics):
    """Convert the selected range into the piano-roll score value."""
    if not parsed:
        raise gr.Error("请先解析乐谱 · Parse a score first.")
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
        message += "\n\n**导入说明 · Import notes**\n" + "\n".join(
            f"- {warning}" for warning in result["warnings"]
        )
    return result["lyrics"], score_value(result["rows"], result["bpm"]), _score_message(message)


def switch_checkpoint(variant):
    """Load the selected checkpoint now so the first generation does not pay for it."""
    if UI_ONLY:
        return gr.Markdown(f"🧪 UI-only mode: would load **{variant}**.")
    if models.variant == variant:
        return gr.Markdown(f"✅ **{variant}** 已就绪 · ready.")
    t0 = time.perf_counter()
    models.get(variant)
    return gr.Markdown(f"✅ 已切换到 **{variant}**（{time.perf_counter() - t0:.0f}s）· Switched to {variant}.")


def toggle_upload(voice):
    return gr.Audio(visible=voice == UPLOAD_VOICE)


def preview_voice(voice):
    path = VOICE_PRESETS.get(voice)
    return gr.Audio(value=path, visible=path is not None)


# Static copy is registered once; language changes update labels without replacing
# user values, the score, selection, or undo history.
LOCALIZED = []


def ui(component, **kwargs):
    translations = {key: value for key, value in kwargs.items() if isinstance(value, tuple)}
    widget = component(**{key: value[0] if key in translations else value for key, value in kwargs.items()})
    if translations:
        LOCALIZED.append((widget, translations))
    return widget


def change_language(language):
    index = 1 if language == "en" else 0
    updates = [gr.update(**{key: value[index] for key, value in translations.items()})
               for _, translations in LOCALIZED]
    return updates + [gr.HTML(language=language)]


def check_lyric_alignment(lyrics, score, language):
    """Preview mismatches on the affected notes without modifying the score."""
    try:
        units = [word for word in _split_lyrics(lyrics) if word.upper() != "SP"]
    except gr.Error as exc:
        return str(exc), gr.HTML(lyric_targets=[])
    rows = score.get("rows", []) if isinstance(score, dict) else []
    groups = list(dict.fromkeys(row["word_index"] for row in rows if not _is_rest(row)))
    missing = set(groups[len(units):])
    targets = [row["uid"] for row in rows if row["word_index"] in missing and not _is_rest(row)]
    if not rows or len(units) == len(groups):
        return "", gr.HTML(lyric_targets=[])
    if language == "en":
        message = f"⚠ {len(groups)} lyric slots · {len(units)} entered. "
        message += "Highlighted notes need lyrics." if missing else f"Remove {len(units) - len(groups)} extra lyric units or add notes."
    else:
        message = f"⚠ 乐谱有 {len(groups)} 个填词位置，输入了 {len(units)} 个字。"
        message += "高亮音符尚缺歌词。" if missing else f"请删去多出的 {len(units) - len(groups)} 个字，或添加音符。"
    return message, gr.HTML(lyric_targets=targets)


GUIDE_ZH = """
### 从随机示例开始
示例来自真实歌声音频的转录。先试听旋律，改一两个音符，再生成歌声。
- **参考音色**：决定用什么声音唱；不需要唱相同歌词或旋律。只上传有权使用的 2–8 秒干净清唱。
- **歌词与旋律**：决定唱什么、音高和每个音的长度。拖音符改音高，拖右缘改时值，双击改字。
- **一字多音**：选中一个字，按“一字多音”，让这个字继续唱下一个音。
- **BPM**：控制速度。拍号仅改变编辑网格。
- **试听旋律**是合成音；左侧播放参考录音；**生成歌声**才是模型输出。

### 歌词与节奏
普通中文按字填入音符。用 `|` 自定义拆分；`SP` 表示休止；音符中的 `-` 表示继续唱前一个字。
填词数量需与乐谱一致，一字多音只计一个填词位置，休止不占填词位置。
支持全音符至三十二分音符及其附点时值。调整时值会移动后续音符。

### 快捷键
空格：试听／暂停；Esc：停止；↑↓：移调；Shift+↑↓：八度；[ / ]：时值；
N：音符；R：休止；M：一字多音；Delete：删除；Ctrl/Cmd+Z：撤销。

### 导入乐谱
支持 ABC 和 MusicXML。解析后选择声部、歌词行和小节范围，再载入。
ABC `w:` 和 MusicXML 内嵌歌词可自动对齐；`W:` 是页面文字，请将对应中文粘贴到歌词框。
"""
GUIDE_EN = """
### Start with a random example
Examples are transcribed from real singing recordings. Preview the melody, edit a note or two, then generate.
- **Voice reference** sets the vocal timbre; it need not contain the same lyrics or melody. Upload only 2–8 seconds of clean singing you have permission to use.
- **Lyrics and melody** specify the words, pitches and note lengths. Drag notes for pitch, drag their right edge for length, double-click to edit a lyric.
- **Melisma** continues the selected word on another note.
- **BPM** controls tempo. Meter changes the editor grid only.
- **Preview melody** plays a synth tone; the left player plays the reference recording; **Generate singing** produces model output.

### Lyrics and rhythm
Chinese text is split per character. Use `|` for custom units, `SP` for rests, and `-` on a note to continue the preceding word.
Match the number of lyric slots: a melisma counts once and rests use no lyric slots.
Supported lengths are whole through thirty-second notes and their dotted forms. Resizing a note shifts later notes.

### Shortcuts
Space: preview/pause; Esc: stop; ↑↓: transpose; Shift+↑↓: octave; [ / ]: length;
N: note; R: rest; M: melisma; Delete: delete; Ctrl/Cmd+Z: undo.

### Import a score
Parse ABC or MusicXML, choose a part, lyric line and measure range, then load it.
ABC `w:` and MusicXML embedded lyrics align automatically. For `W:` page text, paste the matching Chinese passage into the lyrics box.
"""

with gr.Blocks(title="VocalRender") as demo:
    with gr.Row(elem_id="vr-heading"):
        ui(gr.Markdown, value=(
            "# VocalRender\n给定歌词与旋律，用参考音色生成歌声。",
            "# VocalRender\nTurn lyrics and a melody into singing with a reference voice."), elem_id="vr-header")
        language = gr.Radio(choices=[("中文", "zh"), ("English", "en")], value="zh",
                            show_label=False, container=False, elem_id="vr-language", scale=0, min_width=230)
    ui(gr.Markdown, value=(
        "控制 **歌词 · 音高 · 时值 · 速度 · 音色**。参考录音无需演唱相同内容。当前支持 **中文歌词**。",
        "Control **lyrics · pitch · note length · tempo · timbre**. The reference need not sing the same content. **Chinese lyrics only**."), elem_id="vr-intro")
    gr.Markdown("[Model](https://huggingface.co/pymaster/VocalRender) · [Paper](https://arxiv.org/abs/2607.27768) · [Code](https://github.com/pymaster17/VocalRender)", elem_id="vr-links")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=235, elem_id="vr-sidebar"):
            ui(gr.Markdown, value=("### 1 · 选择音色", "### 1 · Choose a voice"))
            voice_preset = ui(gr.Dropdown, choices=(VOICE_CHOICES, [(key, key) for key in VOICE_PRESETS]), value="Alto-1",
                label=("参考音色", "Voice reference"),
                info=("决定用什么声音唱。内置参考来自 GTSinger。", "Sets the vocal timbre. Included references come from GTSinger."), interactive=True)
            voice_preview = ui(gr.Audio, value=VOICE_PRESETS["Alto-1"],
                label=("试听参考录音", "Listen to the reference"), interactive=False, elem_id="vr-reference")
            prompt_audio = ui(gr.Audio, label=("上传 2–8 秒干净清唱", "Upload 2–8 s of clean singing"),
                type="filepath", format="wav", interactive=True, visible=False)
            ui(gr.Markdown, value=("参考无需唱相同歌词或旋律。", "The reference can use different lyrics and melody."))
            checkpoint = ui(gr.Radio, choices=list(CKPT_VARIANTS), value=DEFAULT_CKPT_VARIANT,
                label=("模型版本", "Model version"), interactive=True, visible=len(CKPT_VARIANTS) > 1)
            with ui(gr.Accordion, label=("导入 ABC / MusicXML", "Import ABC / MusicXML"), open=False):
                ui(gr.Markdown, value=("上传乐谱或粘贴 ABC，解析后选择要试听的小节。", "Upload a score or paste ABC, then parse and choose a measure range."))
                abc_score_text = ui(gr.Textbox, label=("ABC 记谱", "ABC notation"), lines=4,
                    placeholder="X:1\nM:4/4\nL:1/4\nQ:1/4=90\nK:C\nC D E F |\nw: 我 爱 唱 歌")
                score_file = ui(gr.File, label=("乐谱文件", "Score file"), file_types=[".abc", ".txt", ".musicxml", ".xml", ".mxl"], type="filepath")
                parse_score_btn = ui(gr.Button, value=("解析乐谱", "Parse score"), variant="secondary")
                imported_score_state = gr.State(None)
                imported_part = ui(gr.Dropdown, label=("作品 / 声部", "Work / vocal part"), choices=[])
                imported_verse = ui(gr.Dropdown, label=("歌词行", "Lyric line"), choices=[])
                with gr.Row():
                    imported_start_measure = ui(gr.Dropdown, label=("起始小节", "Start measure"), choices=[])
                    imported_end_measure = ui(gr.Dropdown, label=("结束小节", "End measure"), choices=[])
                imported_score_info = gr.Markdown("")
                load_imported_btn = ui(gr.Button, value=("载入卷帘", "Load into piano roll"), variant="secondary")
            with ui(gr.Accordion, label=("高级设置", "Advanced settings"), open=False):
                cfg_value = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="CFG")
                inference_timesteps = ui(gr.Slider, minimum=1, maximum=50, value=10, step=1, label=("推理步数", "Inference steps"))
                temperature = ui(gr.Slider, minimum=0.1, maximum=2.0, value=1.0, step=0.1, label=("采样温度", "Temperature"))
                max_len = gr.Slider(100, 3000, value=2000, step=100, label="Max length (patches)")
            with ui(gr.Accordion, label=("使用指南与快捷键", "Guide & shortcuts"), open=False):
                ui(gr.Markdown, value=(GUIDE_ZH, GUIDE_EN))

        with gr.Column(scale=4):
            ui(gr.Markdown, value=("### 2 · 试听与调整旋律", "### 2 · Preview and edit the melody"), elem_id="vr-editor-title")
            ui(gr.Markdown, value=(
                "已随机载入真实歌声转录示例。直接生成，或改一两个音符，听听模型如何跟随乐谱。",
                "Start with a random score transcribed from real singing. Generate as is, or edit a note to hear how the model follows your score."), elem_id="vr-example-note")
            with gr.Row(elem_id="vr-lyrics-row"):
                lyrics_str = ui(gr.Textbox, label=("歌词 · 决定唱什么", "Lyrics · what to sing"), value="",
                    placeholder=("输入中文歌词", "Enter Chinese lyrics"), scale=5)
                apply_lyrics_btn = ui(gr.Button, value=("将歌词填入音符", "Fill notes with lyrics"), variant="secondary", scale=1)
                random_preset_btn = ui(gr.Button, value=("换个随机示例", "Random example"), variant="secondary", scale=1)
            alignment_info = gr.Markdown("", elem_id="vr-alignment")
            with gr.Row(elem_id="vr-generation"):
                easy_run_btn = ui(gr.Button, value=("生成歌声", "Generate singing"), variant="primary", scale=1, elem_id="vr-generate")
                ui(gr.Markdown, value=("使用 ZeroGPU，可能需要排队。旋律试听无需 GPU。", "ZeroGPU generation may queue. Melody preview needs no GPU."))
            score_editor = gr.HTML(value=score_value([], DEFAULT_BPM), html_template=PIANO_ROLL_HTML,
                css_template=PIANO_ROLL_CSS, js_on_load=PIANO_ROLL_JS,
                language="zh", lyric_targets=[], elem_id="piano-roll")
            preset_info = gr.Markdown("", elem_id="vr-preset-info")
            status_out = gr.Markdown("", min_height=0)
            audio_out = ui(gr.Audio, label=("模型生成的歌声", "Model-generated singing"), type="filepath", format="wav", elem_id="vr-result")
            prompt_out = gr.Textbox(label="Generated SVS prompt (debug)", visible=UI_ONLY)

    language.change(change_language, inputs=language,
                    outputs=[widget for widget, _ in LOCALIZED] + [score_editor], queue=False).then(
        check_lyric_alignment, inputs=[lyrics_str, score_editor, language],
        outputs=[alignment_info, score_editor], queue=False, show_progress="hidden")
    lyrics_str.change(check_lyric_alignment, inputs=[lyrics_str, score_editor, language],
                     outputs=[alignment_info, score_editor], queue=False, show_progress="hidden")
    demo.load(load_random_preset, outputs=[lyrics_str, score_editor, preset_info], queue=False)

    # ------------------------------------------------------------------ events
    checkpoint.input(switch_checkpoint, inputs=checkpoint, outputs=status_out)
    voice_preset.change(toggle_upload, inputs=voice_preset, outputs=prompt_audio)
    voice_preset.change(preview_voice, inputs=voice_preset, outputs=voice_preview)

    apply_lyrics_btn.click(
        create_or_apply_lyrics,
        inputs=[lyrics_str, score_editor],
        outputs=[score_editor, preset_info],
    )
    lyrics_str.submit(
        create_or_apply_lyrics,
        inputs=[lyrics_str, score_editor],
        outputs=[score_editor, preset_info],
    )
    random_preset_btn.click(
        fn=load_random_preset,
        inputs=None,
        outputs=[lyrics_str, score_editor, preset_info],
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
        outputs=[lyrics_str, score_editor, preset_info],
    )

    easy_run_btn.click(
        fn=generate_from_score,
        inputs=[
            score_editor,
            checkpoint,
            voice_preset,
            prompt_audio,
            cfg_value,
            inference_timesteps,
            temperature,
            max_len,
        ],
        outputs=[audio_out, status_out, prompt_out],
    )


if __name__ == "__main__":
    # Host/port come from GRADIO_SERVER_NAME / GRADIO_SERVER_PORT (Gradio reads
    # them itself); Spaces sets them automatically, self-hosting usually wants
    # GRADIO_SERVER_NAME=0.0.0.0.
    demo.launch(mcp_server=True, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=CSS)
