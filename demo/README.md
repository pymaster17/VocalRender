---
title: VocalRender Demo
emoji: 🎵
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
short_description: Score-native singing voice synthesis from lyrics and MIDI
python_version: "3.12"
startup_duration_timeout: 1h
---

# VocalRender

Score-native singing voice synthesis (SVS) model that generates 48 kHz singing audio directly from lyrics, MIDI pitches, symbolic note values, and tempo. Takes a prompt audio clip (2-8 seconds of clean singing) to provide the target timbre.

## What does it do?

VocalRender turns a written vocal part into a sung audio performance. You give it:

1. a model checkpoint (`VocalRender-Pro` by default, or the base `VocalRender` checkpoint);
2. one of the six included GTSinger references (`Alto-1/2/3` or `Tenor-1/2/3`), or a short recording that demonstrates the desired vocal color;
3. the lyrics, which the app splits into editable sung units;
4. one or more MIDI pitches and note durations for each lyric unit;
5. the tempo.

It then generates a new vocal performance at 48 kHz. The reference recording provides the
voice character and does not need to contain the same lyrics or melody. This research demo
supports Chinese lyrics only. Inputs containing other languages are rejected before inference.

The editor follows the track + piano-roll workflow of tools such as SynthesizerV. The left
panel is the vocal track (voice reference, checkpoint, score import, advanced settings); the main
area is a piano roll with a transport bar. Type Chinese lyrics and press **应用歌词 Apply lyrics**:
an empty roll receives one quarter note per character (rests for `SP`), while a roll that already
has notes gets the words assigned to its notes in order. Then shape the melody directly:

- drag a note up or down (or use ↑/↓, Shift for octaves) to change its pitch;
- drag a note's right edge (or press `[` / `]`) to change its length — lengths snap to the twelve
  supported note values and later notes shift to stay contiguous;
- double-click a note to edit its lyric: `-` continues the previous word (a melisma), `SP` turns the
  note into a rest, and Chinese text starts a new word;
- use **+ 音符 Note**, **+ 休止 Rest** and **+ 延音 Melisma** (or `N`, `R`, `M`) to insert events after
  the selection, the pencil tool to draw notes, the eraser or `Delete` to remove them, `Ctrl+Z` to undo;
- click or drag the bar-number strip to place the playhead, then press ▶ (`Space`) to audition the
  melody with a synthesised tone before using the GPU. `Space` again pauses where the playhead is and
  the next `Space` resumes from there; ■ (`Esc`) stops and rewinds, `Home`/`End` jump to either end.
  Seeking snaps to the nearest note or rest boundary, falling back to the sixteenth-note grid past
  the end of the score — hold `Alt` for a free position.

The footer shows the event and lyric-unit counts against the model limits and flags anything that
would be rejected. You do not need to understand the model architecture or write code. Other
languages are rejected because the released checkpoint was trained only on Chinese lyrics.

Musicians can also open **Import ABC notation or MusicXML**. ABC notation can be pasted directly
or uploaded as `.abc`/`.txt`; MusicXML can be uploaded as `.musicxml`, `.xml`, or compressed `.mxl`.
After parsing, choose the work or vocal part, lyric line, and a contiguous measure range, then load
it into the piano roll. The importer prefers a lyric-bearing monophonic part and embedded
lyrics, while still allowing either choice to be changed. It preselects a compatible range that
fits the editor limits, so importing a full song does not immediately fail on length.

VocalRender's score vocabulary is monophonic and intentionally compact. Exact whole through
thirty-second note values and their dotted forms are preserved. Other values are quantized only
when the relative error is at most 12.5%, with a visible warning. Chords, overlapping voices,
zero-duration grace notes, or larger rhythmic changes are rejected with their score location
instead of silently choosing a melody. A selected range may contain at most 64 lyric/rest units
and 128 note/rest events. Scores with tempo changes use the range's starting BPM and report that
the range has been flattened to one tempo.

MusicXML lyric extensions, ties, and ABC `w:` underscore (`_`) melismas are converted to
VocalRender's pitch-to-word alignment. If the selected part has no embedded lyrics, enter Chinese
lyrics in the normal textbox before loading the range; the number of lyric units must exactly
match the pitched note attacks. ABC accompaniment labels such as `"C"` are ignored. Uppercase
`W:` lines are unaligned page text (unlike lowercase `w:`), so the demo prompts you to paste the
matching Chinese passage into the lyrics textbox. ABC import is powered by `music21`, which
implements ABC 1.6 and much, but not all, of ABC 2.1.

The **🎲 随机预设 Random preset** button chooses one of 100 ready-to-use score segments sampled from
the processed CloudTest data and loads its lyrics, complete melisma-aware score and BPM into the
piano roll. It does not start inference, so every value can be adjusted before pressing **生成歌声
Generate Singing**. Source song and artist names are not displayed.

Only upload voice recordings that you own or have permission to use.

The six bundled voice references are 48 kHz mono excerpts selected from the natural-performance
control groups of the `ZH-Alto-1` and `ZH-Tenor-1` singers in the GTSinger Chinese dataset.

The music-notation icons use Steinberg's professional
[Bravura](https://github.com/steinbergmedia/bravura) SMuFL font, bundled unmodified under the
[SIL Open Font License](assets/fonts/Bravura-LICENSE.txt).

## Deployment

This demo lives in the [`demo/`](https://github.com/pymaster17/VocalRender/tree/main/demo) directory
of the main [VocalRender](https://github.com/pymaster17/VocalRender) repository; the Hugging Face
Space is a build product published from there. The same `app.py` runs on a ZeroGPU Space and on
your own GPU server.

**Hugging Face Space** — `scripts/sync_space.py` (run by the `sync-space` GitHub workflow on every
push to `main`, or by hand with an `HF_TOKEN`) copies `demo/` and `src/vocalrender/` to the Space.
The YAML header at the top of this file configures the Space; the `spaces` package is provided by
the platform, `@spaces.GPU` schedules generation on ZeroGPU, and checkpoints are fetched from
`pymaster/VocalRender`.

**Self-hosted server** — `spaces` is not required; when it is missing the decorator becomes a
no-op and generation runs on the local GPU directly. Python 3.11+.

```bash
git clone https://github.com/pymaster17/VocalRender.git && cd VocalRender
python -m venv .venv && source .venv/bin/activate
# Install torch first, matching your NVIDIA driver (see https://pytorch.org/get-started/locally/);
# an unpinned `pip install torch` may pick a build that needs a newer driver than you have.
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[demo]"
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python demo/app.py
```

Startup loads the default checkpoint (~5 GB of GPU memory, ~30 s); switching checkpoints in the UI
takes ~20 s. A 15-event phrase renders in 4–8 s on an RTX 6000 Ada.

Or with Docker (single GPU, weights cached in the mounted HF cache; build from the repo root):

```bash
docker build -f demo/Dockerfile -t vocalrender-demo .
docker run --gpus all -p 7860:7860 -v $HOME/.cache/huggingface:/root/.cache/huggingface vocalrender-demo
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` | `127.0.0.1` / `7860` | Bind address and port (read by Gradio). |
| `VOCALRENDER_CKPT_DIR` | unset (download from HF Hub) | Local directory containing `VocalRender-Pro/` and/or `VocalRender/` (each with `config.json`, `model.safetensors`, `audiovae.pth`, tokenizer files). |
| `VOCALRENDER_CKPTS` | `VocalRender-Pro,VocalRender` | Which variants are selectable (loaded on demand, one at a time); with a single one the checkpoint selector is hidden. |
| `VOCALRENDER_DEVICE` | `cuda` | Torch device for inference. |
| `HF_HOME` / `HF_HUB_OFFLINE=1` | — | Standard Hugging Face cache location / offline mode for pre-downloaded weights. |
| `VOCALRENDER_UI_ONLY=1` | unset | Run the interface without models or a GPU (development). |

Only one checkpoint is resident at a time: the default variant is loaded at startup and the other
is loaded (replacing it) when selected in the UI, so GPU memory is that of a single model.
`VOCALRENDER_CKPTS` restricts which variants can be selected at all.

## Development

The piano roll is a single `gr.HTML` component: `assets/piano_roll/piano_roll.html` (shell),
`piano_roll.css` (scoped styles), `score_model.js` (pure score logic, no DOM) and `piano_roll.js`
(interaction, WebAudio transport). The grid draws bar and beat lines as CSS gradients plus a dashed
sixteenth-note sub-division from a repeating one-tile SVG, hidden below 8 px of spacing. It is a
rhythmic reference, not a quantization target: note lengths snap to the twelve model values, and
because events are contiguous and those values are powers of two *and their dotted forms*, a single
32nd shifts every later start off the sixteenth grid (a dotted 32nd shifts it off the 32nd grid too).
Seeking therefore snaps to event boundaries rather than to the drawn grid. Its value is `{"rows": [...], "bpm": int, "beats_per_bar": int}`,
where `rows` is the same `word / word_index / uid / pitch / duration_index` list produced by the
presets and the score importer (`vocalrender.utils.score_import`), so the backend is unchanged.

All commands run from the repository root:

```bash
uv sync --python 3.11 --extra demo --extra dev      # gradio, music21, pytest, playwright (Python >= 3.11)
node --test demo/tests/test_score_model.mjs         # score-model unit tests
uv run pytest                                       # importer (tests/) + editor value helpers (demo/tests/)
VOCALRENDER_UI_ONLY=1 uv run python demo/app.py     # run the UI without models or a GPU
uv run playwright install chromium && uv run python demo/tests/browser/drive_piano_roll.py
uv run python demo/tests/gpu_smoke.py               # on a GPU machine: load, generate, switch checkpoint, generate
python scripts/sync_space.py --dry-run              # list what would be published to the Space
```

`demo/tests/smoke_job.sbatch` runs the GPU smoke test plus a browser-driven generation on a Slurm node.

`VOCALRENDER_UI_ONLY=1` skips the checkpoint download and makes **Generate** return the selected
reference clip, so the whole editing flow can be exercised locally.

## Model

Based on [pymaster/VocalRender](https://huggingface.co/pymaster/VocalRender) — an autoregressive diffusion model initialized from VoxCPM2 speech-pretrained weights.

The demo includes a checkpoint selector for both `VocalRender-Pro` and `VocalRender`, with
`VocalRender-Pro` selected by default. Weights are loaded when a checkpoint is selected and the
previous model is released, so only one model occupies GPU memory.

- **Paper**: [arXiv:2607.27768](https://arxiv.org/abs/2607.27768)
- **Code**: [github.com/pymaster17/VocalRender](https://github.com/pymaster17/VocalRender)
- **License**: Apache 2.0

## Input Format

- **Lyrics**: Normal text is split automatically; optionally use `|` to control syllable boundaries and `SP` for a rest or breath
- **Pitches**: MIDI note numbers (0-127, comma-separated; middle C is 60 and 0 means a rest)
- **Notes**: Duration tokens (`<NOTE_4>` is a quarter note, `<NOTE_8>` is an eighth note, and dotted variants use `DOT`)
- **BPM**: Beats per minute
- **Voice reference**: choose one of six included voices, or upload a 2-8 second singing clip for the target voice timbre
- **Score import**: pasted/uploaded ABC notation or uploaded MusicXML (`.musicxml`, `.xml`, `.mxl`)
