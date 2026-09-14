# VocalRender

**English** | [简体中文](README.zh-CN.md)

[![arXiv](https://img.shields.io/badge/arXiv-2607.27768-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2607.27768)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-VocalRender-FFD21E?logo=huggingface)](https://huggingface.co/pymaster/VocalRender)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-CrawlSinger--OS-FFD21E?logo=huggingface)](https://huggingface.co/datasets/pymaster/CrawlSinger-OS)
[![Online Inference Demo](https://img.shields.io/badge/Hugging%20Face-Online%20Inference%20Demo-FFD21E?logo=huggingface)](https://huggingface.co/spaces/pymaster/VocalRender-demo)
[![Audio Demo](https://img.shields.io/badge/Audio-Demo-183d32.svg)](https://pymaster17.github.io/VocalRender/)

Try both released checkpoints in your browser with the **[online inference demo](https://huggingface.co/spaces/pymaster/VocalRender-demo)**.

**VocalRender** is a score-native singing voice synthesis (SVS) system designed
for real-world composition. It directly renders composer-oriented symbolic
scores into singing audio through an original combination of an interleaved
lyric--note representation, continuous acoustic latents, and autoregressive
diffusion modeling. Its input is a **word / pitch / note interleaved score
prompt**:

```
<BPM_90> 感<P_62><NOTE_8> 受<P_62><NOTE_DOT_16><P_60><NOTE_16> 停<P_59><NOTE_8> ... <audio_start> [audio latents...]
```

Each lyric word is followed by one or more `(pitch, note-duration)` token
pairs, plus a global BPM token. Together with a required prompt-audio clip for
timbre conditioning, the model renders the score into 48 kHz singing audio.

This repository is the minimal open-source release: data preprocessing,
training, and inference.

## Overview

Unlike duration-based SVS systems that require an exact duration for every
word or phoneme, or reference-based systems that depend on time-aligned audio
or an F0 curve, VocalRender takes the same symbolic information used by a
composer: lyrics, MIDI pitches, note values, and a global tempo. The model is
free to realize natural timing and expressive deviations while following the
score.

![Comparison of duration-based, reference-based, and the proposed score-native singing voice synthesis inputs](assets/intro.png)

VocalRender is built around three ideas:

1. **Score-native interleaved representation.** The music-score tokenizer
   serializes BPM followed by each lyric syllable and all of its associated
   `(pitch, note-value)` pairs. This explicitly preserves lyric-to-note
   alignment, including melisma, where one syllable spans multiple notes.
2. **Continuous acoustic representation.** An Audio VAE encodes singing into
   compact continuous latents, retaining fine-grained pitch, timbre, and
   articulation information without discrete acoustic quantization.
3. **Autoregressive diffusion generation.** The AR Transformer generates a
   prosody sketch and predicts the sequence length patch by patch, while
   LocDiT reconstructs high-fidelity local acoustic latents. The VAE decoder
   then renders the completed latent sequence into waveform audio. This avoids
   an explicit duration predictor and time-aligned acoustic guidance.

![Overall architecture of VocalRender and its music-score tokenization process](assets/structure.png)

## Demo inference

This is the shortest end-to-end path: install the package, download the
released checkpoint, and run a bundled score/prompt pair. It requires a
CUDA-capable GPU, but **does not require editing a config, preparing a JSON
file, or supplying your own prompt audio**.

```bash
git clone --recurse-submodules https://github.com/pymaster17/VocalRender.git
cd VocalRender

python -m venv .venv && source .venv/bin/activate
pip install -e .

hf download pymaster/VocalRender \
    --include "VocalRender/*" \
    --local-dir pretrained_models

python scripts/infer_vocalrender_svs_single.py \
    --ckpt_dir pretrained_models/VocalRender \
    --json_file examples/opencpop_demo.json \
    --item_name 2003000087 \
    --prompt_audio examples/prompt_audio/2003000081.wav \
    --output outputs/demo_2003000087.wav
```

The command writes a 48 kHz waveform to `outputs/demo_2003000087.wav`. The
released model, score, and prompt pair above have been tested together from a
clean Hugging Face download. Two additional ready-to-run pairs are bundled:

| Score `item_name` | Bundled prompt audio | Prompt duration |
|---|---|---:|
| `2003000087` | `examples/prompt_audio/2003000081.wav` | 6.17 s |
| `2017000646` | `examples/prompt_audio/2017000644.wav` | 4.19 s |
| `2044001652` | `examples/prompt_audio/2044001666.wav` | 5.33 s |

To use another pair, change only `--item_name`, `--prompt_audio`, and
`--output` according to the table. The demo score contains inference fields
only; `word_dur` and `pitch_dur` are optional visualization/evaluation
metadata. The excerpts are selected from
[OpenCpop](https://wenet.org.cn/opencpop/) and remain subject to its terms.

Prompt audio is required because the released checkpoints were trained with
prompt audio on every sample (`prompt_audio_prob: 1.0`). For your own scores,
use a clean 2-8 second singing clip; it also specifies the target timbre.

**VocalRender** and **VocalRender-Pro** have the same architecture, parameter
count, and speech-pretrained base-model initialization. They differ only in
training recipe (training data and schedule): VocalRender uses two-stage
synthetic pretraining followed by real-data finetuning on CrawlSinger-OS,
whereas VocalRender-Pro is trained longer on the larger real-singing
CrawlSinger corpus. See the paper for the complete settings. To use the Pro
checkpoint with the same inference command:

```bash
hf download pymaster/VocalRender \
    --include "VocalRender-Pro/*" \
    --local-dir pretrained_models

# Then replace --ckpt_dir with pretrained_models/VocalRender-Pro.
```

Each checkpoint download is about 9.5 GB.

## Web demo (piano roll)

The [online inference demo](https://huggingface.co/spaces/pymaster/VocalRender-demo) is
maintained in [`demo/`](demo/): a Gradio app with a piano-roll score editor, ABC / MusicXML
import, reference-voice selection and a checkpoint switch. The Space is published from this
repository (`scripts/sync_space.py`); the same app runs on your own GPU server:

```bash
pip install -e ".[demo]"          # Python >= 3.11; install torch for your driver first
python demo/app.py                # http://127.0.0.1:7860
```

See [demo/README.md](demo/README.md) for the Docker image, environment variables and tests.

## Batch inference

Batch inference runs over a preprocessed validation set and writes generated
WAVs, optional score PNGs, and `metrics_summary.json`:

```bash
python scripts/infer_vocalrender_svs.py --config_path conf/svs_infer.yaml
```

Configure dataset/checkpoint paths in
[conf/svs_infer.yaml](conf/svs_infer.yaml). Two backends are available (see
[docs/inference_backends.md](docs/inference_backends.md)):
`multi_gpu` is the default in-process backend with prompt-audio and score
rendering support; `nano_vllm` provides continuous batching for faster
metric-only runs.

Install the corresponding optional component only when needed:

```bash
# Staff-notation score rendering (save_score: true)
pip install -e ".[viz]"

# nano-vllm inference backend
pip install -e ./nanovllm-voxcpm
```

## Training

### Data preprocessing

The released training data is available as
[CrawlSinger-OS](https://huggingface.co/datasets/pymaster/CrawlSinger-OS),
which contains Muse, Muchin, SongFormDB, OpenSinger, M4Singer, and GTSinger
for training, plus Opencpop as the held-out validation set.
Download the independently sharded archives and restore the directory expected
by [conf/svs_preprocess.yaml](conf/svs_preprocess.yaml):

```bash
hf download pymaster/CrawlSinger-OS \
    --repo-type dataset \
    --local-dir data/CrawlSinger-OS-release

mkdir -p data/CrawlSinger-OS
find data/CrawlSinger-OS-release -type f -name '*.tar' -print0 |
    while IFS= read -r -d '' shard; do
        tar -xf "$shard" -C data/CrawlSinger-OS
    done

for dataset in opensinger m4singer gtsinger opencpop; do
    cp "data/CrawlSinger-OS-release/${dataset}/annotations.json" \
       "data/CrawlSinger-OS/${dataset}/annotations.json"
done
```

The archives contain the three folder-based datasets directly and place the
audio for each of the four JSON-based datasets under `<dataset>/audio/`. The copied
annotation files complete the layout consumed by the default preprocessing
configuration. Keep enough disk space for both the downloaded archives and
the extracted data.

Annotate each audio segment with word/pitch/note fields (see the schema
comment in [conf/svs_preprocess.yaml](conf/svs_preprocess.yaml)):

```jsonc
[
  {
    "item_name": "Alto-1#newboy#0000",
    "wav_fn":    "Alto-1#newboy/0000.wav",
    "word":       ["AP", "感", "受", "SP"],
    "word_dur":   [0.14, 0.31, 0.42, 0.20],
    "pitch":      [0, 62, 62, 0],
    "note":       ["<NOTE_8>", "<NOTE_8>", "<NOTE_DOT_16>", "<NOTE_8>"],
    "pitch_dur":  [0.14, 0.31, 0.42, 0.20],
    "pitch2word": [0, 1, 2, 3],
    "bpm":        90
  }
]
```

`word_dur` and `pitch_dur` are optional input fields used only for
visualization and evaluation. They are not used to construct the score prompt
or train the model. The required score fields are `word`, `pitch`, `note`,
`pitch2word`, and `bpm`.

Then encode audio into AudioVAE-V2 latents (Arrow shards):

```bash
python scripts/preprocess_svs_data.py conf/svs_preprocess.yaml
```

### Prepare the base checkpoint

These steps are required only for training, not for demo inference:

1. Download the **VoxCPM2** pretrained checkpoint into
   `pretrained_models/VoxCPM2` (including `config.json`, model weights, and
   tokenizer files).
2. Extend its tokenizer with 128 pitch, 12 note-duration, and 256 BPM tokens:

```bash
python scripts/setup_svs_tokenizer.py \
    --tokenizer_path pretrained_models/VoxCPM2 \
    --save_path pretrained_models/VoxCPM2
```

Model embeddings are resized automatically when training starts.

### Start training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_vocalrender_svs.py --config_path conf/svs_train.yaml
```

Any config key can be overridden with dotted paths, e.g.
`--set train.batch_size=32 --set runtime.save_path=checkpoints/run2`.
Training resumes automatically from the latest checkpoint under `save_path`.

Validation logs loss plus audio-quality metrics (SingMOS, Audiobox
Aesthetics) and sample audio to TensorBoard.

## Metrics

- **SingMOS** — singing MOS predictor (loaded via `torch.hub`, requires `s3prl`).
- **AES** — Audiobox Aesthetics axes (CE = content enjoyment, PQ = production quality).
- A pluggable `register_metric_backend` seam in
  `vocalrender.evaluation.svs_metrics` lets you add custom metrics without
  editing the evaluator.

## Repository layout

```
conf/               Training / inference / preprocessing YAML configs
scripts/            Entry-point scripts (preprocess, train, infer)
src/vocalrender/    The package (model, training, inference, evaluation)
demo/               Web demo (Gradio piano-roll UI), published to the HF Space
nanovllm-voxcpm/    Optional nano-vllm inference backend (git submodule)
docs/               Architecture and usage documentation
```

See [docs/structure.md](docs/structure.md) for the full tree.

## Acknowledgements

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) (OpenBMB) — the TTS foundation
  model this work builds on; the model architecture (TSLM / LocEnc / LocDiT /
  AudioVAE) and pretrained weights come from the VoxCPM project.
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — the lightweight
  vLLM implementation adapted for the `nano_vllm` inference backend.
- [SingMOS](https://github.com/South-Twilight/SingMOS) and
  [audiobox-aesthetics](https://github.com/facebookresearch/audiobox-aesthetics)
  — evaluation models.

## License

Apache-2.0 (same as upstream VoxCPM). See [LICENSE](LICENSE).
