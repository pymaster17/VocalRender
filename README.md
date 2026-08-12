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

1. one of the three included example voices, or a short recording that demonstrates the desired vocal color;
2. the lyrics, split into sung syllables;
3. the melody as MIDI note numbers;
4. the rhythm and tempo.

It then generates a new vocal performance at 48 kHz. The reference recording provides the
voice character and does not need to contain the same lyrics or melody. This research demo
currently works best with Chinese lyrics.

If you only want to try the model, choose one of the prepared examples in the app and press
**Generate Singing**. You do not need to understand the model architecture or write code.

Only upload voice recordings that you own or have permission to use.

## Model

Based on [pymaster/VocalRender](https://huggingface.co/pymaster/VocalRender) — an autoregressive diffusion model initialized from VoxCPM2 speech-pretrained weights.

- **Paper**: [arXiv:2607.27768](https://arxiv.org/abs/2607.27768)
- **Code**: [github.com/pymaster17/VocalRender](https://github.com/pymaster17/VocalRender)
- **License**: Apache 2.0

## Input Format

- **Lyrics**: Pipe-separated sung syllables (e.g., `我|的|孤|独`); use `SP` for a rest or breath
- **Pitches**: MIDI note numbers (0-127, comma-separated; middle C is 60 and 0 means a rest)
- **Notes**: Duration tokens (`<NOTE_4>` is a quarter note, `<NOTE_8>` is an eighth note, and dotted variants use `DOT`)
- **BPM**: Beats per minute
- **Voice reference**: choose one of three included voices, or upload a 2-8 second singing clip for the target voice timbre
