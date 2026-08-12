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
2. the lyrics, which the app splits into editable sung units;
3. one MIDI pitch and note duration for each lyric unit;
4. the tempo.

It then generates a new vocal performance at 48 kHz. The reference recording provides the
voice character and does not need to contain the same lyrics or melody. This research demo
currently works best with Chinese lyrics.

For the normal workflow, type the lyrics and press **Create word-by-word score**. Chinese lyrics
are split character by character, while space-delimited languages are split into words. Then use
the pitch slider and note-duration menu beside each lyric unit. You do not need to understand the
model architecture or write code.

An advanced raw-score editor remains available for melismas, where one syllable is sung across
multiple notes, and for direct use of VocalRender's native input format.

Only upload voice recordings that you own or have permission to use.

## Model

Based on [pymaster/VocalRender](https://huggingface.co/pymaster/VocalRender) — an autoregressive diffusion model initialized from VoxCPM2 speech-pretrained weights.

- **Paper**: [arXiv:2607.27768](https://arxiv.org/abs/2607.27768)
- **Code**: [github.com/pymaster17/VocalRender](https://github.com/pymaster17/VocalRender)
- **License**: Apache 2.0

## Input Format

- **Lyrics**: Normal text is split automatically; optionally use `|` to control syllable boundaries and `SP` for a rest or breath
- **Pitches**: MIDI note numbers (0-127, comma-separated; middle C is 60 and 0 means a rest)
- **Notes**: Duration tokens (`<NOTE_4>` is a quarter note, `<NOTE_8>` is an eighth note, and dotted variants use `DOT`)
- **BPM**: Beats per minute
- **Voice reference**: choose one of three included voices, or upload a 2-8 second singing clip for the target voice timbre
