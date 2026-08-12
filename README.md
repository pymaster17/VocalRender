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
3. one or more MIDI pitches and note durations for each lyric unit;
4. the tempo.

It then generates a new vocal performance at 48 kHz. The reference recording provides the
voice character and does not need to contain the same lyrics or melody. This research demo
supports Chinese lyrics only. Inputs containing other languages are rejected before inference.

For the normal workflow, type Chinese lyrics and press **Create word-by-word score**. The lyrics
are split character by character. Type a MIDI pitch and use the note-duration slider beside each
lyric unit. If a syllable spans multiple notes, press **+ Melisma note** and edit every note
independently. You do not need to understand the model architecture or write code. Other
languages are rejected because the released checkpoint was trained only on Chinese lyrics.

An advanced raw-score editor remains available for custom rests and direct use of VocalRender's
native input format; the visual editor already supports melismas.

The **🎲 Random preset & generate** button chooses one of 30 ready-to-use score segments sampled
from the processed CloudTest data, loads its lyrics, complete melisma-aware score and BPM into the
editor, and starts generation using the currently selected voice reference.

Only upload voice recordings that you own or have permission to use.

The music-notation icons use Steinberg's professional
[Bravura](https://github.com/steinbergmedia/bravura) SMuFL font, bundled unmodified under the
[SIL Open Font License](assets/fonts/Bravura-LICENSE.txt).

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
