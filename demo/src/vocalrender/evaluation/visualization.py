"""
Visualization utilities for SVS evaluation.

- create_score_condition_figure: matplotlib bar chart for TensorBoard display

These functions accept structured metadata (bpm, word, pitch, note lists)
that are stored directly in the preprocessed Arrow dataset, avoiding
redundant reverse-parsing of SVS prompt strings.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_chinese_font():
    from matplotlib import font_manager

    user_font_path = os.path.expanduser("~/.fonts/NotoSansSC-Regular.ttf")
    if os.path.exists(user_font_path):
        font_manager.fontManager.addfont(user_font_path)
        return font_manager.FontProperties(fname=user_font_path)
    return font_manager.FontProperties(family='sans-serif')


def _aligned_lyrics(words: List[str], n_pitches: int) -> List[str]:
    return list(words[:n_pitches]) + ["·"] * max(0, n_pitches - len(words))


def _draw_score_strip(
    ax_lyric, ax_midi, durs, pitches, lyrics,
    lyric_color, midi_color, edge_color, text_color, chinese_font,
    *, ylabel_lyric='Lyric', ylabel_midi='MIDI',
    lyric_fontsize=14, midi_fontsize=12, label_fontsize=11,
    placeholder=None,
):
    if not durs:
        if placeholder is not None:
            for ax in (ax_lyric, ax_midi):
                ax.text(0.5, 0.5, placeholder, ha='center', va='center',
                        fontsize=12, color=text_color)
                ax.axis('off')
        return

    positions = np.cumsum([0.0] + list(durs[:-1]))
    total_width = sum(durs)

    import matplotlib.pyplot as plt
    for pos, dur, lyric in zip(positions, durs, lyrics):
        ax_lyric.add_patch(plt.Rectangle(
            (pos, 0), dur, 1, facecolor=lyric_color,
            edgecolor=edge_color, linewidth=1.5,
        ))
        ax_lyric.text(pos + dur / 2, 0.5, lyric,
                      ha='center', va='center', fontsize=lyric_fontsize,
                      fontweight='bold', color=text_color,
                      fontproperties=chinese_font)
    for pos, dur, pitch in zip(positions, durs, pitches):
        ax_midi.add_patch(plt.Rectangle(
            (pos, 0), dur, 1, facecolor=midi_color,
            edgecolor=edge_color, linewidth=1.5,
        ))
        ax_midi.text(pos + dur / 2, 0.5, str(pitch),
                     ha='center', va='center', fontsize=midi_fontsize,
                     fontweight='bold', color=text_color)

    for ax, ylabel in ((ax_lyric, ylabel_lyric), (ax_midi, ylabel_midi)):
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight='bold',
                      color=text_color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def create_score_condition_figure(
    bpm: int,
    words: List[str],
    pitches: List[int],
    notes: List[str],
    step: Optional[int] = None,
):
    """
    Create a bar chart visualization of score control conditions for TensorBoard.

    Shows:
    - Top row: lyric characters
    - Bottom row: MIDI pitch values
    - Bar width proportional to note duration

    Args:
        bpm: Beats per minute.
        words: List of lyric characters (one per pitch segment).
        pitches: List of MIDI pitch values.
        notes: List of note token strings (e.g. "<NOTE_4>", "<NOTE_DOT_8>").
        step: Training step (shown in the title if provided).

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from vocalrender.model.svs_utils import get_svs_token_maps

    _, _, _, dur_units, _ = get_svs_token_maps()
    chinese_font = _load_chinese_font()

    if not pitches:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, 'No score data', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    note_durations = [dur_units.get(n, 1.0) for n in notes]
    lyrics = _aligned_lyrics(words, len(pitches))
    total_width = sum(note_durations)

    fig, (ax_lyric, ax_midi) = plt.subplots(
        2, 1, figsize=(max(12, total_width * 1.5), 3),
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05},
    )

    _draw_score_strip(
        ax_lyric, ax_midi, note_durations, pitches, lyrics,
        lyric_color='#E8A038', midi_color='#F5D03A',
        edge_color='#8B6914', text_color='#5A4A1A',
        chinese_font=chinese_font,
    )

    step_str = f" @ Step {step}" if step is not None else ""
    fig.suptitle(f'Score Condition (BPM={bpm}){step_str}', fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

