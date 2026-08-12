"""
SVS Score Rendering — Staff Notation PNG via hand-built LilyPond

Accepts structured metadata (bpm, words, pitches, notes, pitch2word) from
the preprocessed Arrow dataset and renders staff notation images. No SVS
prompt string parsing is needed.

Design goal: **faithfully** mirror the source annotation, even when it is
musically inconsistent. The note glyph is decided by ``pitch`` alone and
the lyric by ``word`` alone — the two are decoupled:

    * pitch > 0            → a pitched note at that MIDI value.
    * pitch == 0           → a rest glyph.
    * real word (non SP/AP)→ its lyric is shown.
    * "SP"/"AP" / empty    → blank lyric line.

So a real syllable annotated on a 0-pitch segment renders as a *rest with
its lyric in the lyric line directly beneath it*, and an "SP" annotated on
a real pitch renders as a *note with a blank lyric* — neither is silently
dropped. This is intentional for visual inspection of the raw labels.

LilyPond's ``\\lyricsto`` skips rests, so it can't place a syllable under a
rest. Instead the lyric line is a timed ``\\lyricmode`` (one duration-tagged
token per segment — note *and* rest) that aligns to the staff by musical
moment, so a lyric sits under its own rest at the normal lyric baseline.

**Melisma slurs.** When one syllable is sung across several notes (a melisma —
``pitch2word`` maps multiple note segments to the same word index), those notes
are joined with a **slur** (连音线), the standard vocal-notation mark for "one
syllable, connected pitches". The slur spans from the first to the last *pitched*
note of each same-word run (runs with a single pitched note get none). Toggle
with ``draw_slurs`` (default on).

The key signature is auto-detected from the pitch distribution
(Krumhansl-Kessler profile correlation) so accidentals are spelled
correctly (flats vs sharps). Pass ``key_root`` to override.
"""

import os
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


# ──────────────────────────────────────────────────────────────
# Duration mapping: SVS note token → LilyPond duration string
# ──────────────────────────────────────────────────────────────
_NOTE_TOKEN_TO_LY = {
    "<NOTE_1>":      "1",
    "<NOTE_DOT_1>":  "1.",
    "<NOTE_2>":      "2",
    "<NOTE_DOT_2>":  "2.",
    "<NOTE_4>":      "4",
    "<NOTE_DOT_4>":  "4.",
    "<NOTE_8>":      "8",
    "<NOTE_DOT_8>":  "8.",
    "<NOTE_16>":     "16",
    "<NOTE_DOT_16>": "16.",
    "<NOTE_32>":     "32",
    "<NOTE_DOT_32>": "32.",
}


# ──────────────────────────────────────────────────────────────
# Key detection (Krumhansl-Kessler) and key-aware spelling
# ──────────────────────────────────────────────────────────────

# Krumhansl-Kessler major key profile (perceptual weights)
_KK_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88]

# Sharp keys: C, G, D, A, E, B  (pitch classes 0, 7, 2, 9, 4, 11)
# Flat keys:  F, Bb, Eb, Ab, Db, Gb (pitch classes 5, 10, 3, 8, 1, 6)
_FLAT_KEY_ROOTS = {5, 10, 3, 8, 1, 6}

# Pitch class → LilyPond note name (Dutch: cis=C#, des=Db, …). Flat keys use
# flat spelling so accidentals match the key signature.
_LY_NAMES_SHARP = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]
_LY_NAMES_FLAT  = ["c", "des", "d", "ees", "e", "f", "ges", "g", "aes", "a", "bes", "b"]

# Pitch class → LilyPond ``\key`` tonic name.
_KEY_ROOT_TO_LY_SHARP = {
    0: "c", 1: "cis", 2: "d", 3: "dis", 4: "e", 5: "f",
    6: "fis", 7: "g", 8: "gis", 9: "a", 10: "ais", 11: "b",
}
_KEY_ROOT_TO_LY_FLAT = {
    0: "c", 1: "des", 2: "d", 3: "ees", 4: "e", 5: "f",
    6: "ges", 7: "g", 8: "aes", 9: "a", 10: "bes", 11: "b",
}


def detect_key(pitches: List[int]) -> int:
    """Detect the most likely major key root from a MIDI pitch distribution.

    Uses Krumhansl-Kessler profile correlation.

    Args:
        pitches: MIDI pitch per segment (0 / negative = rest, ignored).

    Returns:
        Key root as pitch class (0=C, 1=Db, 2=D, ... 11=B). Defaults to
        0 (C) when there are no pitched events.
    """
    counts = [0] * 12
    for p in pitches:
        if p > 0:
            counts[int(p) % 12] += 1
    total = sum(counts)
    if total == 0:
        return 0  # default C

    best_root, best_score = 0, -999.0
    mean_obs = total / 12
    mean_prof = sum(_KK_MAJOR_PROFILE) / 12
    for root in range(12):
        rotated = [counts[(root + i) % 12] for i in range(12)]
        num = sum((r - mean_obs) * (pf - mean_prof)
                  for r, pf in zip(rotated, _KK_MAJOR_PROFILE))
        den_obs = sum((r - mean_obs) ** 2 for r in rotated) ** 0.5
        den_prof = sum((pf - mean_prof) ** 2 for pf in _KK_MAJOR_PROFILE) ** 0.5
        score = num / (den_obs * den_prof) if den_obs * den_prof > 0 else 0
        if score > best_score:
            best_score = score
            best_root = root
    return best_root


def _midi_to_lily(midi_val: int, use_flats: bool) -> str:
    """MIDI pitch → LilyPond note string (e.g. 60 → ``c'``).

    MIDI octave convention: 60 = C4 (middle C) = LilyPond ``c'``.
    """
    name = (_LY_NAMES_FLAT if use_flats else _LY_NAMES_SHARP)[midi_val % 12]
    ly_oct = midi_val // 12 - 1 - 3  # MIDI octave (60→4) → LilyPond (c'=4)
    if ly_oct > 0:
        name += "'" * ly_oct
    elif ly_oct < 0:
        name += "," * (-ly_oct)
    return name


def _ly_text(word: str) -> str:
    """Escape a lyric/markup string for a LilyPond double-quoted token."""
    return word.replace("\\", "").replace('"', '')


def build_lily_source(
    bpm: int,
    words: List[str],
    pitches: List[int],
    notes: List[str],
    pitch2word: Optional[List[int]] = None,
    key_root: Optional[int] = None,
    draw_slurs: bool = True,
    draw_beams: bool = False,
) -> str:
    """
    Build a LilyPond source string from structured SVS metadata.

    Glyph (note vs rest) follows ``pitches``; lyric follows ``words`` — the
    two are decoupled so the raw annotation is shown faithfully even when it
    is musically wrong (see the module docstring).

    Args:
        bpm: Beats per minute.
        words: Lyric characters, **one per syllable** (may contain "AP"/"SP"
            rest markers). With melisma there are fewer words than notes.
        pitches: MIDI pitch per note segment (0 for rests).
        notes: Note token strings, e.g. ["<NOTE_4>", ...], one per segment.
        pitch2word: ``pitch2word[i]`` = the word index that note ``i`` belongs
            to (melisma alignment). ``None``/mismatched → identity 1:1.
        key_root: Optional key root pitch class (0=C..11=B). ``None`` →
            auto-detect via the Krumhansl-Kessler profile.
        draw_slurs: When True (default), join each melisma's notes (multiple
            notes under one word index) with a slur (连音线).
        draw_beams: When True, **explicitly** beam maximal runs of consecutive
            beamable notes (eighth-and-shorter, broken by rests / quarter-plus
            notes). Off by default — a full score relies on LilyPond's metric
            auto-beaming. Enable it for a short **excerpt** that doesn't fill a
            measure (LilyPond leaves an incomplete measure's notes flagged, so an
            excerpt would otherwise look un-beamed vs the complete score).

    Returns:
        Complete LilyPond source text.
    """
    if not pitches or not notes:
        raise ValueError("Empty pitches or notes list")

    # ``pitch2word`` is per-note; a valid one matches ``pitches`` in length.
    # Anything else falls back to identity (positional word↔note).
    if not pitch2word or len(pitch2word) != len(pitches):
        pitch2word = list(range(len(pitches)))

    # Melisma slurs: within each maximal run of consecutive segments sharing a
    # real (non SP/AP) word index, slur from the first to the last *pitched*
    # note when the run has ≥2 of them. Rests advance without breaking the run
    # (the slur spans them); a single-note syllable gets no slur. Keyed on the
    # word *index*, so two adjacent identical characters stay separate.
    n = len(pitches)
    slur_open = [False] * n
    slur_close = [False] * n
    if draw_slurs:
        i = 0
        while i < n:
            widx = pitch2word[i]
            word = words[widx] if 0 <= widx < len(words) else ""
            if not word or word.upper() in ("AP", "SP"):
                i += 1
                continue
            j = i
            while j < n and pitch2word[j] == widx:
                j += 1
            pitched = [k for k in range(i, j) if int(pitches[k]) > 0]
            if len(pitched) >= 2:
                slur_open[pitched[0]] = True
                slur_close[pitched[-1]] = True
            i = j

    # Manual beams: maximal runs of consecutive beamable pitched notes (an
    # eighth or shorter), broken by rests or quarter-plus notes. Only used for
    # excerpts (see ``draw_beams``); a full score keeps LilyPond's auto-beaming.
    _BEAMABLE = {"<NOTE_8>", "<NOTE_DOT_8>", "<NOTE_16>", "<NOTE_DOT_16>",
                 "<NOTE_32>", "<NOTE_DOT_32>"}
    beam_open = [False] * n
    beam_close = [False] * n
    if draw_beams:
        i = 0
        while i < n:
            tok = notes[i] if i < len(notes) else ""
            if int(pitches[i]) > 0 and tok in _BEAMABLE:
                j = i
                while (j < n and int(pitches[j]) > 0
                       and (notes[j] if j < len(notes) else "") in _BEAMABLE):
                    j += 1
                if j - i >= 2:
                    beam_open[i] = True
                    beam_close[j - 1] = True
                i = j
            else:
                i += 1

    if key_root is None:
        key_root = detect_key(pitches)
    use_flats = key_root in _FLAT_KEY_ROOTS
    key_ly = (_KEY_ROOT_TO_LY_FLAT if use_flats else _KEY_ROOT_TO_LY_SHARP)[key_root]

    # Clef from the median pitched note (bass for low voices).
    pitched = [m for m in pitches if m > 0]
    if pitched:
        median = sorted(pitched)[len(pitched) // 2]
        clef = "bass" if median < 55 else "treble"
    else:
        clef = "treble"

    # One duration-tagged token per segment in BOTH lines so the timed
    # \lyricmode aligns to the staff by moment (and a lyric can land under a
    # rest). Melody: note or rest. Lyric: the syllable's text, or \skip.
    melody: List[str] = []
    lyrics: List[str] = []
    prev_word_idx = None
    for i in range(len(pitches)):
        midi = int(pitches[i])
        dur = _NOTE_TOKEN_TO_LY.get(notes[i] if i < len(notes) else "", "4")
        widx = pitch2word[i]
        word = words[widx] if 0 <= widx < len(words) else ""

        is_sp = word.upper() in ("AP", "SP")
        is_real = bool(word) and not is_sp
        # Show the lyric on the first segment of each syllable; later segments
        # sharing the word index are melisma and stay blank. Keyed on word
        # *index* so two adjacent identical characters are separate syllables.
        show_lyric = is_real and widx != prev_word_idx
        prev_word_idx = widx if is_real else None

        # Glyph follows pitch; lyric follows word — fully decoupled.
        tok = f"{_midi_to_lily(midi, use_flats)}{dur}" if midi > 0 else f"r{dur}"
        # Post-events on pitched notes only (beam/slur flags never set on rests).
        # Order: beam bracket then slur — ``c'8[(`` … ``f'8])``.
        if beam_open[i]:
            tok += "["
        if beam_close[i]:
            tok += "]"
        if slur_open[i]:
            tok += "("
        if slur_close[i]:
            tok += ")"
        melody.append(tok)
        lyrics.append(f'"{_ly_text(word)}"{dur}' if show_lyric else f"\\skip {dur}")

    # Layout: reuse music21's natural look — the lilypond-book snippet
    # preamble + an empty \paper (justified default line-breaking, lines
    # stretched to full width, no blank right margin). It paginates one
    # system per page, which render_lily_to_png stitches back together.
    # Layout (\paper/preamble) and content (the << staff + lyrics >> body)
    # are orthogonal in LilyPond, so this keeps the note/lyric decoupling.
    return (
        '\\version "2.24.0"\n'
        '\\include "lilypond-book-preamble.ly"\n'
        '\\header { tagline = ##f }\n'
        '\\paper { }\n'
        '<<\n'
        '  \\new Staff { \\new Voice = "mel" {\n'
        f'    \\clef {clef} \\key {key_ly} \\major \\time 4/4 \\tempo 4 = {int(bpm)}\n'
        f'    {" ".join(melody)}\n'
        '  } }\n'
        f'  \\new Lyrics \\lyricmode {{ {" ".join(lyrics)} }}\n'
        '>>\n'
    )


def _autocrop_png(path: str, margin: int = 12) -> str:
    """Trim surrounding whitespace from a rendered PNG.

    LilyPond emits a full A4 page, so a short melody sits at the top with a
    large blank margin below. This crops to the bounding box of the actual
    content (equivalent to lilypond's ``-dcrop``), leaving a small ``margin``
    of padding. Returns ``path`` unchanged on any failure / blank image.
    """
    try:
        from PIL import Image, ImageChops

        im = Image.open(path)
        if im.mode in ("RGBA", "LA"):
            bbox = im.split()[-1].getbbox()  # non-transparent region
        else:
            rgb = im.convert("RGB")
            bg = Image.new("RGB", rgb.size, (255, 255, 255))
            bbox = ImageChops.difference(rgb, bg).getbbox()
        if bbox is None:
            return path
        w, h = im.size
        left = max(0, bbox[0] - margin)
        upper = max(0, bbox[1] - margin)
        right = min(w, bbox[2] + margin)
        lower = min(h, bbox[3] + margin)
        im.crop((left, upper, right, lower)).save(path)
    except Exception as e:
        import sys
        print(f"[Score Rendering] Autocrop skipped for {path}: {e}", file=sys.stderr)
    return path


def _stack_pngs_vertically(
    page_paths: List[str], out_path: str, margin: int = 12, gap: int = 24
) -> str:
    """Crop each LilyPond page to its content and stack them vertically.

    LilyPond writes a multi-page score as ``<base>-page1.png``,
    ``<base>-page2.png``, … (no ``<base>.png``). Each page is a full A4
    sheet with the systems at the top. We crop every page to its bounding
    box and paste them top-to-bottom into one continuous score image.
    """
    from PIL import Image, ImageChops

    crops = []
    for p in page_paths:
        im = Image.open(p).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bbox = ImageChops.difference(im, bg).getbbox()
        if bbox is None:
            continue  # blank trailing page
        left = max(0, bbox[0] - margin)
        upper = max(0, bbox[1] - margin)
        right = min(im.size[0], bbox[2] + margin)
        lower = min(im.size[1], bbox[3] + margin)
        crops.append(im.crop((left, upper, right, lower)))

    if not crops:
        raise RuntimeError("all LilyPond pages were blank")

    width = max(c.width for c in crops)
    height = sum(c.height for c in crops) + gap * (len(crops) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for c in crops:
        canvas.paste(c, (0, y))
        y += c.height + gap
    canvas.save(out_path)
    return out_path


def render_lily_to_png(ly_source: str, output_path: str, resolution: int = 300) -> str:
    """
    Render a LilyPond source string to a PNG at ``resolution`` DPI.

    The lilypond-book preamble (see :func:`build_lily_source`) emits one
    system per page, so a single-line score is one page that
    :func:`_autocrop_png` trims to content, and any multi-line score is
    several pages that :func:`_stack_pngs_vertically` stitches back into one
    continuous image.

    Args:
        ly_source: Complete LilyPond source text (see :func:`build_lily_source`).
        output_path: Output file path (without extension; .png is added).
        resolution: Render DPI (300 ≈ 3× the music21 default).

    Returns:
        Path to the generated (cropped) PNG file.
    """
    import glob
    import re
    import shutil
    import subprocess
    import tempfile

    import lilypond as lilypond_pkg

    if output_path.endswith('.png'):
        output_path = output_path[:-4]

    # output_path may nest a subdir (e.g. muse item_name "song/segment");
    # lilypond won't create missing parents.
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    lily_exec = str(lilypond_pkg.executable())
    final_png = output_path + '.png'
    with tempfile.TemporaryDirectory() as td:
        ly_path = os.path.join(td, 'score.ly')
        with open(ly_path, 'w', encoding='utf-8') as f:
            f.write(ly_source)
        base = os.path.join(td, 'score')
        # capture_output swallows lilypond's chatty stderr; gate on the PNG
        # existing rather than returncode (barcheck warnings give rc=1 even on
        # a good render).
        subprocess.run(
            [lily_exec, '-f', 'png', f'-dresolution={resolution}',
             '-dbackend=eps', '-o', base, ly_path],
            capture_output=True, text=True,
        )
        rendered = base + '.png'
        if os.path.exists(rendered):
            shutil.copy2(rendered, final_png)          # single page
        else:
            pages = sorted(
                glob.glob(base + '-page*.png'),
                key=lambda p: int(re.search(r'-page(\d+)\.png$', p).group(1)),
            )
            if not pages:
                raise RuntimeError(f"lilypond produced no PNG for {output_path}")
            _stack_pngs_vertically(pages, final_png)    # multi-page stitch

    return _autocrop_png(final_png)


def render_score(
    bpm: int,
    words: List[str],
    pitches: List[int],
    notes: List[str],
    output_path: str,
    pitch2word: Optional[List[int]] = None,
    key_root: Optional[int] = None,
    draw_slurs: bool = True,
    draw_beams: bool = False,
) -> Optional[str]:
    """
    Convenience function: structured metadata → staff notation PNG.

    Args:
        bpm: Beats per minute.
        words: Lyric characters, one per syllable.
        pitches: MIDI pitch per note segment.
        notes: Note token strings, one per note segment.
        output_path: Output PNG file path (without extension).
        pitch2word: Per-note word index for melisma-correct lyric placement
            (see :func:`build_lily_source`). ``None`` → legacy 1:1 mapping.
        key_root: Optional key root pitch class to override auto-detection.
        draw_slurs: Join each melisma's notes with a slur (see
            :func:`build_lily_source`). Default on.
        draw_beams: Explicitly beam beamable runs — for excerpts that don't fill
            a measure (see :func:`build_lily_source`). Default off.

    Returns:
        Path to the generated PNG file, or None on failure.
    """
    try:
        ly = build_lily_source(bpm, words, pitches, notes,
                               pitch2word=pitch2word, key_root=key_root,
                               draw_slurs=draw_slurs, draw_beams=draw_beams)
        return render_lily_to_png(ly, output_path)
    except Exception as e:
        import sys
        print(f"[Score Rendering] Failed to render score: {e}", file=sys.stderr)
        return None


def _render_one(args: Tuple) -> Tuple[str, Optional[str]]:
    """Worker function for parallel rendering.

    Accepts a 6-tuple ``(bpm, words, pitches, notes, pitch2word, output_path)``
    or a 7-tuple with a trailing ``key_root`` override.
    """
    bpm, words, pitches, notes, pitch2word, output_path = args[:6]
    key_root = args[6] if len(args) > 6 else None
    return output_path, render_score(bpm, words, pitches, notes, output_path,
                                     pitch2word=pitch2word, key_root=key_root)


def render_scores_parallel(
    tasks: List[Tuple],
    max_workers: int = 16,
) -> Dict[str, Optional[str]]:
    """
    Render multiple scores to PNGs in parallel.

    Args:
        tasks: List of (bpm, words, pitches, notes, pitch2word, output_path)
            tuples (optional trailing key_root).
        max_workers: Number of parallel workers.

    Returns:
        Dict mapping output_path → rendered PNG path (or None on failure).
    """
    if not tasks:
        return {}

    import sys
    print(f"[Score Rendering] Rendering {len(tasks)} scores with {max_workers} workers...",
          file=sys.stderr)

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_render_one, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                output_path, rendered_path = future.result()
                results[output_path] = rendered_path
            except Exception as e:
                task_data = futures[future]
                output_path = task_data[5]  # (bpm, words, pitches, notes, pitch2word, output_path)
                print(f"[Score Rendering] Failed: {output_path}: {e}", file=sys.stderr)
                results[output_path] = None

    success = sum(1 for v in results.values() if v is not None)
    print(f"[Score Rendering] Done: {success}/{len(tasks)} scores rendered.", file=sys.stderr)
    return results
