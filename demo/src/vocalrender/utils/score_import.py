"""Safe, CPU-only import of ABC notation and MusicXML for the demo.

The model consumes a deliberately small score vocabulary.  This module keeps
the lossy boundary explicit: files are parsed into a serialisable neutral
representation first, then a selected part and measure range are converted to
the demo's ``word/pitch/note/pitch2word/bpm`` representation.
"""

from __future__ import annotations

import hashlib
import re
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import defusedxml

# music21's documentation recommends installing the hardened XML parsers before
# importing music21 in any server that accepts user supplied MusicXML.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="defusedxml.cElementTree is deprecated",
        category=DeprecationWarning,
    )
    defusedxml.defuse_stdlib()

from music21 import chord, converter, harmony, note, stream, tempo  # noqa: E402


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_MXL_MEMBERS = 64
MAX_MXL_EXPANDED_BYTES = 20 * 1024 * 1024
MAX_MXL_COMPRESSION_RATIO = 200
MAX_SCORE_WORDS = 64
MAX_SCORE_EVENTS = 128

NOTE_QUARTER_LENGTHS: Tuple[Tuple[float, str], ...] = (
    (0.125, "<NOTE_32>"),
    (0.1875, "<NOTE_DOT_32>"),
    (0.25, "<NOTE_16>"),
    (0.375, "<NOTE_DOT_16>"),
    (0.5, "<NOTE_8>"),
    (0.75, "<NOTE_DOT_8>"),
    (1.0, "<NOTE_4>"),
    (1.5, "<NOTE_DOT_4>"),
    (2.0, "<NOTE_2>"),
    (3.0, "<NOTE_DOT_2>"),
    (4.0, "<NOTE_1>"),
    (6.0, "<NOTE_DOT_1>"),
)
NOTE_TO_DURATION_INDEX = {
    token: index
    for index, token in enumerate(
        (
            "<NOTE_DOT_1>", "<NOTE_1>", "<NOTE_DOT_2>", "<NOTE_2>",
            "<NOTE_DOT_4>", "<NOTE_4>", "<NOTE_DOT_8>", "<NOTE_8>",
            "<NOTE_DOT_16>", "<NOTE_16>", "<NOTE_DOT_32>", "<NOTE_32>",
        )
    )
}
SUPPORTED_EXTENSIONS = {".abc", ".txt", ".xml", ".musicxml", ".mxl"}
CHINESE_RE = re.compile(r"^[\u3400-\u9fff]+$")


class ScoreImportError(ValueError):
    """An actionable error caused by an unsupported or malformed score."""


def _safe_mxl(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MXL_MEMBERS:
                raise ScoreImportError(
                    f"Compressed MusicXML has {len(infos)} files; the limit is {MAX_MXL_MEMBERS}."
                )
            expanded = 0
            for info in infos:
                member = Path(info.filename.replace("\\", "/"))
                if info.flag_bits & 0x1:
                    raise ScoreImportError("Encrypted .mxl archives are not supported.")
                if member.is_absolute() or ".." in member.parts:
                    raise ScoreImportError("Unsafe path found inside the .mxl archive.")
                expanded += info.file_size
                if expanded > MAX_MXL_EXPANDED_BYTES:
                    raise ScoreImportError("The expanded .mxl file is larger than 20 MiB.")
                if info.file_size and info.compress_size == 0:
                    raise ScoreImportError("Invalid compressed member in the .mxl archive.")
                if info.compress_size and info.file_size / info.compress_size > MAX_MXL_COMPRESSION_RATIO:
                    raise ScoreImportError("Suspicious compression ratio in the .mxl archive.")
    except zipfile.BadZipFile as exc:
        raise ScoreImportError("The uploaded .mxl file is not a valid ZIP archive.") from exc


def _validate_file(path_value: str) -> Tuple[Path, str]:
    path = Path(path_value)
    if not path.is_file():
        raise ScoreImportError("The uploaded score file is no longer available.")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ScoreImportError(
            "Use .abc/.txt for ABC notation or .musicxml/.xml/.mxl for MusicXML."
        )
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ScoreImportError("Score uploads are limited to 5 MiB.")
    if suffix == ".mxl":
        _safe_mxl(path)
    return path, "abc" if suffix in {".abc", ".txt"} else "musicxml"


def _abc_lyric_tokens(text: str) -> List[str]:
    """Tokenize the useful subset of the ABC ``w:`` alignment language."""
    text = re.sub(r"\\\s*$", "", text)
    text = text.replace("\\-", "\u0000").replace("~", " ")
    text = re.sub(r"([_*|])", r" \1 ", text)
    text = text.replace("-", " ").replace("\u0000", "-")
    tokens: List[str] = []
    for token_value in text.split():
        if CHINESE_RE.fullmatch(token_value) and len(token_value) > 1:
            tokens.extend(token_value)
        else:
            tokens.append(token_value)
    return tokens


def _append_abc_lyric_run(verses: List[List[str]], run: Sequence[str]) -> None:
    for verse_index, lyric_line in enumerate(run):
        while len(verses) <= verse_index:
            verses.append([])
        verses[verse_index].extend(_abc_lyric_tokens(lyric_line))


def _extract_abc_part_verses(source: str) -> List[List[List[str]]]:
    """Collect ``w:`` lines per sequential ABC ``V:`` voice block."""
    voices: Dict[str, List[List[str]]] = {"__default__": []}
    voice_order = ["__default__"]
    current_voice = "__default__"
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        voice_match = re.match(r"^\s*V\s*:\s*([^\s]+)", lines[index])
        inline_voice_match = re.match(r"^\s*\[V\s*:\s*([^\]\s]+)\]", lines[index])
        if voice_match or inline_voice_match:
            current_voice = (voice_match or inline_voice_match).group(1)
            if current_voice not in voices:
                voices[current_voice] = []
                voice_order.append(current_voice)
        # ABC distinguishes lower-case w: (note-aligned lyrics) from upper-case
        # W: (free-form words printed after the tune).  Treating W: as aligned
        # silently assigns syllables to the wrong notes in many archive files.
        if re.match(r"^\s*w\s*:", lines[index]):
            run: List[str] = []
            while index < len(lines):
                match = re.match(r"^\s*w\s*:\s*(.*)$", lines[index])
                if not match:
                    break
                run.append(match.group(1))
                index += 1
            _append_abc_lyric_run(voices[current_voice], run)
            continue
        index += 1
    populated = [voices[voice_id] for voice_id in voice_order if voices[voice_id]]
    return populated or [[]]


def _split_abc_works(source: str) -> List[str]:
    lines = source.splitlines()
    starts = [index for index, line in enumerate(lines) if re.match(r"^\s*X\s*:", line)]
    if len(starts) <= 1:
        return [source]
    preamble = lines[:starts[0]]
    works = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        works.append("\n".join([*preamble, *lines[start:end]]))
    return works


def _extract_abc_voice_labels(source: str) -> List[str]:
    labels: Dict[str, str] = {}
    for line in source.splitlines():
        match = re.match(r"^\s*V\s*:\s*([^\s]+)(.*)$", line)
        if not match:
            continue
        voice_id, attributes = match.groups()
        name_match = re.search(r'\b(?:name|nm)\s*=\s*"([^"]+)"', attributes)
        labels.setdefault(voice_id, name_match.group(1) if name_match else f"Voice {voice_id}")
    return list(labels.values())


def _score_list(parsed: stream.Stream) -> List[stream.Stream]:
    if isinstance(parsed, stream.Opus):
        return list(parsed.scores)
    return [parsed]


def _part_list(work: stream.Stream) -> List[stream.Stream]:
    parts = list(getattr(work, "parts", []))
    return parts or [work]


def _measure_label(element: Any) -> str:
    measure = element.getContextByClass(stream.Measure)
    if measure is None:
        return "0"
    suffix = getattr(measure, "numberSuffix", None) or ""
    return f"{measure.number}{suffix}"


def _absolute_offset(element: Any, part: stream.Stream) -> float:
    try:
        return float(element.getOffsetInHierarchy(part))
    except Exception:
        return float(element.offset)


def _resolve_embedded_lyrics(events: List[Dict[str, Any]], verse_ids: Sequence[str]) -> None:
    for verse_id in verse_ids:
        anchor = -1
        previous_pitch: Optional[int] = None
        previous_resolved: Optional[Dict[str, Any]] = None
        for event in events:
            if event["kind"] != "note":
                continue
            lyric = event["lyrics"].get(verse_id)
            tie_type = event.get("tie")
            if lyric is not None and lyric["text"]:
                anchor += 1
                event.setdefault("resolved_lyrics", {})[verse_id] = {
                    "anchor": anchor, "text": lyric["text"],
                }
                previous_resolved = event["resolved_lyrics"][verse_id]
            elif (
                lyric is not None
                or tie_type in {"continue", "stop"}
                and previous_pitch == event.get("pitch")
            ) and previous_resolved:
                event.setdefault("resolved_lyrics", {})[verse_id] = dict(previous_resolved)
            previous_pitch = event.get("pitch")


def _apply_abc_lyrics(events: List[Dict[str, Any]], verses: Sequence[Sequence[str]]) -> List[str]:
    verse_ids: List[str] = []
    pitched = [event for event in events if event["kind"] == "note"]
    for verse_index, tokens in enumerate(verses, start=1):
        verse_id = str(verse_index)
        verse_ids.append(verse_id)
        event_index = 0
        anchor = -1
        anchor_text = ""
        for lyric_token in tokens:
            if lyric_token == "|":
                continue
            if event_index >= len(pitched):
                break
            event = pitched[event_index]
            event_index += 1
            if lyric_token == "*":
                continue
            if lyric_token == "_":
                if anchor >= 0:
                    event.setdefault("resolved_lyrics", {})[verse_id] = {
                        "anchor": anchor, "text": anchor_text,
                    }
                continue
            anchor += 1
            anchor_text = lyric_token
            event["lyrics"][verse_id] = {"text": lyric_token, "syllabic": None}
            event.setdefault("resolved_lyrics", {})[verse_id] = {
                "anchor": anchor, "text": lyric_token,
            }
        # A tie continuation consumes no fresh lyric even when ``_`` was omitted.
        previous: Optional[Dict[str, Any]] = None
        for event in pitched:
            resolved = event.get("resolved_lyrics", {}).get(verse_id)
            if resolved:
                previous = resolved
            elif event.get("tie") in {"continue", "stop"} and previous:
                event.setdefault("resolved_lyrics", {})[verse_id] = dict(previous)
    return verse_ids


def _extract_part(
    part: stream.Stream,
    *,
    work_index: int,
    part_index: int,
    work_title: str,
    abc_verses: Sequence[Sequence[str]],
    abc_part_label: Optional[str] = None,
) -> Dict[str, Any]:
    # music21 represents ABC guitar-chord annotations (for example "C") as
    # zero-duration ChordSymbol objects, and ChordSymbol subclasses Chord.
    # They describe accompaniment harmony rather than notes to be sung.
    elements = [
        element for element in part.recurse().notesAndRests
        if not isinstance(element, harmony.ChordSymbol)
    ]
    elements.sort(key=lambda item: (_absolute_offset(item, part), item.classSortOrder))
    events: List[Dict[str, Any]] = []
    verse_ids = set()
    for element in elements:
        lyrics: Dict[str, Dict[str, Any]] = {}
        for lyric in getattr(element, "lyrics", []):
            verse_id = str(lyric.identifier or lyric.number or 1)
            verse_ids.add(verse_id)
            lyrics[verse_id] = {
                "text": (lyric.text or "").strip(),
                "syllabic": lyric.syllabic,
            }
        if isinstance(element, note.Rest):
            kind, pitch_value, chord_pitches = "rest", 0, []
        elif isinstance(element, chord.Chord):
            kind, pitch_value = "chord", None
            chord_pitches = [int(pitch.midi) for pitch in element.pitches]
        else:
            kind, pitch_value, chord_pitches = "note", int(element.pitch.midi), []
        events.append({
            "kind": kind,
            "pitch": pitch_value,
            "chord_pitches": chord_pitches,
            "quarter_length": float(element.duration.quarterLength),
            "offset": _absolute_offset(element, part),
            "measure": _measure_label(element),
            "tie": getattr(getattr(element, "tie", None), "type", None),
            "lyrics": lyrics,
        })

    if abc_verses:
        verse_ids = set(_apply_abc_lyrics(events, abc_verses))
    else:
        _resolve_embedded_lyrics(events, sorted(verse_ids))

    tempos = []
    for mark in part.recurse().getElementsByClass(tempo.MetronomeMark):
        bpm = mark.getQuarterBPM()
        if bpm:
            tempos.append({
                "bpm": float(bpm),
                "offset": _absolute_offset(mark, part),
                "measure": _measure_label(mark),
            })

    measures = list(dict.fromkeys(event["measure"] for event in events))
    lyric_events = sum(
        bool(event.get("resolved_lyrics")) for event in events if event["kind"] == "note"
    )
    invalid_events = sum(event["kind"] == "chord" or event["quarter_length"] <= 0 for event in events)
    overlaps = _find_overlaps(events)
    part_name = abc_part_label or getattr(part, "partName", None) or f"Part {part_index + 1}"
    key = f"{work_index}:{part_index}"
    return {
        "key": key,
        "label": f"{work_title} — {part_name}",
        "work_index": work_index,
        "part_index": part_index,
        "events": events,
        "measures": measures,
        "verses": sorted(verse_ids),
        "tempos": tempos,
        "stats": {
            "events": len(events),
            "lyric_events": lyric_events,
            "invalid_events": invalid_events,
            "overlaps": len(overlaps),
        },
    }


def _find_overlaps(events: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    sounding = [event for event in events if event["kind"] in {"note", "chord"}]
    sounding.sort(key=lambda event: event["offset"])
    overlaps = []
    for previous, current in zip(sounding, sounding[1:]):
        previous_end = previous["offset"] + previous["quarter_length"]
        if current["offset"] < previous_end - 1e-7:
            overlaps.append((previous, current))
    return overlaps


def parse_score(*, file_path: Optional[str] = None, abc_text: str = "") -> Dict[str, Any]:
    """Parse one uploaded file or pasted ABC string into a neutral score."""
    abc_text = (abc_text or "").strip()
    if file_path:
        path, input_format = _validate_file(file_path)
        source_for_parser: Any = path
        source_bytes = path.read_bytes()
        try:
            abc_source = source_bytes.decode("utf-8-sig") if input_format == "abc" else ""
        except UnicodeDecodeError as exc:
            raise ScoreImportError("ABC files must be UTF-8 encoded.") from exc
    elif abc_text:
        if len(abc_text.encode("utf-8")) > MAX_UPLOAD_BYTES:
            raise ScoreImportError("Pasted ABC notation is limited to 5 MiB.")
        input_format = "abc"
        source_for_parser = abc_text
        abc_source = abc_text
        source_bytes = abc_text.encode("utf-8")
    else:
        raise ScoreImportError("Upload a score file or paste some ABC notation first.")

    try:
        parsed = converter.parse(source_for_parser, format=input_format, forceSource=True)
    except Exception as exc:
        raise ScoreImportError(f"Could not parse the {input_format.upper()} score: {exc}") from exc

    abc_works = _split_abc_works(abc_source) if input_format == "abc" else []
    abc_work_verses = [_extract_abc_part_verses(work) for work in abc_works]
    abc_work_labels = [_extract_abc_voice_labels(work) for work in abc_works]
    parts: List[Dict[str, Any]] = []
    for work_index, work in enumerate(_score_list(parsed)):
        metadata = getattr(work, "metadata", None)
        # MusicXML commonly maps <work-title> to bestTitle/movementName rather
        # than Metadata.title, notably in files exported by MuseScore/music21.
        title = (
            getattr(metadata, "bestTitle", None)
            or getattr(metadata, "title", None)
            or getattr(metadata, "movementName", None)
            or f"Work {work_index + 1}"
        )
        for part_index, part in enumerate(_part_list(work)):
            abc_verses: Sequence[Sequence[str]] = []
            if work_index < len(abc_work_verses):
                voice_verses = abc_work_verses[work_index]
                if part_index < len(voice_verses):
                    abc_verses = voice_verses[part_index]
            parts.append(_extract_part(
                part,
                work_index=work_index,
                part_index=part_index,
                work_title=title,
                abc_verses=abc_verses,
                abc_part_label=(
                    abc_work_labels[work_index][part_index]
                    if work_index < len(abc_work_labels)
                    and part_index < len(abc_work_labels[work_index])
                    else None
                ),
            ))
    parts = [part for part in parts if part["events"]]
    if not parts:
        raise ScoreImportError("The score contains no notes or rests.")

    ranked = sorted(
        parts,
        key=lambda part: (
            part["stats"]["invalid_events"] == 0 and part["stats"]["overlaps"] == 0,
            part["stats"]["lyric_events"] > 0,
            part["stats"]["lyric_events"],
            -part["part_index"],
        ),
        reverse=True,
    )
    parse_warnings = (
        ["Both an upload and pasted ABC were provided; the uploaded file was used."]
        if file_path and abc_text else []
    )
    if input_format == "abc" and re.search(r"^\s*W\s*:", abc_source, flags=re.MULTILINE):
        parse_warnings.append(
            "This ABC contains W: block lyrics, which are not aligned to notes. "
            "Choose ‘Use lyrics textbox’ and paste the Chinese words for your selected measures."
        )
    return {
        "format": input_format,
        "source_id": hashlib.sha256(source_bytes).hexdigest()[:12],
        "parts": parts,
        "default_part": ranked[0]["key"],
        "warnings": parse_warnings,
    }


def get_part(parsed: Dict[str, Any], part_key: str) -> Dict[str, Any]:
    for part in parsed.get("parts", []):
        if part["key"] == part_key:
            return part
    raise ScoreImportError("Choose a valid score part.")


def part_summary(part: Dict[str, Any]) -> str:
    stats = part["stats"]
    lyric_status = f"{len(part['verses'])} embedded lyric line(s)" if part["verses"] else "no embedded lyrics"
    start_measure, end_measure = recommended_measure_range(part)
    recommendation = (
        f" A model-ready range is preselected: measures {start_measure}–{end_measure}."
        if start_measure is not None else
        " No directly compatible monophonic measure was found; choose a melody-only part or simplify the score."
    )
    compatibility = []
    if stats["invalid_events"]:
        compatibility.append(f"{stats['invalid_events']} chord/grace event(s)")
    if stats["overlaps"]:
        compatibility.append(f"{stats['overlaps']} overlap(s)")
    compatibility_text = (
        " Unsupported events outside the selected range are okay. Detected: "
        + ", ".join(compatibility) + "."
        if compatibility else ""
    )
    return (
        f"**{part['label']}** — {stats['events']} events, {len(part['measures'])} measures, "
        f"{lyric_status}.{recommendation}{compatibility_text}"
    )


def _event_has_supported_duration(event: Dict[str, Any]) -> bool:
    value = event["quarter_length"]
    if value <= 0:
        return False
    target, _ = min(NOTE_QUARTER_LENGTHS, key=lambda item: (abs(item[0] - value), item[0]))
    return abs(target - value) / value <= 0.125 + 1e-9


def recommended_measure_range(part: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return the longest early, monophonic range that fits model limits.

    The lyric count cannot be known before external lyrics are supplied, so one
    event is conservatively budgeted as one lyric/rest unit.  Users may extend
    the range when embedded melismas reduce the real unit count.
    """
    measures = part.get("measures", [])
    events_by_measure = {
        measure: [event for event in part["events"] if event["measure"] == measure]
        for measure in measures
    }
    best: Optional[Tuple[int, int, int]] = None
    event_limit = min(MAX_SCORE_EVENTS, MAX_SCORE_WORDS)
    for start_index in range(len(measures)):
        selected: List[Dict[str, Any]] = []
        for end_index in range(start_index, len(measures)):
            measure_events = events_by_measure[measures[end_index]]
            if (
                not measure_events
                or any(event["kind"] == "chord" or not _event_has_supported_duration(event)
                       for event in measure_events)
            ):
                break
            candidate = [*selected, *measure_events]
            if len(candidate) > event_limit or _find_overlaps(candidate):
                break
            selected = candidate
            score = len(selected)
            if best is None or score > best[2]:
                best = (start_index, end_index, score)
    if best is None:
        return None, None
    return measures[best[0]], measures[best[1]]


def _selected_events(part: Dict[str, Any], start_measure: str, end_measure: str) -> List[Dict[str, Any]]:
    measures = part["measures"]
    if start_measure not in measures or end_measure not in measures:
        raise ScoreImportError("Choose valid start and end measures.")
    start = measures.index(start_measure)
    end = measures.index(end_measure)
    if start > end:
        raise ScoreImportError("The start measure must not come after the end measure.")
    selected = [event for event in part["events"] if start <= measures.index(event["measure"]) <= end]
    if len(selected) > MAX_SCORE_EVENTS:
        raise ScoreImportError(
            f"This range has {len(selected)} events; select at most {MAX_SCORE_EVENTS}."
        )
    return selected


def _duration_token(value: float, event: Dict[str, Any], warnings: List[str]) -> str:
    if value <= 0:
        raise ScoreImportError(
            f"Grace/zero-duration event at measure {event['measure']}, offset {event['offset']:.3g} is unsupported."
        )
    target, token_value = min(NOTE_QUARTER_LENGTHS, key=lambda item: (abs(item[0] - value), item[0]))
    relative_error = abs(target - value) / value
    if relative_error > 0.125 + 1e-9:
        raise ScoreImportError(
            f"Duration {value:g} quarter notes at measure {event['measure']} cannot be safely quantized."
        )
    if abs(target - value) > 1e-9:
        warnings.append(
            f"Measure {event['measure']}: quantized {value:g} quarter notes to {target:g} ({token_value})."
        )
    return token_value


def _split_external_lyrics(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if "|" in text:
        return [item.strip() for item in text.split("|") if item.strip()]
    normalized = re.sub(r"(?i)(?<![A-Za-z])SP(?![A-Za-z])", "", text)
    return re.findall(r"[\u3400-\u9fff]", normalized)


def convert_selection(
    parsed: Dict[str, Any],
    *,
    part_key: str,
    verse_id: str,
    start_measure: str,
    end_measure: str,
    external_lyrics: str = "",
) -> Dict[str, Any]:
    """Convert a selected range to editable demo rows and model metadata."""
    part = get_part(parsed, part_key)
    events = _selected_events(part, str(start_measure), str(end_measure))
    overlaps = _find_overlaps(events)
    if overlaps:
        _, current = overlaps[0]
        raise ScoreImportError(
            f"Overlapping/polyphonic notes at measure {current['measure']}, offset {current['offset']:.3g} are unsupported."
        )
    chord_event = next((event for event in events if event["kind"] == "chord"), None)
    if chord_event:
        raise ScoreImportError(
            f"Chord at measure {chord_event['measure']}, offset {chord_event['offset']:.3g} is unsupported; choose a monophonic part."
        )

    use_embedded = verse_id and verse_id != "__external__"
    if use_embedded and verse_id not in part["verses"]:
        raise ScoreImportError("Choose a valid embedded lyric line.")
    external_words = _split_external_lyrics(external_lyrics) if not use_embedded else []
    external_index = 0
    warnings = list(parsed.get("warnings", []))
    words: List[str] = []
    rows: List[Dict[str, Any]] = []
    anchor_to_word: Dict[Any, int] = {}
    previous_word: Optional[int] = None
    previous_pitch: Optional[int] = None

    for event_index, event in enumerate(events):
        token_value = _duration_token(event["quarter_length"], event, warnings)
        if event["kind"] == "rest":
            word_index = len(words)
            words.append("SP")
            pitch_value = 0
            previous_word = None
            previous_pitch = None
        else:
            pitch_value = int(event["pitch"])
            resolved = event.get("resolved_lyrics", {}).get(verse_id) if use_embedded else None
            if resolved:
                anchor_key = (verse_id, resolved["anchor"])
                if anchor_key not in anchor_to_word:
                    anchor_to_word[anchor_key] = len(words)
                    words.append(resolved["text"])
                word_index = anchor_to_word[anchor_key]
            elif event.get("tie") in {"continue", "stop"} and previous_word is not None and previous_pitch == pitch_value:
                word_index = previous_word
            elif use_embedded:
                raise ScoreImportError(
                    f"Missing lyric alignment at measure {event['measure']}, offset {event['offset']:.3g}. "
                    "Fix the lyric extension/tie or choose external lyrics."
                )
            else:
                if external_index >= len(external_words):
                    raise ScoreImportError(
                        "There are fewer external lyric units than pitched note attacks in the selected range."
                    )
                word_index = len(words)
                words.append(external_words[external_index])
                external_index += 1
            previous_word = word_index
            previous_pitch = pitch_value

        occurrence = sum(row["word_index"] == word_index for row in rows)
        rows.append({
            "word": words[word_index],
            "word_index": word_index,
            "uid": (
                f"import-{parsed.get('source_id', 'score')}-{part_key}-"
                f"{start_measure}-{end_measure}-{event_index}-{occurrence}"
            ),
            "pitch": pitch_value,
            "duration_index": NOTE_TO_DURATION_INDEX[token_value],
        })

    if not use_embedded and external_index != len(external_words):
        raise ScoreImportError(
            f"There are {len(external_words)} external lyric units but only {external_index} pitched note attacks."
        )
    if any(word.upper() != "SP" and not CHINESE_RE.fullmatch(word) for word in words):
        raise ScoreImportError("The selected lyric line contains non-Chinese text; this checkpoint supports Chinese only.")
    if len(words) > MAX_SCORE_WORDS:
        raise ScoreImportError(
            f"This range has {len(words)} lyric/rest units; select at most {MAX_SCORE_WORDS}."
        )

    range_start = events[0]["offset"]
    range_end = max(event["offset"] + event["quarter_length"] for event in events)
    all_tempos = sorted(part["tempos"], key=lambda mark: mark["offset"])
    active_tempos = [mark for mark in all_tempos if mark["offset"] <= range_start + 1e-9]
    if active_tempos:
        bpm = int(round(active_tempos[-1]["bpm"]))
    else:
        bpm = 120
        warnings.append("No tempo marking was active at the range start; using 120 BPM.")
    range_tempos = [
        mark for mark in all_tempos
        if range_start - 1e-9 <= mark["offset"] < range_end - 1e-9
    ]
    distinct = {bpm, *(int(round(mark["bpm"])) for mark in range_tempos)}
    if len(distinct) > 1:
        warnings.append(
            f"The range contains tempo changes ({', '.join(map(str, sorted(distinct)))} BPM); "
            f"the demo will use the starting value {bpm} BPM."
        )
    if not 1 <= bpm <= 255:
        raise ScoreImportError(f"Detected BPM {bpm}; VocalRender supports values from 1 to 255.")

    return {
        "lyrics": "|".join(words),
        "rows": rows,
        "bpm": bpm,
        "warnings": warnings,
        "summary": f"Loaded {len(rows)} events and {len(words)} lyric/rest units from measures {start_measure}–{end_measure}.",
    }
