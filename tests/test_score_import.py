from pathlib import Path
import zipfile

import pytest

from vocalrender.utils.score_import import (
    ScoreImportError,
    convert_selection,
    parse_score,
    recommended_measure_range,
)


ABC_SCORE = """\
X:1
T:ABC Vocal
M:4/4
L:1/4
Q:1/4=90
K:C
C-C E z | F G2 |
w: 我 _ 爱 | 你 好
"""


MUSICXML_SCORE = """\
<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>3</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <direction><sound tempo="100"/></direction>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>3</duration><type>quarter</type>
        <lyric number="1"><text>我</text><extend type="start"/></lyric></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>3</duration><type>quarter</type>
        <lyric number="1"><extend type="stop"/></lyric></note>
      <note><rest/><duration>3</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><type>eighth</type>
        <time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>
        <lyric number="1"><text>爱</text></lyric></note>
    </measure>
  </part>
</score-partwise>
"""


def _selection(parsed, verse="1", lyrics=""):
    part = parsed["parts"][0]
    return convert_selection(
        parsed,
        part_key=part["key"],
        verse_id=verse,
        start_measure=part["measures"][0],
        end_measure=part["measures"][-1],
        external_lyrics=lyrics,
    )


def test_abc_text_import_preserves_tie_melisma_rest_and_tempo():
    parsed = parse_score(abc_text=ABC_SCORE)
    result = _selection(parsed)

    assert parsed["format"] == "abc"
    assert result["lyrics"] == "我|爱|SP|你|好"
    assert [row["pitch"] for row in result["rows"]] == [60, 60, 64, 0, 65, 67]
    assert [row["word_index"] for row in result["rows"][:3]] == [0, 0, 1]
    assert result["bpm"] == 90


def test_musicxml_import_reads_extend_and_quantizes_triplet(tmp_path: Path):
    score_path = tmp_path / "score.musicxml"
    score_path.write_text(MUSICXML_SCORE, encoding="utf-8")

    parsed = parse_score(file_path=str(score_path))
    result = _selection(parsed)

    assert result["lyrics"] == "我|SP|爱"
    assert [row["word_index"] for row in result["rows"]] == [0, 0, 1, 2]
    assert result["bpm"] == 100
    assert any("quantized" in warning for warning in result["warnings"])


def test_score_without_embedded_lyrics_uses_exact_external_alignment():
    parsed = parse_score(abc_text="X:1\nM:2/4\nL:1/4\nK:C\nC D|")
    result = _selection(parsed, verse="__external__", lyrics="你|好")
    assert result["lyrics"] == "你|好"
    assert result["bpm"] == 120
    assert any("120 BPM" in warning for warning in result["warnings"])

    with pytest.raises(ScoreImportError, match="fewer external lyric"):
        _selection(parsed, verse="__external__", lyrics="你")


def test_abc_voice_lyrics_are_kept_with_their_parts():
    parsed = parse_score(abc_text="""\
X:1
T:Duo
M:2/4
L:1/4
K:C
V:1 name="Soprano"
C D|
w: 我 爱
V:2 name="Alto"
E F|
w: 你 好
    """)
    assert len(parsed["parts"]) == 2
    assert parsed["parts"][0]["label"].endswith("Soprano")
    assert parsed["parts"][1]["label"].endswith("Alto")
    first_measure = parsed["parts"][0]["measures"][0]
    second_measure = parsed["parts"][1]["measures"][0]
    first = convert_selection(
        parsed, part_key="0:0", verse_id="1", start_measure=first_measure, end_measure=first_measure
    )
    second = convert_selection(
        parsed, part_key="0:1", verse_id="1", start_measure=second_measure, end_measure=second_measure
    )
    assert first["lyrics"] == "我|爱"
    assert second["lyrics"] == "你|好"


def test_abc_collection_lyrics_do_not_leak_between_works():
    parsed = parse_score(abc_text="""\
X:1
T:First
M:1/4
L:1/4
K:C
C|
w: 我
X:2
T:Second
M:1/4
L:1/4
K:C
D|
w: 你
""")
    assert len(parsed["parts"]) == 2
    assert parsed["parts"][0]["events"][0]["resolved_lyrics"]["1"]["text"] == "我"
    assert parsed["parts"][1]["events"][0]["resolved_lyrics"]["1"]["text"] == "你"


def test_chord_is_rejected_with_measure_location():
    parsed = parse_score(abc_text="X:1\nM:1/4\nL:1/4\nK:C\n[CEG]|\nw: 我")
    with pytest.raises(ScoreImportError, match="Chord at measure"):
        _selection(parsed)


def test_abc_chord_symbols_are_ignored_but_sounding_chords_are_rejected():
    parsed = parse_score(abc_text='X:1\nM:2/4\nL:1/4\nK:C\n"C"C "G7"D|')
    part = parsed["parts"][0]
    assert [event["kind"] for event in part["events"]] == ["note", "note"]
    assert part["stats"]["invalid_events"] == 0
    assert _selection(parsed, verse="__external__", lyrics="你|好")["lyrics"] == "你|好"

    sounding_chord = parse_score(abc_text="X:1\nM:1/4\nL:1/4\nK:C\n[CEG]|")
    with pytest.raises(ScoreImportError, match="Chord at measure"):
        _selection(sounding_chord, verse="__external__", lyrics="你")


def test_uppercase_abc_words_are_not_treated_as_note_aligned_lyrics():
    parsed = parse_score(abc_text="X:1\nM:2/4\nL:1/4\nK:C\nC D|\nW: 你 好")
    assert parsed["parts"][0]["verses"] == []
    assert any("W: block lyrics" in warning for warning in parsed["warnings"])
    assert _selection(parsed, verse="__external__", lyrics="你|好")["lyrics"] == "你|好"


def test_musicxml_work_title_is_used_in_part_label(tmp_path: Path):
    score_path = tmp_path / "titled.musicxml"
    score_path.write_text(
        MUSICXML_SCORE.replace(
            '<score-partwise version="4.0">',
            '<score-partwise version="4.0"><work><work-title>茉莉花</work-title></work>',
        ),
        encoding="utf-8",
    )
    parsed = parse_score(file_path=str(score_path))
    assert parsed["parts"][0]["label"].startswith("茉莉花 —")


def test_recommended_range_skips_chords_and_respects_editor_limit():
    notes = " ".join("C" for _ in range(40))
    parsed = parse_score(
        abc_text=f"X:1\nM:none\nL:1/4\nK:C\n[CEG]|{notes}|{notes}|"
    )
    part = parsed["parts"][0]
    start, end = recommended_measure_range(part)
    selected = [
        event for event in part["events"]
        if part["measures"].index(start)
        <= part["measures"].index(event["measure"])
        <= part["measures"].index(end)
    ]
    assert start != part["measures"][0]
    assert len(selected) <= 64
    assert all(event["kind"] != "chord" for event in selected)


def test_overlap_and_unsafe_quantization_are_rejected():
    parsed = parse_score(abc_text="X:1\nM:2/4\nL:1/4\nK:C\nC D|")
    part = parsed["parts"][0]
    part["events"][1]["offset"] = 0.5
    with pytest.raises(ScoreImportError, match="Overlapping/polyphonic"):
        _selection(parsed, verse="__external__", lyrics="你|好")

    part["events"][1]["offset"] = 1.0
    part["events"][0]["quarter_length"] = 0.6
    with pytest.raises(ScoreImportError, match="cannot be safely quantized"):
        _selection(parsed, verse="__external__", lyrics="你|好")


def test_import_range_event_limit_is_enforced():
    notes = " ".join("C" for _ in range(129))
    parsed = parse_score(abc_text=f"X:1\nM:none\nL:1/8\nK:C\n{notes}")
    with pytest.raises(ScoreImportError, match="select at most 128"):
        _selection(parsed, verse="__external__", lyrics="你" * 129)


def test_tempo_at_range_start_wins_over_later_change():
    parsed = parse_score(abc_text="X:1\nM:1/4\nL:1/4\nK:C\nC|D|")
    part = parsed["parts"][0]
    assert len(part["measures"]) == 2
    second_event = part["events"][1]
    part["tempos"] = [
        {"bpm": 90.0, "offset": 0.0, "measure": part["measures"][0]},
        {
            "bpm": 120.0,
            "offset": second_event["offset"] + 0.5,
            "measure": part["measures"][1],
        },
    ]
    result = convert_selection(
        parsed,
        part_key=part["key"],
        verse_id="__external__",
        start_measure=part["measures"][1],
        end_measure=part["measures"][1],
        external_lyrics="你",
    )
    assert result["bpm"] == 90
    assert any("tempo changes" in warning for warning in result["warnings"])


def test_unsupported_extension_is_rejected(tmp_path: Path):
    score_path = tmp_path / "score.pdf"
    score_path.write_bytes(b"not a score")
    with pytest.raises(ScoreImportError, match="Use .abc"):
        parse_score(file_path=str(score_path))


def test_compressed_musicxml_import_and_archive_path_validation(tmp_path: Path):
    container = """\
<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="score.musicxml" media-type="application/vnd.recordare.musicxml+xml"/></rootfiles>
</container>
"""
    mxl_path = tmp_path / "score.mxl"
    with zipfile.ZipFile(mxl_path, "w") as archive:
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("score.musicxml", MUSICXML_SCORE)
    parsed = parse_score(file_path=str(mxl_path))
    assert parsed["format"] == "musicxml"
    assert _selection(parsed)["lyrics"] == "我|SP|爱"

    unsafe_path = tmp_path / "unsafe.mxl"
    with zipfile.ZipFile(unsafe_path, "w") as archive:
        archive.writestr("../score.musicxml", MUSICXML_SCORE)
    with pytest.raises(ScoreImportError, match="Unsafe path"):
        parse_score(file_path=str(unsafe_path))


def test_musicxml_external_entity_is_blocked(tmp_path: Path):
    score_path = tmp_path / "entity.musicxml"
    score_path.write_text(
        """<?xml version="1.0"?>
<!DOCTYPE score-partwise [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>&xxe;</part-name></score-part></part-list>
  <part id="P1"><measure number="1"><note><rest/><duration>1</duration></note></measure></part>
</score-partwise>""",
        encoding="utf-8",
    )
    with pytest.raises(ScoreImportError, match="EntitiesForbidden"):
        parse_score(file_path=str(score_path))
