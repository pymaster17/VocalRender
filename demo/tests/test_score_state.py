"""Tests for the piano-roll value helpers in app.py (UI-only mode, no models)."""
import importlib
import pytest


@pytest.fixture(scope="module")
def app():
    # conftest.py sets VOCALRENDER_UI_ONLY=1 and puts demo/ on sys.path.
    return importlib.import_module("app")


def _row(word, word_index, pitch, duration_index, uid=None):
    return {
        "word": word,
        "word_index": word_index,
        "uid": uid or f"{word}-{word_index}-{pitch}",
        "pitch": pitch,
        "duration_index": duration_index,
    }


def test_score_value_renumbers_word_indices(app):
    rows = [_row("我", 3, 60, 5), _row("爱", 7, 62, 5), _row("爱", 7, 64, 5)]
    value = app.score_value(rows, 90)
    assert [r["word_index"] for r in value["rows"]] == [0, 1, 1]
    assert value["bpm"] == 90
    assert value["beats_per_bar"] == 4


def test_rows_from_score_value_validates_and_normalises(app):
    value = {"rows": [_row("sp", 0, 55, 7), _row("我", 1, 60, 5)], "bpm": 64}
    rows = app.rows_from_score_value(value)
    assert rows[0]["word"] == "SP" and rows[0]["pitch"] == 55
    assert rows[1]["word"] == "我"
    with pytest.raises(app.gr.Error):
        app.rows_from_score_value({"rows": [], "bpm": 64})
    with pytest.raises(app.gr.Error):
        app.rows_from_score_value({"rows": [_row("la", 0, 60, 5)], "bpm": 64})
    with pytest.raises(app.gr.Error):
        app.rows_from_score_value({"rows": [_row("我", 0, 200, 5)], "bpm": 64})
    with pytest.raises(app.gr.Error):
        app.rows_from_score_value({"rows": [_row("我", 0, 60, 12)], "bpm": 64})
    with pytest.raises(app.gr.Error):
        app.rows_from_score_value({"rows": [_row("我", i, 60, 5) for i in range(129)], "bpm": 64})


def test_bpm_from_score_value(app):
    assert app.bpm_from_score_value({"bpm": 88.4}) == 88
    with pytest.raises(app.gr.Error):
        app.bpm_from_score_value({"bpm": 0})
    with pytest.raises(app.gr.Error):
        app.bpm_from_score_value({"bpm": "fast"})


def test_create_lyrics_on_empty_roll_makes_quarter_notes(app):
    value, _message = app.create_or_apply_lyrics("我SP爱", {"rows": [], "bpm": 72})
    assert [(r["word"], r["pitch"], r["duration_index"]) for r in value["rows"]] == [
        ("我", 60, 5), ("SP", 0, 5), ("爱", 60, 5),
    ]
    assert value["bpm"] == 72


def test_apply_lyrics_relabels_sung_words_in_order(app):
    existing = app.score_value(
        [_row("SP", 0, 0, 7), _row("a", 1, 60, 5), _row("a", 1, 62, 5), _row("b", 2, 64, 5)],
        64,
    )
    for row in existing["rows"]:
        if row["word"] != "SP":
            row["word"] = "啊"
    value, message = app.create_or_apply_lyrics("我|爱|你", existing)
    assert [r["word"] for r in value["rows"]] == ["SP", "我", "我", "爱"]
    assert "1 extra lyric unit" in message.constructor_args["value"]


def test_generate_from_score_builds_the_model_entry_in_ui_only_mode(app):
    preset = app.SCORE_PRESETS[0]
    value = app.score_value(app._preset_to_rows(preset), preset["bpm"])
    audio, status, prompt = app.generate_from_score(
        value, app.DEFAULT_CKPT_VARIANT, "Alto-1", None, 2.0, 10, 1.0, 2000,
    )
    assert audio == app.VOICE_PRESETS["Alto-1"]
    tokens = prompt.split(" ")
    assert len(tokens) == len(preset["pitches"])
    assert tokens[0] == f"{preset['words'][0]}:{preset['pitches'][0]}:{preset['notes'][0]}"


def test_load_random_preset_round_trips_every_preset(app):
    for preset in app.SCORE_PRESETS:
        value = app.score_value(app._preset_to_rows(preset), preset["bpm"])
        rows = app.rows_from_score_value(value)
        assert [r["pitch"] for r in rows] == preset["pitches"]
        assert [r["word_index"] for r in rows] == preset["pitch2word"]
        assert [app.NOTE_DURATION_OPTIONS[r["duration_index"]][1] for r in rows] == preset["notes"]
