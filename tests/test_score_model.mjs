import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const M = require("../assets/piano_roll/score_model.js");

const Q = M.QUARTER_INDEX;
const E = M.EIGHTH_INDEX;
const D16 = M.DURATION_INDEX_BY_TOKEN["<NOTE_DOT_16>"];

function row(word, wordIndex, pitch, durationIndex) {
  return { word, word_index: wordIndex, uid: `${word}-${wordIndex}-${pitch}`, pitch, duration_index: durationIndex };
}

// SP(8th) 有(d16, 32) 一(4th) 位(16th) 位(16th)  -> a melisma on 位
const SAMPLE = [
  row("SP", 0, 0, E),
  row("有", 1, 60, D16),
  row("一", 2, 62, Q),
  row("位", 3, 64, M.DURATION_INDEX_BY_TOKEN["<NOTE_16>"]),
  row("位", 3, 65, M.DURATION_INDEX_BY_TOKEN["<NOTE_16>"]),
];

test("duration table matches the model vocabulary in tick units", () => {
  assert.equal(M.DURATIONS.length, 12);
  assert.equal(M.DURATIONS[Q].token, "<NOTE_4>");
  assert.equal(M.DURATIONS[Q].ticks, 16);
  assert.equal(M.DURATIONS[0].ticks, 96);
  assert.equal(M.DURATIONS[11].ticks, 2);
  assert.deepEqual(M.DURATIONS.map((d) => d.ticks), [96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2]);
});

test("layout places contiguous events and marks heads, rests and continuations", () => {
  const events = M.layout(SAMPLE);
  assert.deepEqual(events.map((e) => e.start), [0, 8, 14, 30, 34]);
  assert.deepEqual(events.map((e) => e.lyric), ["SP", "有", "一", "位", "-"]);
  assert.deepEqual(events.map((e) => e.isHead), [true, true, true, true, false]);
  assert.equal(events[0].isRest, true);
  assert.equal(M.totalTicks(SAMPLE), 38);
});

test("counts and validate enforce the model limits", () => {
  assert.deepEqual(M.counts(SAMPLE), { events: 5, words: 4 });
  assert.deepEqual(M.validate(SAMPLE, 64), []);
  assert.equal(M.validate([], 64).length, 1);
  assert.ok(M.validate(SAMPLE, 300).some((p) => p.en.includes("Tempo")));
  const many = Array.from({ length: 129 }, (_, i) => row("啊", i, 60, Q));
  const problems = M.validate(many, 64);
  assert.ok(problems.some((p) => p.en.includes("events")));
  assert.ok(problems.some((p) => p.en.includes("lyric units")));
  assert.ok(M.validate([row("hello", 0, 60, Q)], 64).some((p) => p.en.includes("not Chinese")));
});

test("nearestDurationIndex snaps arbitrary tick lengths", () => {
  assert.equal(M.nearestDurationIndex(16), Q);
  assert.equal(M.nearestDurationIndex(17), Q);
  assert.equal(M.nearestDurationIndex(30), M.DURATION_INDEX_BY_TOKEN["<NOTE_2>"]);
  assert.equal(M.nearestDurationIndex(1), 11);
  assert.equal(M.nearestDurationIndex(500), 0);
});

test("setPitch, transpose and setDuration clamp and do not mutate", () => {
  const next = M.setPitch(SAMPLE, 1, 200);
  assert.equal(next[1].pitch, 127);
  assert.equal(SAMPLE[1].pitch, 60);
  const up = M.transpose(SAMPLE, [0, 1, 2], 12);
  assert.deepEqual(up.map((r) => r.pitch), [0, 72, 74, 64, 65]);
  const longer = M.stepDuration(SAMPLE, [2], -1);
  assert.equal(longer[2].duration_index, Q - 1);
  assert.equal(M.setDuration(SAMPLE, 2, 99)[2].duration_index, 11);
});

test("insertNote after a melisma note inserts after the whole group and renumbers", () => {
  const { rows, index } = M.insertNote(SAMPLE, 3, 67, Q, "娘");
  assert.equal(index, 5);
  assert.deepEqual(rows.map((r) => r.word_index), [0, 1, 2, 3, 3, 4]);
  assert.equal(rows[5].word, "娘");
  assert.equal(rows[5].pitch, 67);
});

test("insertNote in the middle shifts later word indices", () => {
  const { rows, index } = M.insertNote(SAMPLE, 1, 65, Q);
  assert.equal(index, 2);
  assert.equal(rows[2].word, M.DEFAULT_LYRIC);
  assert.deepEqual(rows.map((r) => r.word_index), [0, 1, 2, 3, 4, 4]);
  const first = M.insertRest(SAMPLE, -1, E);
  assert.equal(first.index, 0);
  assert.equal(first.rows[0].word, "SP");
  assert.deepEqual(first.rows.map((r) => r.word_index), [0, 1, 2, 3, 4, 4]);
});

test("addMelisma duplicates a note at the end of its group; rests are ignored", () => {
  const { rows, index } = M.addMelisma(SAMPLE, 3);
  assert.equal(index, 5);
  assert.equal(rows.length, 6);
  assert.deepEqual(rows.slice(3).map((r) => r.word_index), [3, 3, 3]);
  assert.equal(rows[5].pitch, 64);
  assert.notEqual(rows[5].uid, rows[3].uid);
  assert.equal(M.addMelisma(SAMPLE, 0).rows, SAMPLE);
});

test("deleteRows removes events and compacts word indices", () => {
  const rows = M.deleteRows(SAMPLE, [1]);
  assert.deepEqual(rows.map((r) => r.word), ["SP", "一", "位", "位"]);
  assert.deepEqual(rows.map((r) => r.word_index), [0, 1, 2, 2]);
  const oneMelisma = M.deleteRows(SAMPLE, [3]);
  assert.deepEqual(M.layout(oneMelisma).map((e) => e.lyric), ["SP", "有", "一", "位"]);
});

test("setLyric with '-' merges into the previous word", () => {
  const { rows, error } = M.setLyric(SAMPLE, 2, "-");
  assert.equal(error, null);
  assert.deepEqual(rows.map((r) => r.word), ["SP", "有", "有", "位", "位"]);
  assert.deepEqual(rows.map((r) => r.word_index), [0, 1, 1, 2, 2]);
  assert.ok(M.setLyric(SAMPLE, 0, "-").error);
  assert.ok(M.setLyric(SAMPLE, 1, "-").error, "cannot continue a rest");
});

test("setLyric with text on a continuation splits a new word", () => {
  const { rows, error } = M.setLyric(SAMPLE, 4, "娘");
  assert.equal(error, null);
  assert.deepEqual(rows.map((r) => r.word), ["SP", "有", "一", "位", "娘"]);
  assert.deepEqual(rows.map((r) => r.word_index), [0, 1, 2, 3, 4]);
  const renamed = M.setLyric(SAMPLE, 3, "姑").rows;
  assert.deepEqual(renamed.map((r) => r.word), ["SP", "有", "一", "姑", "姑"]);
  assert.ok(M.setLyric(SAMPLE, 1, "la").error);
});

test("setLyric SP turns a note into a rest and detaches continuations", () => {
  const { rows } = M.setLyric(SAMPLE, 3, "SP");
  assert.deepEqual(rows.map((r) => r.word), ["SP", "有", "一", "SP", "位"]);
  assert.deepEqual(rows.map((r) => r.word_index), [0, 1, 2, 3, 4]);
  assert.equal(rows[3].pitch, 0);
  const sung = M.setLyric(rows, 3, "位").rows;
  assert.equal(sung[3].pitch, M.DEFAULT_PITCH);
});

test("splitLyrics mirrors the Python splitter", () => {
  assert.deepEqual(M.splitLyrics("我爱唱歌"), ["我", "爱", "唱", "歌"]);
  assert.deepEqual(M.splitLyrics("我 SP 爱"), ["我", "SP", "爱"]);
  assert.deepEqual(M.splitLyrics("我|爱|唱歌"), ["我", "爱", "唱歌"]);
  assert.deepEqual(M.splitLyrics(""), []);
});

test("applyLyrics fills sung words in order and reports counts", () => {
  const result = M.applyLyrics(SAMPLE, "我爱你SP他");
  assert.deepEqual(result.rows.map((r) => r.word), ["SP", "我", "爱", "你", "你"]);
  assert.equal(result.assigned, 3);
  assert.equal(result.available, 3);
  assert.equal(result.provided, 4);
  const short = M.applyLyrics(SAMPLE, "我");
  assert.deepEqual(short.rows.map((r) => r.word), ["SP", "我", "一", "位", "位"]);
});

test("rowsFromLyrics creates quarter notes at middle C with rests", () => {
  const rows = M.rowsFromLyrics("我SP爱");
  assert.deepEqual(rows.map((r) => [r.word, r.pitch, r.duration_index, r.word_index]), [
    ["我", 60, Q, 0],
    ["SP", 0, Q, 1],
    ["爱", 60, Q, 2],
  ]);
  assert.equal(new Set(rows.map((r) => r.uid)).size, 3);
});

test("midiName", () => {
  assert.equal(M.midiName(60), "C4");
  assert.equal(M.midiName(69), "A4");
  assert.equal(M.midiName(0), "SP");
});
