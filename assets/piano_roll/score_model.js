/*
 * Pure score-model logic for the VocalRender piano roll.
 *
 * The score is the same ordered list of rows the Python side consumes:
 *   {word, word_index, uid, pitch, duration_index}
 * Every row is exactly one model event with one of the twelve fixed note
 * values; events are contiguous in time, rests are rows whose word is "SP",
 * and a melisma is a run of consecutive rows sharing one word_index.
 *
 * All functions treat their inputs as immutable and return new arrays.
 * The file has no DOM dependency so it can be unit-tested with `node --test`.
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) {
    module.exports = factory();
  } else {
    root.VocalRenderScoreModel = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // One tick is a 64th note, so every supported value is an integer.
  const TICKS_PER_QUARTER = 16;
  const TICKS_PER_BEAT = TICKS_PER_QUARTER;

  // Index order must match NOTE_DURATION_OPTIONS in app.py (longest first).
  const DURATIONS = [
    { token: "<NOTE_DOT_1>", ticks: 96, en: "Dotted whole", zh: "附点全音符", key: "1." },
    { token: "<NOTE_1>", ticks: 64, en: "Whole", zh: "全音符", key: "1" },
    { token: "<NOTE_DOT_2>", ticks: 48, en: "Dotted half", zh: "附点二分", key: "2." },
    { token: "<NOTE_2>", ticks: 32, en: "Half", zh: "二分音符", key: "2" },
    { token: "<NOTE_DOT_4>", ticks: 24, en: "Dotted quarter", zh: "附点四分", key: "4." },
    { token: "<NOTE_4>", ticks: 16, en: "Quarter", zh: "四分音符", key: "4" },
    { token: "<NOTE_DOT_8>", ticks: 12, en: "Dotted eighth", zh: "附点八分", key: "8." },
    { token: "<NOTE_8>", ticks: 8, en: "Eighth", zh: "八分音符", key: "8" },
    { token: "<NOTE_DOT_16>", ticks: 6, en: "Dotted sixteenth", zh: "附点十六分", key: "16." },
    { token: "<NOTE_16>", ticks: 4, en: "Sixteenth", zh: "十六分音符", key: "16" },
    { token: "<NOTE_DOT_32>", ticks: 3, en: "Dotted thirty-second", zh: "附点三十二分", key: "32." },
    { token: "<NOTE_32>", ticks: 2, en: "Thirty-second", zh: "三十二分音符", key: "32" },
  ];
  const DURATION_INDEX_BY_TOKEN = Object.fromEntries(DURATIONS.map((d, i) => [d.token, i]));
  const QUARTER_INDEX = DURATION_INDEX_BY_TOKEN["<NOTE_4>"];
  const EIGHTH_INDEX = DURATION_INDEX_BY_TOKEN["<NOTE_8>"];

  const MAX_WORDS = 64;
  const MAX_EVENTS = 128;
  const MIN_PITCH = 0;
  const MAX_PITCH = 127;
  const MIN_BPM = 1;
  const MAX_BPM = 255;
  const DEFAULT_PITCH = 60;
  const DEFAULT_LYRIC = "啊";
  const REST = "SP";
  const CHINESE_RE = /^[㐀-鿿]+$/;
  const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

  let uidCounter = 0;
  function newUid(prefix) {
    uidCounter += 1;
    return `${prefix || "roll"}-${Date.now().toString(36)}-${uidCounter}`;
  }

  function isRest(row) {
    return String(row.word || "").toUpperCase() === REST;
  }

  function clampPitch(pitch) {
    return Math.min(MAX_PITCH, Math.max(MIN_PITCH, Math.round(Number(pitch) || 0)));
  }

  function clampDurationIndex(index) {
    return Math.min(DURATIONS.length - 1, Math.max(0, Math.round(Number(index) || 0)));
  }

  function midiName(pitch) {
    if (pitch <= 0) return REST;
    return `${NOTE_NAMES[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
  }

  /** Closest supported duration for an arbitrary tick count. */
  function nearestDurationIndex(ticks) {
    let best = QUARTER_INDEX;
    let bestDiff = Infinity;
    DURATIONS.forEach((d, i) => {
      const diff = Math.abs(d.ticks - ticks);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = i;
      }
    });
    return best;
  }

  /** Reassign word_index so it is contiguous from 0 in order of first appearance. */
  function renumber(rows) {
    const mapping = new Map();
    return rows.map((row) => {
      if (!mapping.has(row.word_index)) mapping.set(row.word_index, mapping.size);
      return { ...row, word_index: mapping.get(row.word_index) };
    });
  }

  /** Index range [start, end) of the rows sharing the word of rows[index]. */
  function groupRange(rows, index) {
    const wordIndex = rows[index].word_index;
    let start = index;
    while (start > 0 && rows[start - 1].word_index === wordIndex) start -= 1;
    let end = index + 1;
    while (end < rows.length && rows[end].word_index === wordIndex) end += 1;
    return [start, end];
  }

  /** Time layout: one entry per row with tick positions and display flags. */
  function layout(rows) {
    let cursor = 0;
    return rows.map((row, index) => {
      const ticks = DURATIONS[clampDurationIndex(row.duration_index)].ticks;
      const entry = {
        index,
        uid: row.uid,
        start: cursor,
        ticks,
        end: cursor + ticks,
        pitch: row.pitch,
        wordIndex: row.word_index,
        word: row.word,
        isRest: isRest(row),
        isHead: index === 0 || rows[index - 1].word_index !== row.word_index,
      };
      entry.lyric = entry.isRest ? REST : entry.isHead ? row.word : "-";
      cursor += ticks;
      return entry;
    });
  }

  function totalTicks(rows) {
    return rows.reduce((sum, row) => sum + DURATIONS[clampDurationIndex(row.duration_index)].ticks, 0);
  }

  function counts(rows) {
    return {
      events: rows.length,
      words: new Set(rows.map((row) => row.word_index)).size,
    };
  }

  /** Returns a list of {en, zh} problems; empty when the score can be generated. */
  function validate(rows, bpm) {
    const problems = [];
    const c = counts(rows);
    if (rows.length === 0) {
      problems.push({ en: "The score is empty.", zh: "乐谱为空。" });
    }
    if (c.events > MAX_EVENTS) {
      problems.push({ en: `Too many events (${c.events}/${MAX_EVENTS}).`, zh: `音符事件过多（${c.events}/${MAX_EVENTS}）。` });
    }
    if (c.words > MAX_WORDS) {
      problems.push({ en: `Too many lyric units (${c.words}/${MAX_WORDS}).`, zh: `歌词单元过多（${c.words}/${MAX_WORDS}）。` });
    }
    const badLyric = rows.find((row) => !isRest(row) && !CHINESE_RE.test(String(row.word || "")));
    if (badLyric) {
      problems.push({ en: `Lyric "${badLyric.word}" is not Chinese.`, zh: `歌词“${badLyric.word}”不是中文。` });
    }
    const bpmValue = Number(bpm);
    if (!Number.isInteger(bpmValue) || bpmValue < MIN_BPM || bpmValue > MAX_BPM) {
      problems.push({ en: `Tempo must be ${MIN_BPM}–${MAX_BPM} BPM.`, zh: `速度须在 ${MIN_BPM}–${MAX_BPM} BPM 之间。` });
    }
    return problems;
  }

  function setPitch(rows, index, pitch) {
    return rows.map((row, i) => (i === index ? { ...row, pitch: clampPitch(pitch) } : row));
  }

  /** Shift pitched (non-rest) rows at the given indices by delta semitones. */
  function transpose(rows, indices, delta) {
    const set = new Set(indices);
    return rows.map((row, i) => (set.has(i) && !isRest(row) ? { ...row, pitch: clampPitch(row.pitch + delta) } : row));
  }

  function setDuration(rows, index, durationIndex) {
    return rows.map((row, i) => (i === index ? { ...row, duration_index: clampDurationIndex(durationIndex) } : row));
  }

  /** Step the duration of the given rows one supported value longer (-1) or shorter (+1). */
  function stepDuration(rows, indices, step) {
    const set = new Set(indices);
    return rows.map((row, i) => (set.has(i) ? { ...row, duration_index: clampDurationIndex(row.duration_index + step) } : row));
  }

  /**
   * Insert a new word (note or rest) after rows[afterIndex]. If afterIndex is
   * inside a melisma group the new word goes after the whole group so that
   * groups stay contiguous. afterIndex = -1 inserts at the beginning.
   * Returns {rows, index} with the index of the inserted row.
   */
  function insertWord(rows, afterIndex, fields) {
    let at = afterIndex + 1;
    if (afterIndex >= 0 && afterIndex < rows.length) at = groupRange(rows, afterIndex)[1];
    at = Math.min(Math.max(at, 0), rows.length);
    const wordIndex = at === 0 ? 0 : rows[at - 1].word_index + 1;
    const row = {
      word: fields.word,
      word_index: wordIndex,
      uid: fields.uid || newUid(),
      pitch: clampPitch(fields.pitch),
      duration_index: clampDurationIndex(fields.duration_index),
    };
    const next = [
      ...rows.slice(0, at),
      row,
      ...rows.slice(at).map((r) => ({ ...r, word_index: r.word_index + 1 })),
    ];
    return { rows: renumber(next), index: at };
  }

  function insertNote(rows, afterIndex, pitch, durationIndex, word) {
    return insertWord(rows, afterIndex, {
      word: word || DEFAULT_LYRIC,
      pitch: pitch === undefined ? DEFAULT_PITCH : pitch,
      duration_index: durationIndex === undefined ? QUARTER_INDEX : durationIndex,
    });
  }

  function insertRest(rows, afterIndex, durationIndex) {
    return insertWord(rows, afterIndex, {
      word: REST,
      pitch: 0,
      duration_index: durationIndex === undefined ? EIGHTH_INDEX : durationIndex,
    });
  }

  /** Duplicate rows[index] as an extra melisma note at the end of its group. */
  function addMelisma(rows, index) {
    if (isRest(rows[index])) return { rows, index };
    const [, end] = groupRange(rows, index);
    const copy = { ...rows[index], uid: newUid("mel") };
    return { rows: [...rows.slice(0, end), copy, ...rows.slice(end)], index: end };
  }

  function deleteRows(rows, indices) {
    const set = new Set(indices);
    return renumber(rows.filter((_, i) => !set.has(i)));
  }

  /** Turn a rest into a sung note (keeps its duration). */
  function restToNote(rows, index, pitch, word) {
    return rows.map((row, i) =>
      i === index ? { ...row, word: word || DEFAULT_LYRIC, pitch: clampPitch(pitch === undefined ? DEFAULT_PITCH : pitch) } : row,
    );
  }

  /**
   * Edit the lyric of one note the way SynthesizerV does:
   *   "-"   attach this note (and the rest of its group) to the previous word
   *   "SP"  turn the note into a rest
   *   text  give this note (and the following continuation notes) a new word
   * Returns {rows, error} where error is {en, zh} or null.
   */
  function setLyric(rows, index, text) {
    const value = String(text || "").trim();
    const row = rows[index];
    const [start, end] = groupRange(rows, index);
    const next = rows.map((r) => ({ ...r }));

    if (value === "-" || value === "－") {
      if (index === 0) return { rows, error: { en: "The first note cannot be a continuation.", zh: "第一个音符不能是延音。" } };
      if (start !== index) return { rows: next, error: null };
      const previous = rows[index - 1];
      if (isRest(previous)) return { rows, error: { en: "A continuation cannot follow a rest.", zh: "延音不能跟在休止符之后。" } };
      if (isRest(row)) return { rows, error: { en: "A rest cannot become a continuation.", zh: "休止符不能改为延音。" } };
      for (let i = start; i < end; i += 1) {
        next[i].word = previous.word;
        next[i].word_index = previous.word_index;
      }
      return { rows: renumber(next), error: null };
    }

    if (value.toUpperCase() === REST) {
      const detached = next[index].word_index + 0.5; // temporary unique index, renumbered below
      next[index].word = REST;
      next[index].pitch = 0;
      next[index].word_index = detached;
      for (let i = index + 1; i < end; i += 1) next[i].word_index = detached + 0.25;
      return { rows: renumber(next), error: null };
    }

    if (!CHINESE_RE.test(value)) {
      return { rows, error: { en: "Lyrics must be Chinese characters, '-' or SP.", zh: "歌词须为中文字符、“-”或 SP。" } };
    }
    if (isRest(row) && next[index].pitch === 0) next[index].pitch = DEFAULT_PITCH;
    if (start === index) {
      for (let i = start; i < end; i += 1) next[i].word = value;
      return { rows: next, error: null };
    }
    const detached = next[index].word_index + 0.5;
    for (let i = index; i < end; i += 1) {
      next[i].word = value;
      next[i].word_index = detached;
    }
    return { rows: renumber(next), error: null };
  }

  /** Split a lyric string into sung units, mirroring app.py's _split_lyrics. */
  function splitLyrics(text) {
    const source = String(text || "").trim();
    if (!source) return [];
    if (source.includes("|")) {
      return source.split("|").map((unit) => unit.trim()).filter(Boolean);
    }
    const normalized = source.replace(/(?<![A-Za-z])SP(?![A-Za-z])/gi, "|SP|");
    return normalized.match(/SP|[㐀-鿿]/gi) || [];
  }

  /**
   * Assign lyric units to the sung (non-rest) words in order. SP units in the
   * text are ignored because rests come from the roll itself.
   * Returns {rows, assigned, available, provided}.
   */
  function applyLyrics(rows, text) {
    const units = splitLyrics(text).filter((unit) => unit.toUpperCase() !== REST);
    const next = rows.map((r) => ({ ...r }));
    let unit = 0;
    let lastWordIndex = null;
    let available = 0;
    for (let i = 0; i < next.length; i += 1) {
      if (isRest(next[i])) continue;
      if (next[i].word_index !== lastWordIndex) {
        lastWordIndex = next[i].word_index;
        available += 1;
        if (unit < units.length) {
          const word = units[unit];
          unit += 1;
          for (let j = i; j < next.length && next[j].word_index === lastWordIndex; j += 1) next[j].word = word;
        }
      }
    }
    return { rows: next, assigned: unit, available, provided: units.length };
  }

  /** Quarter notes at middle C for every unit, like app.py's _score_rows_from_lyrics. */
  function rowsFromLyrics(text) {
    const units = splitLyrics(text);
    return units.map((word, index) => ({
      word,
      word_index: index,
      uid: newUid("lyr"),
      pitch: word.toUpperCase() === REST ? 0 : DEFAULT_PITCH,
      duration_index: QUARTER_INDEX,
    }));
  }

  return {
    TICKS_PER_QUARTER,
    TICKS_PER_BEAT,
    DURATIONS,
    DURATION_INDEX_BY_TOKEN,
    QUARTER_INDEX,
    EIGHTH_INDEX,
    MAX_WORDS,
    MAX_EVENTS,
    MIN_PITCH,
    MAX_PITCH,
    MIN_BPM,
    MAX_BPM,
    DEFAULT_PITCH,
    DEFAULT_LYRIC,
    REST,
    newUid,
    isRest,
    clampPitch,
    clampDurationIndex,
    midiName,
    nearestDurationIndex,
    renumber,
    groupRange,
    layout,
    totalTicks,
    counts,
    validate,
    setPitch,
    transpose,
    setDuration,
    stepDuration,
    insertNote,
    insertRest,
    addMelisma,
    deleteRows,
    restToNote,
    setLyric,
    splitLyrics,
    applyLyrics,
    rowsFromLyrics,
  };
});
