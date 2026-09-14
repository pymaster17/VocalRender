/*
 * VocalRender piano roll widget. Runs inside gr.HTML's js_on_load with
 * `element`, `props`, `trigger` and `watch` in scope; the score model
 * (VocalRenderScoreModel) is concatenated in front of this file.
 *
 * props.value = {rows, bpm, beats_per_bar}; JS commits a deep copy after every
 * completed edit, Python replaces it through watch("value").
 */
(function () {
  "use strict";
  const M = VocalRenderScoreModel;
  const root = element.querySelector(".vr-roll");
  if (!root) return;

  const lang = () => props.language === "en" ? "en" : "zh";
  const t = (zh, en) => lang() === "en" ? en : zh;
  function localize() {
    root.querySelectorAll("[data-zh][data-en]").forEach(node => { node.textContent = node.dataset[lang()]; });
    el.play.textContent = state.playing ? t("⏸ 暂停", "⏸ Pause") : t("▶ 试听旋律", "▶ Preview melody");
    buildDurationOptions();
    render();
  }
  function highlightLyrics() {
    const targets = new Set(props.lyric_targets || []);
    el.notes.querySelectorAll(".vr-note").forEach(note => note.classList.toggle("lyric-missing", targets.has(note.dataset.uid)));
  }

  const ROW_H = 16;
  const TOP_PITCH = 108;
  const BOTTOM_PITCH = 21;
  const ROWS = TOP_PITCH - BOTTOM_PITCH + 1;
  const GRID_H = ROWS * ROW_H;
  const ZOOM_LEVELS = [1.5, 2, 3, 4, 6, 8];
  // Sixteenth note. Only a visual sub-division and a seek snap: note lengths
  // still snap to the twelve model values, the shortest of which is a 32nd.
  const SUB_GRID_TICKS = M.TICKS_PER_QUARTER / 4;
  const TAIL_TICKS = 8 * M.TICKS_PER_QUARTER;
  const BLACK = new Set([1, 3, 6, 8, 10]);
  /* SMuFL "Metronome marks" (U+ECA0..), not the Unicode Musical Symbols block.
   * Both exist in Bravura, but the Unicode ones are the staff glyphs: their
   * stems run 0.875em above the baseline, meant to be placed against a staff.
   * The metronome cuts are the same notes drawn for running text -- 0.69em
   * above the baseline -- and metAugmentationDot is the dot that matches them. */
  const GLYPHS = {
    1: "\uECA2",   // metNoteWhole
    2: "\uECA3",   // metNoteHalfUp
    4: "\uECA5",   // metNoteQuarterUp
    8: "\uECA7",   // metNote8thUp
    16: "\uECA9",  // metNote16thUp
    32: "\uECAB",  // metNote32ndUp
  };
  const AUGMENTATION_DOT = "\uECB7";

  const el = {
    toolbar: root.querySelector(".vr-toolbar"),
    play: root.querySelector(".vr-play"),
    bpm: root.querySelector(".vr-bpm"),
    meter: root.querySelector(".vr-meter"),
    inspector: root.querySelector(".vr-inspector"),
    lyric: root.querySelector(".vr-lyric"),
    pitch: root.querySelector(".vr-pitch"),
    pitchName: root.querySelector(".vr-pitch-name"),
    duration: root.querySelector(".vr-duration"),
    glyph: root.querySelector(".vr-duration-glyph"),
    ruler: root.querySelector(".vr-ruler"),
    bars: root.querySelector(".vr-bars"),
    cursor: root.querySelector(".vr-cursor"),
    hover: root.querySelector(".vr-hover"),
    time: root.querySelector(".vr-time"),
    rests: root.querySelector(".vr-rests"),
    keys: root.querySelector(".vr-keys"),
    gridWrap: root.querySelector(".vr-grid-wrap"),
    grid: root.querySelector(".vr-grid"),
    restcols: root.querySelector(".vr-restcols"),
    notes: root.querySelector(".vr-notes"),
    snaps: root.querySelector(".vr-snaps"),
    playhead: root.querySelector(".vr-playhead"),
    end: root.querySelector(".vr-end"),
    counts: root.querySelector(".vr-counts"),
    message: root.querySelector(".vr-message"),
    undo: root.querySelector('[data-action="undo"]'),
    redo: root.querySelector('[data-action="redo"]'),
    melisma: root.querySelector('[data-action="melisma"]'),
    del: root.querySelector('[data-action="delete"]'),
  };

  const state = {
    rows: [],
    bpm: 64,
    beatsPerBar: 4,
    selected: new Set(),
    tool: "select",
    zoom: 4, // index into ZOOM_LEVELS (6 px per 64th note)
    undo: [],
    redo: [],
    drag: null,
    audio: null,
    // Transport position in ticks. It outlives playback: pausing writes the
    // current playhead back here, so play resumes where it stopped.
    position: 0,
    playing: null,
    message: null,
    lastClick: null,
  };

  // ------------------------------------------------------------------ utils
  const clone = (v) => JSON.parse(JSON.stringify(v));
  const ppt = () => ZOOM_LEVELS[state.zoom];
  const pitchTop = (pitch) => (TOP_PITCH - Math.min(TOP_PITCH, Math.max(BOTTOM_PITCH, pitch))) * ROW_H;
  const pitchFromY = (y) => Math.max(BOTTOM_PITCH, Math.min(TOP_PITCH, TOP_PITCH - Math.floor(y / ROW_H)));
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const selectedIndices = () => state.rows.map((r, i) => (state.selected.has(r.uid) ? i : -1)).filter((i) => i >= 0);
  const firstSelected = () => { const s = selectedIndices(); return s.length ? s[0] : -1; };
  const lastSelected = () => { const s = selectedIndices(); return s.length ? s[s.length - 1] : -1; };
  const isDark = () => !!(document.querySelector(".dark") || (document.body && document.body.classList.contains("dark")));

  function normalizeValue(value) {
    const v = value && typeof value === "object" ? value : {};
    return {
      rows: Array.isArray(v.rows) ? clone(v.rows) : [],
      bpm: Number.isFinite(Number(v.bpm)) ? Number(v.bpm) : 64,
      beatsPerBar: [2, 3, 4, 6].includes(Number(v.beats_per_bar)) ? Number(v.beats_per_bar) : 4,
    };
  }

  function currentValue() {
    return { rows: clone(state.rows), bpm: state.bpm, beats_per_bar: state.beatsPerBar };
  }

  /** Push the current state to the undo stack and publish it to Gradio. */
  function commit(nextRows, extra) {
    // The oscillators for the whole take are scheduled up front, so any edit
    // invalidates them. Pause rather than stop: the playhead stays put.
    if (state.playing) pausePlayback();
    const before = currentValue();
    state.undo.push(before);
    if (state.undo.length > 100) state.undo.shift();
    state.redo = [];
    if (nextRows) state.rows = nextRows;
    if (extra && extra.bpm !== undefined) state.bpm = extra.bpm;
    if (extra && extra.beatsPerBar !== undefined) state.beatsPerBar = extra.beatsPerBar;
    pruneSelection();
    publish();
    render();
  }

  function publish() {
    props.value = currentValue();
  }

  function pruneSelection() {
    const uids = new Set(state.rows.map((r) => r.uid));
    for (const uid of Array.from(state.selected)) if (!uids.has(uid)) state.selected.delete(uid);
  }

  function setMessage(problem, ok) {
    state.message = problem ? (problem.zh ? t(problem.zh, problem.en) : String(problem)) : null;
    el.message.textContent = state.message || "";
    el.message.classList.toggle("ok", !!ok);
  }

  function loadFromProps() {
    const v = normalizeValue(props.value);
    state.rows = v.rows;
    state.bpm = v.bpm;
    state.beatsPerBar = v.beatsPerBar;
    state.selected.clear();
    state.undo = [];
    state.redo = [];
    stopPlayback();
    setMessage(null);
    render();
    scrollToNotes();
  }

  // --------------------------------------------------------------- rendering
  function buildKeys() {
    const parts = [];
    for (let pitch = TOP_PITCH; pitch >= BOTTOM_PITCH; pitch -= 1) {
      const black = BLACK.has(pitch % 12);
      const isC = pitch % 12 === 0;
      parts.push(
        `<div class="vr-key ${black ? "black" : "white"}${isC ? " c" : ""}" data-pitch="${pitch}" style="top:${pitchTop(pitch)}px">${isC || !black ? M.midiName(pitch) : ""}</div>`,
      );
    }
    el.keys.innerHTML = parts.join("");
    el.keys.style.height = `${GRID_H}px`;
  }

  function gridBackground() {
    // Octave row shading (top row of each 12-row block is C).
    const stops = [];
    for (let i = 0; i < 12; i += 1) {
      const pitchClass = (12 - i) % 12; // C, B, A#, ... C#
      const color = BLACK.has(pitchClass) ? "var(--vr-black-row)" : "var(--vr-white-row)";
      stops.push(`${color} ${i * ROW_H}px ${(i + 1) * ROW_H}px`);
    }
    const rowsLayer = `linear-gradient(to bottom, ${stops.join(", ")})`;
    const beatPx = M.TICKS_PER_BEAT * ppt();
    const barPx = beatPx * state.beatsPerBar;
    const subPx = SUB_GRID_TICKS * ppt();
    const beats = `repeating-linear-gradient(to right, var(--vr-beat-line) 0 1px, transparent 1px ${beatPx}px)`;
    const bars = `repeating-linear-gradient(to right, var(--vr-bar-line) 0 1px, transparent 1px ${barPx}px)`;
    const sub = subGridImage(subPx);
    const octaveLines = `repeating-linear-gradient(to bottom, var(--vr-c-line) 0 1px, transparent 1px ${12 * ROW_H}px)`;
    const rowLines = `repeating-linear-gradient(to bottom, var(--vr-beat-line) 0 1px, transparent 1px ${ROW_H}px)`;
    el.grid.style.backgroundImage = [bars, beats, sub, octaveLines, rowLines, rowsLayer].join(", ");
    el.grid.style.backgroundSize = `${barPx}px 100%, ${beatPx}px 100%, ${subPx}px 8px, 100% ${12 * ROW_H}px, 100% ${ROW_H}px, 100% ${12 * ROW_H}px`;
    el.grid.style.backgroundPosition = `-1px 0, -1px 0, -1px 0, 0 ${ROW_H - 1}px, 0 ${ROW_H - 1}px, 0 0`;
  }

  /* Dashed sixteenth-note lines. A repeating-linear-gradient can only draw
   * solid vertical lines, so the dash pattern comes from a one-tile SVG that
   * the browser repeats. Recomputed only when the zoom or the theme changes. */
  let subGridCache = { key: "", image: "none" };
  function subGridImage(subPx) {
    const key = `${subPx}|${isDark()}`;
    if (subGridCache.key === key) return subGridCache.image;
    let image = "none";
    if (subPx >= 8) {
      const color = getComputedStyle(root).getPropertyValue("--vr-sub-line").trim() || "rgba(0,0,0,.05)";
      const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${subPx}' height='8'><rect x='0' y='0' width='1' height='4' fill='${color}'/></svg>`;
      image = `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
    }
    subGridCache = { key, image };
    return image;
  }

  const handleWidth = (w) => (w >= 24 ? 7 : Math.max(2, Math.floor(w / 3)));

  /* The twelve legal lengths are 2^n quarters and their dotted (x1.5) forms,
   * so they thin out geometrically: near the note head they are a few pixels
   * apart, a whole note away they are half a bar apart. Resizing therefore
   * feels like the resolution degrades as you drag, which looks like a bug
   * unless you can see where the model's vocabulary actually lies. Draw every
   * legal end position on the handle, so the spacing reads as the score
   * alphabet it is rather than as sloppy snapping. */
  function showSnapMarks(index) {
    const ev = M.layout(state.rows)[index];
    if (!ev || ev.isRest === undefined) return;
    const scale = ppt();
    const top = pitchTop(ev.pitch) + ROW_H + 3;
    const parts = [
      `<div class="vr-snap-rail" style="left:${ev.start * scale}px;top:${top + 5}px;width:${M.DURATIONS[0].ticks * scale}px"></div>`,
    ];
    M.DURATIONS.forEach((d, i) => {
      const active = i === M.clampDurationIndex(state.rows[index].duration_index);
      parts.push(
        `<div class="vr-snap${d.key.endsWith(".") ? " dotted" : ""}${active ? " active" : ""}" data-i="${i}" style="left:${(ev.start + d.ticks) * scale}px;top:${top}px;height:${active ? 16 : 11}px"></div>`,
      );
      if (active) {
        parts.push(
          // Above the note: the band below it belongs to the next pitch row.
          `<div class="vr-snap-tip" style="left:${(ev.start + d.ticks) * scale}px;top:${Math.max(0, pitchTop(ev.pitch) - 19)}px">${escapeHtml(t(d.zh, d.en))}</div>`,
        );
      }
    });
    el.snaps.innerHTML = parts.join("");
    el.snaps.classList.add("on");
  }

  function hideSnapMarks() {
    if (state.drag && state.drag.kind === "resize") return;
    el.snaps.classList.remove("on");
    el.snaps.innerHTML = "";
  }

  function render() {
    root.classList.toggle("vr-dark", isDark());
    const events = M.layout(state.rows);
    const total = M.totalTicks(state.rows);
    const scale = ppt();
    const width = Math.max((total + TAIL_TICKS) * scale, el.gridWrap.clientWidth || 600);
    el.grid.style.width = `${width}px`;
    el.grid.style.height = `${GRID_H}px`;
    el.ruler.style.width = `${width}px`;
    el.grid.className = `vr-grid ${state.tool}`;
    gridBackground();

    // Bar numbers.
    const barTicks = M.TICKS_PER_BEAT * state.beatsPerBar;
    const barCount = Math.ceil(width / (barTicks * scale)) + 1;
    const bars = [];
    for (let bar = 0; bar < barCount; bar += 1) bars.push(`<div class="vr-bar" style="left:${bar * barTicks * scale}px">${bar + 1}</div>`);
    el.bars.innerHTML = bars.join("");

    // Notes, rests and rest columns.
    const notes = [];
    const rests = [];
    const cols = [];
    for (const ev of events) {
      const left = ev.start * scale;
      const w = Math.max(ev.ticks * scale - 1, 3);
      const sel = state.selected.has(ev.uid) ? " selected" : "";
      if (ev.isRest) {
        rests.push(`<div class="vr-rest${sel}" data-index="${ev.index}" data-uid="${escapeHtml(ev.uid)}" style="left:${left}px;width:${w}px" title="休止 Rest ${M.DURATIONS[state.rows[ev.index].duration_index].zh}">SP<div class="vr-handle" data-index="${ev.index}" style="width:${handleWidth(w)}px"></div></div>`);
        cols.push(`<div class="vr-restcol" style="left:${left}px;width:${ev.ticks * scale}px"></div>`);
      } else {
        const d = M.DURATIONS[M.clampDurationIndex(state.rows[ev.index].duration_index)];
        const title = `${escapeHtml(ev.word)} · ${M.midiName(ev.pitch)} (${ev.pitch}) · ${t(d.zh, d.en)}`;
        notes.push(
          `<div class="vr-note${ev.isHead ? "" : " continuation"}${sel}" data-index="${ev.index}" data-uid="${escapeHtml(ev.uid)}" style="left:${left}px;top:${pitchTop(ev.pitch)}px;width:${w}px" title="${title}">${escapeHtml(ev.lyric)}<div class="vr-handle" data-index="${ev.index}" style="width:${handleWidth(w)}px"></div></div>`,
        );
      }
    }
    el.notes.innerHTML = notes.join("");
    highlightLyrics();
    el.rests.innerHTML = rests.join("");
    el.restcols.innerHTML = cols.join("");
    el.end.style.left = `${total * scale}px`;

    // Keyboard highlight for selected pitches.
    const selectedPitches = new Set(selectedIndices().map((i) => state.rows[i].pitch));
    el.keys.querySelectorAll(".vr-key").forEach((key) => key.classList.toggle("highlight", selectedPitches.has(Number(key.dataset.pitch))));

    renderInspector();
    renderFooter();
    if (!state.playing) {
      state.position = Math.max(0, Math.min(state.position, total));
      renderTransport(state.position);
    }
    el.bpm.value = state.bpm;
    el.meter.value = String(state.beatsPerBar);
    el.undo.disabled = state.undo.length === 0;
    el.redo.disabled = state.redo.length === 0;
    const sel = selectedIndices();
    el.del.disabled = sel.length === 0;
    el.melisma.disabled = !sel.some((i) => !M.isRest(state.rows[i]));
  }

  function renderInspector() {
    const index = firstSelected();
    el.inspector.classList.toggle("empty", index < 0);
    if (index < 0) return;
    const row = state.rows[index];
    const events = M.layout(state.rows);
    if (document.activeElement !== el.lyric) el.lyric.value = events[index].lyric;
    if (document.activeElement !== el.pitch) el.pitch.value = row.pitch;
    el.pitch.disabled = M.isRest(row);
    el.lyric.disabled = false;
    el.pitchName.textContent = M.midiName(row.pitch);
    el.duration.value = String(row.duration_index);
    const d = M.DURATIONS[M.clampDurationIndex(row.duration_index)];
    const denominator = Number(d.key.replace(".", ""));
    el.glyph.textContent = GLYPHS[denominator] + (d.key.endsWith(".") ? AUGMENTATION_DOT : "");
  }

  function renderFooter() {
    const c = M.counts(state.rows);
    el.counts.textContent = `${t("音符/休止", "Notes/rests")} ${c.events}/${M.MAX_EVENTS} · ${t("歌词单元", "Lyric units")} ${c.words}/${M.MAX_WORDS} · ${formatSeconds(M.totalTicks(state.rows))}`;
    const problems = M.validate(state.rows, state.bpm);
    el.counts.classList.toggle("over", problems.some((p) => p.en.includes("Too many")));
    if (!state.message) {
      const p = problems.find((x) => !x.en.includes("empty"));
      el.message.textContent = p ? t(p.zh, p.en) : "";
      el.message.classList.remove("ok");
    }
  }

  function formatSeconds(ticks) {
    const seconds = (ticks / M.TICKS_PER_QUARTER) * (60 / Math.max(1, state.bpm));
    return `${seconds.toFixed(1)}s`;
  }

  function scrollToNotes() {
    const pitched = state.rows.filter((r) => !M.isRest(r)).map((r) => r.pitch);
    const center = pitched.length ? pitched.reduce((a, b) => a + b, 0) / pitched.length : 64;
    el.gridWrap.scrollTop = Math.max(0, pitchTop(Math.round(center)) - el.gridWrap.clientHeight / 2);
    el.gridWrap.scrollLeft = 0;
    syncScroll();
  }

  function syncScroll() {
    el.ruler.style.transform = `translateX(${-el.gridWrap.scrollLeft}px)`;
    el.keys.style.transform = `translateY(${-el.gridWrap.scrollTop}px)`;
  }

  function ensureVisible(index) {
    if (index < 0) return;
    const ev = M.layout(state.rows)[index];
    const left = ev.start * ppt();
    const right = ev.end * ppt();
    if (left < el.gridWrap.scrollLeft) el.gridWrap.scrollLeft = Math.max(0, left - 40);
    else if (right > el.gridWrap.scrollLeft + el.gridWrap.clientWidth) el.gridWrap.scrollLeft = right - el.gridWrap.clientWidth + 40;
    if (!ev.isRest) {
      const top = pitchTop(ev.pitch);
      if (top < el.gridWrap.scrollTop || top + ROW_H > el.gridWrap.scrollTop + el.gridWrap.clientHeight) {
        el.gridWrap.scrollTop = Math.max(0, top - el.gridWrap.clientHeight / 2);
      }
    }
    syncScroll();
  }

  // ------------------------------------------------------------- selection
  function select(index, additive) {
    if (!additive) state.selected.clear();
    // Keeps the old "play from the selected note" behaviour, but now the
    // playhead shows where that is. Never disturbs a running take.
    if (!state.playing && !additive && index >= 0 && index < state.rows.length) {
      state.position = M.layout(state.rows)[index].start;
    }
    if (index >= 0 && index < state.rows.length) {
      const uid = state.rows[index].uid;
      if (additive && state.selected.has(uid)) state.selected.delete(uid);
      else state.selected.add(uid);
    }
    render();
  }

  function selectRange(from, to) {
    const [a, b] = from < to ? [from, to] : [to, from];
    for (let i = a; i <= b; i += 1) state.selected.add(state.rows[i].uid);
    render();
  }

  // ---------------------------------------------------------------- editing
  function actionInsertNote() {
    const after = lastSelected() >= 0 ? lastSelected() : state.rows.length - 1;
    const ref = after >= 0 ? state.rows[after] : null;
    const pitch = ref && !M.isRest(ref) ? ref.pitch : M.DEFAULT_PITCH;
    const duration = ref ? ref.duration_index : M.QUARTER_INDEX;
    const { rows, index } = M.insertNote(state.rows, after, pitch, duration);
    state.selected.clear();
    state.selected.add(rows[index].uid);
    commit(rows);
    ensureVisible(index);
  }

  function actionInsertRest() {
    const after = lastSelected() >= 0 ? lastSelected() : state.rows.length - 1;
    const { rows, index } = M.insertRest(state.rows, after, M.EIGHTH_INDEX);
    state.selected.clear();
    state.selected.add(rows[index].uid);
    commit(rows);
    ensureVisible(index);
  }

  function actionMelisma() {
    const index = lastSelected();
    if (index < 0 || M.isRest(state.rows[index])) return;
    const { rows, index: newIndex } = M.addMelisma(state.rows, index);
    state.selected.clear();
    state.selected.add(rows[newIndex].uid);
    commit(rows);
    ensureVisible(newIndex);
  }

  function actionDelete(indices) {
    const targets = indices || selectedIndices();
    if (!targets.length) return;
    const next = M.deleteRows(state.rows, targets);
    const focus = Math.min(targets[0], next.length - 1);
    state.selected.clear();
    if (focus >= 0) state.selected.add(next[focus].uid);
    commit(next);
  }

  function actionTranspose(delta) {
    const indices = selectedIndices();
    if (!indices.length) return;
    commit(M.transpose(state.rows, indices, delta));
    ensureVisible(indices[0]);
  }

  function actionStepDuration(step) {
    const indices = selectedIndices();
    if (!indices.length) return;
    commit(M.stepDuration(state.rows, indices, step));
  }

  function actionSetLyric(index, text) {
    const { rows, error } = M.setLyric(state.rows, index, text);
    if (error) {
      setMessage(error);
      render();
      return false;
    }
    setMessage(null);
    commit(rows);
    return true;
  }

  function actionUndo() {
    if (!state.undo.length) return;
    state.redo.push(currentValue());
    const v = normalizeValue(state.undo.pop());
    state.rows = v.rows; state.bpm = v.bpm; state.beatsPerBar = v.beatsPerBar;
    pruneSelection();
    publish();
    render();
  }

  function actionRedo() {
    if (!state.redo.length) return;
    state.undo.push(currentValue());
    const v = normalizeValue(state.redo.pop());
    state.rows = v.rows; state.bpm = v.bpm; state.beatsPerBar = v.beatsPerBar;
    pruneSelection();
    publish();
    render();
  }

  function setTool(tool) {
    state.tool = tool;
    root.querySelectorAll(".vr-tool").forEach((btn) => btn.classList.toggle("active", btn.dataset.tool === tool));
    render();
  }

  function setZoom(delta) {
    const anchorTick = el.gridWrap.scrollLeft / ppt();
    state.zoom = Math.max(0, Math.min(ZOOM_LEVELS.length - 1, state.zoom + delta));
    render();
    el.gridWrap.scrollLeft = anchorTick * ppt();
    syncScroll();
  }

  /** Pencil: click on a rest converts it, on a note inserts after it, past the end appends. */
  function pencilAt(tick, pitch) {
    const events = M.layout(state.rows);
    const hit = events.find((ev) => tick >= ev.start && tick < ev.end);
    const duration = Number(el.duration.value) || M.QUARTER_INDEX;
    let rows;
    let index;
    if (!hit) {
      ({ rows, index } = M.insertNote(state.rows, state.rows.length - 1, pitch, duration));
    } else if (hit.isRest) {
      rows = M.restToNote(state.rows, hit.index, pitch);
      index = hit.index;
    } else {
      ({ rows, index } = M.insertNote(state.rows, hit.index, pitch, duration));
    }
    state.selected.clear();
    state.selected.add(rows[index].uid);
    commit(rows);
  }

  // --------------------------------------------------------- lyric editing
  function openLyricEditor(index) {
    closeLyricEditor();
    const ev = M.layout(state.rows)[index];
    const input = document.createElement("input");
    input.type = "text";
    input.className = "vr-lyric-editor";
    input.maxLength = 8;
    input.value = ev.lyric;
    input.style.left = `${ev.start * ppt()}px`;
    input.style.top = `${ev.isRest ? 0 : pitchTop(ev.pitch)}px`;
    input.style.width = `${Math.max(ev.ticks * ppt(), 48)}px`;
    let done = false;
    const finish = (apply) => {
      if (done) return;
      done = true;
      const text = input.value;
      // Removing a focused input fires blur synchronously, which re-enters finish().
      try { if (input.isConnected) input.remove(); } catch (err) { /* already detached */ }
      root.focus({ preventScroll: true });
      if (apply && text !== ev.lyric) actionSetLyric(index, text);
    };
    input.vrFinish = finish;
    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); finish(true); }
      if (e.key === "Escape") { e.preventDefault(); finish(false); }
    });
    input.addEventListener("blur", () => finish(true));
    el.grid.appendChild(input);
    input.focus();
    input.select();
  }

  function closeLyricEditor(apply) {
    const existing = el.grid.querySelector(".vr-lyric-editor");
    if (!existing) return;
    if (existing.vrFinish) existing.vrFinish(apply !== false);
    else existing.remove();
  }

  // ---------------------------------------------------------------- pointer
  function gridPoint(e) {
    const rect = el.grid.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function onPointerDown(e) {
    if (e.button !== 0) return;
    const handle = e.target.closest(".vr-handle");
    const note = e.target.closest(".vr-note, .vr-rest");
    const inGrid = el.grid.contains(e.target) || el.rests.contains(e.target);
    if (!inGrid) return;
    closeLyricEditor();
    root.focus({ preventScroll: true });

    if (note) {
      const index = Number(note.dataset.index);
      if (state.tool === "eraser") { actionDelete([index]); return; }
      // Notes are re-rendered on selection, so native dblclick never fires; detect it here.
      const now = Date.now();
      const last = state.lastClick;
      const isDouble = !!last && last.index === index && now - last.time < 400
        && Math.abs(last.x - e.clientX) < 4 && Math.abs(last.y - e.clientY) < 4;
      state.lastClick = isDouble ? null : { index, time: now, x: e.clientX, y: e.clientY };
      if (isDouble && state.tool === "select") {
        select(index, false);
        openLyricEditor(index);
        e.preventDefault();
        return;
      }
      if (handle && state.tool === "select") {
        if (!state.selected.has(state.rows[index].uid)) select(index, e.shiftKey);
        startResize(index, e);
        e.preventDefault();
        return;
      }
      if (state.tool === "pencil" && note.classList.contains("vr-note")) {
        const p = gridPoint(e);
        pencilAt(Math.floor(p.x / ppt()), pitchFromY(p.y));
        return;
      }
      if (e.shiftKey && firstSelected() >= 0) selectRange(firstSelected(), index);
      else if (e.ctrlKey || e.metaKey) select(index, true);
      else if (!state.selected.has(state.rows[index].uid)) select(index, false);
      else render();
      if (note.classList.contains("vr-note")) startMove(e);
      e.preventDefault();
      return;
    }

    // Empty grid area.
    const p = gridPoint(e);
    if (state.tool === "pencil") {
      pencilAt(Math.floor(p.x / ppt()), pitchFromY(p.y));
    } else if (state.tool === "select") {
      const tick = Math.floor(p.x / ppt());
      const hit = M.layout(state.rows).find((ev) => ev.isRest && tick >= ev.start && tick < ev.end);
      if (hit) select(hit.index, e.ctrlKey || e.metaKey);
      else if (state.selected.size) { state.selected.clear(); render(); }
    }
  }

  function startMove(e) {
    const indices = selectedIndices().filter((i) => !M.isRest(state.rows[i]));
    if (!indices.length) return;
    state.drag = { kind: "move", startY: e.clientY, indices, base: state.rows, delta: 0 };
    attachDragListeners();
  }

  function startResize(index, e) {
    const ev = M.layout(state.rows)[index];
    state.drag = { kind: "resize", index, startTick: ev.start, base: state.rows, current: state.rows[index].duration_index };
    showSnapMarks(index);
    attachDragListeners();
  }

  function attachDragListeners() {
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp, { once: true });
  }

  function onPointerMove(e) {
    const d = state.drag;
    if (!d) return;
    if (d.kind === "move") {
      const delta = -Math.round((e.clientY - d.startY) / ROW_H);
      if (delta === d.delta) return;
      d.delta = delta;
      state.rows = M.transpose(d.base, d.indices, delta);
      render();
      el.notes.querySelectorAll(".vr-note.selected").forEach((n) => n.classList.add("dragging"));
    } else if (d.kind === "resize") {
      const p = gridPoint(e);
      const ticks = Math.max(1, p.x / ppt() - d.startTick);
      const index = M.nearestDurationIndex(ticks);
      if (index === d.current) return;
      d.current = index;
      state.rows = M.setDuration(d.base, d.index, index);
      render();
      showSnapMarks(d.index);
    }
  }

  function onPointerUp() {
    window.removeEventListener("pointermove", onPointerMove);
    const d = state.drag;
    state.drag = null;
    hideSnapMarks();
    if (!d) return;
    const changed = JSON.stringify(state.rows) !== JSON.stringify(d.base);
    const edited = state.rows;
    state.rows = d.base;
    if (changed) commit(edited);
    else render();
  }

  // ---------------------------------------------------------------- keyboard
  function onKeyDown(e) {
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") return;
    state.lastClick = null;
    const ctrl = e.ctrlKey || e.metaKey;
    const key = e.key;
    let handled = true;
    if (ctrl && key.toLowerCase() === "z" && e.shiftKey) actionRedo();
    else if (ctrl && key.toLowerCase() === "z") actionUndo();
    else if (ctrl && key.toLowerCase() === "y") actionRedo();
    else if (ctrl && key.toLowerCase() === "a") { state.rows.forEach((r) => state.selected.add(r.uid)); render(); }
    else if (key === "Delete" || key === "Backspace") actionDelete();
    else if (key === "ArrowUp") actionTranspose(e.shiftKey ? 12 : 1);
    else if (key === "ArrowDown") actionTranspose(e.shiftKey ? -12 : -1);
    else if (key === "ArrowLeft") { const i = firstSelected(); select(i > 0 ? i - 1 : (i < 0 ? state.rows.length - 1 : 0), false); ensureVisible(firstSelected()); }
    else if (key === "ArrowRight") { const i = lastSelected(); select(i >= 0 && i < state.rows.length - 1 ? i + 1 : (i < 0 ? 0 : i), false); ensureVisible(firstSelected()); }
    else if (key === "[") actionStepDuration(1);
    else if (key === "]") actionStepDuration(-1);
    else if (key === "Enter" || key === "F2") { const i = firstSelected(); if (i >= 0) openLyricEditor(i); }
    else if (key === " ") togglePlayback();
    else if (key === "Escape") { state.selected.clear(); stopPlayback(); render(); }
    else if (key === "Home") seek(0);
    else if (key === "End") seek(M.totalTicks(state.rows));
    else if (key === "+" || key === "=") setZoom(1);
    else if (key === "-" || key === "_") setZoom(-1);
    else if (!ctrl && key.toLowerCase() === "n") actionInsertNote();
    else if (!ctrl && key.toLowerCase() === "r") actionInsertRest();
    else if (!ctrl && key.toLowerCase() === "m") actionMelisma();
    else if (!ctrl && key.toLowerCase() === "v") setTool("select");
    else if (!ctrl && key.toLowerCase() === "b") setTool("pencil");
    else if (!ctrl && key.toLowerCase() === "e") setTool("eraser");
    else handled = false;
    if (handled) e.preventDefault();
  }

  // ---------------------------------------------------------------- playback
  /** Live transport position in ticks, whether or not audio is running. */
  function transportPosition() {
    const p = state.playing;
    if (!p) return state.position;
    const elapsed = Math.max(0, state.audio.currentTime - p.t0);
    return Math.min(p.from + elapsed / p.secPerTick, M.totalTicks(state.rows));
  }

  /** Move the playhead, the ruler cursor and the time readout. */
  function renderTransport(tick) {
    const x = tick * ppt();
    el.playhead.style.left = `${x}px`;
    el.cursor.style.left = `${x}px`;
    el.time.textContent = formatSeconds(tick);
  }

  function seek(tick) {
    const total = M.totalTicks(state.rows);
    state.position = Math.max(0, Math.min(Math.round(tick), total));
    if (state.playing) {
      // Restarting is the only way to reschedule: the take is one-shot.
      teardownAudio();
      startPlayback();
    } else {
      renderTransport(state.position);
    }
  }

  function togglePlayback() {
    if (state.playing) pausePlayback();
    else startPlayback();
  }

  function startPlayback() {
    if (!state.rows.length) return;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) { setMessage({ zh: "浏览器不支持 WebAudio。", en: "WebAudio is not available." }); return; }
    if (!state.audio) state.audio = new Ctx();
    const ctx = state.audio;
    if (ctx.state === "suspended") ctx.resume();
    const events = M.layout(state.rows);
    const total = M.totalTicks(state.rows);
    // Pressing play at the end rewinds, the way a transport usually does.
    const from = state.position >= total ? 0 : state.position;
    state.position = from;
    const secPerTick = 60 / Math.max(1, state.bpm) / M.TICKS_PER_QUARTER;
    const master = ctx.createGain();
    master.gain.value = 0.35;
    master.connect(ctx.destination);
    const t0 = ctx.currentTime + 0.05;
    let endTime = t0;
    for (const ev of events) {
      if (ev.end <= from) continue;
      const start = t0 + (Math.max(ev.start, from) - from) * secPerTick;
      const stop = t0 + (ev.end - from) * secPerTick;
      endTime = Math.max(endTime, stop);
      if (ev.isRest || ev.pitch <= 0) continue;
      const osc = ctx.createOscillator();
      osc.type = "triangle";
      osc.frequency.value = 440 * Math.pow(2, (ev.pitch - 69) / 12);
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(1, start + 0.012);
      gain.gain.setValueAtTime(1, Math.max(start + 0.012, stop - 0.04));
      gain.gain.linearRampToValueAtTime(0, stop);
      osc.connect(gain).connect(master);
      osc.start(start);
      osc.stop(stop + 0.01);
    }
    state.playing = { master, t0, from, secPerTick, endTime, raf: 0 };
    el.play.classList.add("playing");
    el.play.textContent = t("⏸ 暂停", "⏸ Pause");
    el.playhead.classList.add("active");
    const tick = () => {
      if (!state.playing) return;
      if (ctx.currentTime >= state.playing.endTime) {
        // Finished: leave the playhead at the end; the next play rewinds.
        state.position = total;
        teardownAudio();
        renderTransport(state.position);
        return;
      }
      const position = transportPosition();
      renderTransport(position);
      const x = position * ppt();
      if (x > el.gridWrap.scrollLeft + el.gridWrap.clientWidth - 20 || x < el.gridWrap.scrollLeft) {
        el.gridWrap.scrollLeft = Math.max(0, x - 60);
        syncScroll();
      }
      state.playing.raf = requestAnimationFrame(tick);
    };
    tick();
  }

  /** Silence the take and drop the transport, leaving state.position alone. */
  function teardownAudio() {
    const p = state.playing;
    if (!p) return;
    cancelAnimationFrame(p.raf);
    try { p.master.gain.setTargetAtTime(0, state.audio.currentTime, 0.01); } catch (err) { /* ignore */ }
    setTimeout(() => { try { p.master.disconnect(); } catch (err) { /* ignore */ } }, 100);
    state.playing = null;
    el.play.classList.remove("playing");
    el.play.textContent = t("▶ 试听旋律", "▶ Preview melody");
    el.playhead.classList.remove("active");
  }

  /** Pause where the playhead currently is. */
  function pausePlayback() {
    if (!state.playing) return;
    const position = transportPosition();
    teardownAudio();
    state.position = Math.max(0, Math.round(position));
    renderTransport(state.position);
  }

  /** Stop and rewind to the start. */
  function stopPlayback() {
    teardownAudio();
    state.position = 0;
    renderTransport(0);
  }

  // ------------------------------------------------------------------ wiring
  function buildDurationOptions() {
    el.duration.innerHTML = M.DURATIONS.map((d, i) => `<option value="${i}">${t(d.zh, d.en)}</option>`).join("");
    el.duration.value = String(firstSelected() >= 0 ? state.rows[firstSelected()].duration_index : M.QUARTER_INDEX);
  }

  root.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-action], button[data-tool]");
    if (!btn || !root.contains(btn)) return;
    e.preventDefault();
    if (btn.dataset.tool) { setTool(btn.dataset.tool); return; }
    const action = btn.dataset.action;
    if (action === "play") togglePlayback();
    else if (action === "stop") stopPlayback();
    else if (action === "add-note") actionInsertNote();
    else if (action === "add-rest") actionInsertRest();
    else if (action === "melisma") actionMelisma();
    else if (action === "delete") actionDelete();
    else if (action === "undo") actionUndo();
    else if (action === "redo") actionRedo();
    else if (action === "zoom-in") setZoom(1);
    else if (action === "zoom-out") setZoom(-1);
    root.focus({ preventScroll: true });
  });

  el.bpm.addEventListener("change", () => {
    const bpm = Math.round(Number(el.bpm.value));
    if (!Number.isFinite(bpm)) { el.bpm.value = state.bpm; return; }
    commit(null, { bpm: Math.max(M.MIN_BPM, Math.min(M.MAX_BPM, bpm)) });
  });
  el.meter.addEventListener("change", () => commit(null, { beatsPerBar: Number(el.meter.value) }));

  el.lyric.addEventListener("change", () => {
    const index = firstSelected();
    if (index >= 0) actionSetLyric(index, el.lyric.value);
  });
  el.lyric.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); el.lyric.blur(); root.focus({ preventScroll: true }); } });
  el.pitch.addEventListener("change", () => {
    const index = firstSelected();
    if (index >= 0) commit(M.setPitch(state.rows, index, el.pitch.value));
  });
  el.duration.addEventListener("change", () => {
    const indices = selectedIndices();
    if (!indices.length) return;
    const value = Number(el.duration.value);
    let rows = state.rows;
    for (const i of indices) rows = M.setDuration(rows, i, value);
    commit(rows);
  });

  el.keys.addEventListener("click", (e) => {
    const key = e.target.closest(".vr-key");
    if (!key) return;
    const indices = selectedIndices().filter((i) => !M.isRest(state.rows[i]));
    if (indices.length) {
      let rows = state.rows;
      for (const i of indices) rows = M.setPitch(rows, i, Number(key.dataset.pitch));
      commit(rows);
    } else {
      previewPitch(Number(key.dataset.pitch));
    }
  });

  function previewPitch(pitch) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    if (!state.audio) state.audio = new Ctx();
    const ctx = state.audio;
    if (ctx.state === "suspended") ctx.resume();
    const osc = ctx.createOscillator();
    osc.type = "triangle";
    osc.frequency.value = 440 * Math.pow(2, (pitch - 69) / 12);
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.36);
  }

  /* Ruler scrubbing, after SingScope's transport: the bar-number strip is a
   * seek zone, a dashed line previews where a click would land, and dragging
   * scrubs. Alt drops the snap entirely.
   *
   * Events are contiguous, so a note starts at the sum of the durations before
   * it -- and the legal durations are powers of two and their dotted forms, not
   * a uniform grid. One 32nd pushes everything after it off the sixteenth grid
   * and one dotted 32nd pushes it off the 32nd grid too. Snapping to the drawn
   * grid alone would therefore put the boundaries you actually want to audition
   * from out of reach, so event edges are the primary targets and the grid is
   * only the fallback in the empty space beyond the score. */
  const SNAP_PX = 7;
  function snapTick(tick) {
    const px = ppt();
    let edge = null;
    let edgeDist = Infinity;
    const consider = (candidate) => {
      const dist = Math.abs(tick - candidate) * px;
      if (dist < edgeDist) { edgeDist = dist; edge = candidate; }
    };
    for (const ev of M.layout(state.rows)) consider(ev.start);
    consider(M.totalTicks(state.rows));
    if (edge === null) return Math.round(tick / SUB_GRID_TICKS) * SUB_GRID_TICKS;
    if (edgeDist <= SNAP_PX) return edge;
    const grid = Math.round(tick / SUB_GRID_TICKS) * SUB_GRID_TICKS;
    return edgeDist < Math.abs(tick - grid) * px ? edge : grid;
  }

  function rulerTick(e, free) {
    const rect = el.bars.getBoundingClientRect();
    const tick = Math.max(0, (e.clientX - rect.left) / ppt());
    return free ? tick : snapTick(tick);
  }

  el.bars.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    closeLyricEditor();
    root.focus({ preventScroll: true });
    const scrub = (ev) => seek(rulerTick(ev, ev.altKey));
    scrub(e);
    const up = () => {
      window.removeEventListener("pointermove", scrub);
      el.hover.classList.remove("on");
    };
    window.addEventListener("pointermove", scrub);
    window.addEventListener("pointerup", up, { once: true });
  });
  el.bars.addEventListener("pointermove", (e) => {
    el.hover.classList.add("on");
    el.hover.style.left = `${rulerTick(e, e.altKey) * ppt()}px`;
  });
  el.bars.addEventListener("pointerleave", () => el.hover.classList.remove("on"));

  el.notes.addEventListener("pointerover", (e) => {
    const handle = e.target.closest(".vr-handle");
    if (handle && state.tool === "select") showSnapMarks(Number(handle.dataset.index));
  });
  el.notes.addEventListener("pointerout", (e) => {
    if (e.target.closest(".vr-handle")) hideSnapMarks();
  });

  root.addEventListener("pointerdown", onPointerDown);
  root.addEventListener("keydown", onKeyDown);
  el.gridWrap.addEventListener("scroll", syncScroll);
  el.gridWrap.addEventListener("wheel", (e) => {
    if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return; // native horizontal scroll
    if (e.ctrlKey) { e.preventDefault(); setZoom(e.deltaY < 0 ? 1 : -1); }
  }, { passive: false });
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => render()).observe(el.gridWrap);

  watch("value", () => loadFromProps());
  watch("language", () => localize());
  watch("lyric_targets", () => highlightLyrics());
  localize();

  buildKeys();
  buildDurationOptions();
  loadFromProps();
})();
