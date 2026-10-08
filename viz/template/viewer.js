/* Byte replay viewer.
 *
 * Plays back one recorded run. The food grid is stored as an initial state plus
 * per-tick deltas, so the grid for any tick is reconstructed by replaying those
 * deltas rather than storing 2601 cells per tick.
 */
(function () {
  "use strict";

  const DATA = window.RUN_DATA;
  const S = DATA.static;
  const FRAMES = DATA.frames;
  const W = S.grid_width;
  const H = S.grid_height;
  const N = FRAMES.length;
  const TRAIL_LEN = 60;
  const EAT_FLASH = 6;      // ticks a meal stays visible for

  const css = getComputedStyle(document.documentElement);
  const C = {
    grid: css.getPropertyValue("--grid").trim(),
    food: css.getPropertyValue("--blue").trim(),
    pending: css.getPropertyValue("--blue-mid").trim(),
    agent: css.getPropertyValue("--gold").trim(),
    sensor: css.getPropertyValue("--gold-light").trim(),
    dead: css.getPropertyValue("--dead").trim(),
    edge: css.getPropertyValue("--edge").trim(),
    warn: css.getPropertyValue("--warn").trim(),
    exc: css.getPropertyValue("--gold-light").trim(),
    inh: css.getPropertyValue("--blue").trim(),
    text: css.getPropertyValue("--text").trim(),
    dim: css.getPropertyValue("--dim").trim(),
    panelEdge: css.getPropertyValue("--edge").trim(),
    neuronIdle: css.getPropertyValue("--panel").trim(),
    neuronEdge: css.getPropertyValue("--edge").trim(),
  };

  // ---------------------------------------------------------------
  // Reconstructed world state
  // ---------------------------------------------------------------

  let food = new Uint8Array(W * H);      // 1 where food is present
  let pending = new Uint8Array(W * H);   // eaten, awaiting regrow
  let builtTo = -1;                      // which tick `food` currently represents

  function resetGrid() {
    food.fill(0);
    pending.fill(0);
    for (const i of DATA.initial_food) food[i] = 1;
    builtTo = 0;
  }

  function applyFrame(f) {
    for (const i of f.food_removed) { food[i] = 0; pending[i] = 1; }
    for (const i of f.food_added) { food[i] = 1; pending[i] = 0; }
  }

  /* Bring `food`/`pending` up to frame index n.
     Stepping forward is incremental; seeking backwards rebuilds from the start,
     which is cheap because each frame touches only a handful of cells. */
  function buildTo(n) {
    if (n === builtTo) return;
    if (n < builtTo) resetGrid();
    for (let i = builtTo + 1; i <= n; i++) applyFrame(FRAMES[i]);
    builtTo = n;
  }

  // ---------------------------------------------------------------
  // Events (for scrub markers)
  // ---------------------------------------------------------------

  // Beyond this many meals they stop being individually interesting.
  const EAT_MARKER_LIMIT = 40;

  const eatTicks = [];
  const notable = [];
  FRAMES.forEach(function (f, i) {
    if (f.ate !== null && f.ate !== undefined) eatTicks.push(i);
    // A whole-grid reseed at a phase boundary changes far more cells than
    // ordinary eating or regrowth ever does.
    if (f.food_added.length + f.food_removed.length > 30) {
      notable.push({ i: i, kind: "phase" });
    }
    // Only the tick the agent died on, not every tick after it.
    if (!f.alive && (i === 0 || FRAMES[i - 1].alive)) {
      notable.push({ i: i, kind: "death" });
    }
  });

  // A camping agent eats hundreds of times. Marking each one turns the scrub
  // bar into a solid band and leaves the jump buttons stepping through meals
  // instead of reaching the phase switch or the death. Keep meal markers only
  // while they are sparse enough to read; phase and death always show.
  const events = eatTicks.length <= EAT_MARKER_LIMIT
    ? notable.concat(eatTicks.map(i => ({ i: i, kind: "ate" })))
    : notable;

  const EVENT_COLOR = { ate: C.food, phase: C.warn, death: C.dead };

  function drawMarkers() {
    const el = document.getElementById("markers");
    el.innerHTML = "";
    if (N < 2) return;
    const seen = new Set();
    for (const ev of events) {
      const key = ev.i + ev.kind;
      if (seen.has(key)) continue;
      seen.add(key);
      const tick = document.createElement("div");
      tick.style.cssText =
        "position:absolute;top:4px;width:2px;height:7px;border-radius:1px;" +
        "background:" + EVENT_COLOR[ev.kind] + ";opacity:.75;left:" +
        (ev.i / (N - 1)) * 100 + "%;";
      el.appendChild(tick);
    }
  }

  // ---------------------------------------------------------------
  // Canvas
  // ---------------------------------------------------------------

  const canvas = document.getElementById("world");
  const ctx = canvas.getContext("2d");
  let cell = 8;

  function sizeCanvas(el, c, w, h) {
    const dpr = window.devicePixelRatio || 1;
    el.width = Math.round(w * dpr);
    el.height = Math.round(h * dpr);
    el.style.width = w + "px";
    el.style.height = h + "px";
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function fitCanvas() {
    const panels = document.querySelectorAll(".panel");
    const wp = panels[0];
    const avail = Math.min(wp.clientWidth - 20, wp.clientHeight - 34);
    const size = Math.max(180, avail);
    cell = size / Math.max(W, H);
    sizeCanvas(canvas, ctx, W * cell, H * cell);

    const bp = panels[1];
    if (bp) {
      const bw = Math.max(180, bp.clientWidth - 20);
      const bh = Math.max(180, bp.clientHeight - 34);
      bCell = { w: bw, h: bh };
      sizeCanvas(document.getElementById("brain"), bctx, bw, bh);
    }
  }

  function wrapDelta(d, span) {
    if (d > span / 2) return d - span;
    if (d < -span / 2) return d + span;
    return d;
  }

  function draw(n) {
    const f = FRAMES[n];
    ctx.clearRect(0, 0, W * cell, H * cell);

    // lattice
    if (cell >= 5) {
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x <= W; x++) { ctx.moveTo(x * cell, 0); ctx.lineTo(x * cell, H * cell); }
      for (let y = 0; y <= H; y++) { ctx.moveTo(0, y * cell); ctx.lineTo(W * cell, y * cell); }
      ctx.stroke();
    }

    // cells eaten and awaiting regrowth
    ctx.fillStyle = C.pending;
    ctx.globalAlpha = 0.28;
    for (let i = 0; i < pending.length; i++) {
      if (pending[i] && !food[i]) {
        const cx = (i % W) * cell, cy = ((i / W) | 0) * cell;
        ctx.beginPath();
        ctx.arc(cx + cell / 2, cy + cell / 2, Math.max(1, cell * 0.16), 0, 6.2832);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;

    // food
    ctx.fillStyle = C.food;
    const pad = cell * 0.18, side = cell - pad * 2;
    for (let i = 0; i < food.length; i++) {
      if (food[i]) {
        ctx.fillRect((i % W) * cell + pad, ((i / W) | 0) * cell + pad, side, side);
      }
    }

    // trail — drawn as discrete cells, not a polyline: the world is toroidal,
    // so connecting points would streak a line across the map on every wrap
    const from = Math.max(0, n - TRAIL_LEN);
    for (let i = from; i < n; i++) {
      const age = (i - from) / Math.max(1, n - from);
      ctx.globalAlpha = 0.05 + 0.3 * age;
      ctx.fillStyle = C.agent;
      const r = cell * (0.12 + 0.14 * age);
      ctx.beginPath();
      ctx.arc((FRAMES[i].x + 0.5) * cell, (FRAMES[i].y + 0.5) * cell, r, 0, 6.2832);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // recent meals. A camping agent regrows and eats the same cell within one
    // tick, so the food grid never changes and the only sign of eating is this.
    for (let i = Math.max(0, n - EAT_FLASH); i <= n; i++) {
      const cellIdx = FRAMES[i].ate;
      if (cellIdx === null || cellIdx === undefined) continue;
      const age = 1 - (n - i) / EAT_FLASH;          // 1 = this tick
      const ex = (cellIdx % W) * cell, ey = ((cellIdx / W) | 0) * cell;
      ctx.globalAlpha = 0.15 + 0.75 * age;
      ctx.strokeStyle = C.food;
      ctx.lineWidth = Math.max(1, cell * 0.14);
      const grow = cell * (0.5 + 0.5 * (1 - age));
      ctx.beginPath();
      ctx.arc(ex + cell / 2, ey + cell / 2, grow, 0, 6.2832);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // the five sensed cells
    const sense = f.sense || {};
    const probes = [
      ["on_food", f.y, f.x],
      ["food_north", (f.y - 1 + H) % H, f.x],
      ["food_south", (f.y + 1) % H, f.x],
      ["food_west", f.y, (f.x - 1 + W) % W],
      ["food_east", f.y, (f.x + 1) % W],
    ];
    ctx.lineWidth = Math.max(1, cell * 0.09);
    for (const [key, py, px] of probes) {
      if (!(key in sense)) continue;
      const lit = sense[key] > 0;
      ctx.strokeStyle = C.sensor;
      ctx.globalAlpha = lit ? 0.95 : 0.25;
      ctx.strokeRect(px * cell + 1, py * cell + 1, cell - 2, cell - 2);
      if (lit) {
        ctx.globalAlpha = 0.3;
        ctx.fillStyle = C.sensor;
        ctx.fillRect(px * cell + 1, py * cell + 1, cell - 2, cell - 2);
      }
    }
    ctx.globalAlpha = 1;

    drawAgent(f, n);
  }

  function drawAgent(f, n) {
    const cx = (f.x + 0.5) * cell, cy = (f.y + 0.5) * cell;
    const frac = Math.max(0, Math.min(1, f.energy / S.energy_capacity));
    const ring = cell * 2.0, core = cell * 0.42;
    const colour = f.alive ? C.agent : C.dead;

    // glow: the agent is one cell among thousands and needs to stand out
    ctx.globalAlpha = 0.1;
    ctx.fillStyle = colour;
    ctx.beginPath(); ctx.arc(cx, cy, ring, 0, 6.2832); ctx.fill();
    ctx.globalAlpha = 1;

    // energy track + arc. Energy is encoded as arc LENGTH, not hue, so it stays
    // readable for any colour vision.
    ctx.lineWidth = Math.max(1.5, cell * 0.2);
    ctx.strokeStyle = C.edge;
    ctx.beginPath(); ctx.arc(cx, cy, ring, 0, 6.2832); ctx.stroke();

    if (f.alive && frac > 0) {
      ctx.strokeStyle = frac < 0.25 ? C.warn : C.agent;
      ctx.beginPath();
      ctx.arc(cx, cy, ring, -Math.PI / 2, -Math.PI / 2 + 6.2832 * frac);
      ctx.stroke();
    }

    // heading from the previous position, wrap-aware
    if (n > 0 && f.alive) {
      const dy = wrapDelta(f.y - FRAMES[n - 1].y, H);
      const dx = wrapDelta(f.x - FRAMES[n - 1].x, W);
      if (dy || dx) {
        const uy = Math.sign(dy), ux = Math.sign(dx);
        ctx.strokeStyle = C.agent;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(cx + ux * core * 1.6, cy + uy * core * 1.6);
        ctx.lineTo(cx + ux * ring * 0.8, cy + uy * ring * 0.8);
        ctx.stroke();
        ctx.lineCap = "butt";
      }
    }

    ctx.fillStyle = colour;
    ctx.beginPath(); ctx.arc(cx, cy, core, 0, 6.2832); ctx.fill();

    if (!f.alive) {
      ctx.strokeStyle = C.dead;
      ctx.lineWidth = Math.max(1.5, cell * 0.16);
      ctx.lineCap = "round";
      const d = core * 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - d, cy - d); ctx.lineTo(cx + d, cy + d);
      ctx.moveTo(cx - d, cy + d); ctx.lineTo(cx + d, cy - d);
      ctx.stroke();
      ctx.lineCap = "butt";
    }
  }


  // ---------------------------------------------------------------
  // Brain
  // ---------------------------------------------------------------
  // Weights are stored as sparse change events rather than a value per beat,
  // because a connection only moves while one of its modulators fires and
  // stops permanently once it saturates at +/-1. Rebuilding them works the
  // same way as the food grid: step forward incrementally, replay from the
  // start when seeking backwards.

  const BRAIN = DATA.brain || null;
  const bctx = document.getElementById("brain").getContext("2d");

  let bLayout = null;     // node id / input key -> {x, y} in unit space
  let bWeights = null;    // current weight per edge
  let bChanges = null;    // beat index -> [[edge, value], ...]
  let bOffsets = null;    // first beat index of each decision
  let bBuiltTo = -1;      // beat the weights currently represent
  let bCell = 0;          // px per unit, set by fitCanvas

  if (BRAIN) {
    const S2 = BRAIN.static;

    // Rows: input sources, the neurons they feed, everything unlabelled, then
    // the output neurons. Derived from the recording, not hardcoded, so a
    // different brain size still lays out sensibly.
    const byRole = r => S2.neurons.filter(n => n.role === r).map(n => n.id);
    const rows = [
      S2.inputs.map(k => "in:" + k),
      byRole("sensory"),
      byRole("hidden"),
      byRole("output"),
    ].filter(r => r.length);

    bLayout = {};
    rows.forEach(function (row, ri) {
      const y = rows.length === 1 ? 0.5 : ri / (rows.length - 1);
      row.forEach(function (key, ci) {
        bLayout[key] = { x: (ci + 0.5) / row.length, y: y };
      });
    });

    bChanges = new Map();
    for (const [beat, edge, val] of BRAIN.weight_changes) {
      if (!bChanges.has(beat)) bChanges.set(beat, []);
      bChanges.get(beat).push([edge, val]);
    }

    bOffsets = new Array(BRAIN.beats_per_tick.length);
    let acc = 0;
    for (let i = 0; i < BRAIN.beats_per_tick.length; i++) {
      bOffsets[i] = acc;
      acc += BRAIN.beats_per_tick[i];
    }
  }

  /* World frame t is produced by decision t-1; frame 0 predates any decision. */
  function decisionOf(t) { return t - 1; }

  /* Last beat of a decision: the committed state the action was taken on. */
  function lastBeatOf(d) {
    if (!BRAIN || d < 0 || d >= BRAIN.beats_per_tick.length) return -1;
    return bOffsets[d] + BRAIN.beats_per_tick[d] - 1;
  }

  function resetWeights() {
    bWeights = BRAIN.static.edges.map(e => e.w0);
    bBuiltTo = -1;
  }

  function weightsTo(beat) {
    if (beat === bBuiltTo) return;
    if (beat < bBuiltTo) resetWeights();
    for (let b = bBuiltTo + 1; b <= beat; b++) {
      const list = bChanges.get(b);
      if (list) for (const [edge, val] of list) bWeights[edge] = val;
    }
    bBuiltTo = beat;
  }

  const ACTION_LABEL = {
    stay: "STAY", move_north: "N", move_east: "E",
    move_south: "S", move_west: "W",
  };

  function nodePos(key, w, h, pad) {
    const p = bLayout[key];
    return { x: pad + p.x * (w - 2 * pad), y: pad + p.y * (h - 2 * pad) };
  }

  function drawBrain(n) {
    if (!BRAIN) return;
    const S2 = BRAIN.static;
    const W2 = bCell.w, H2 = bCell.h;
    bctx.clearRect(0, 0, W2, H2);

    const d = decisionOf(FRAMES[n].t);
    const beat = lastBeatOf(d);
    if (beat >= 0) weightsTo(beat); else resetWeights();
    const act = beat >= 0 ? BRAIN.activations[beat] : 0;
    const cand = (d >= 0 && d < BRAIN.candidates_per_tick.length)
      ? BRAIN.candidates_per_tick[d] : 0;

    const pad = Math.min(W2, H2) * 0.14;
    const r = Math.max(7, Math.min(W2, H2) * 0.045);
    const fired = id => (act >> id & 1) === 1;

    // --- edges ---
    for (let i = 0; i < S2.edges.length; i++) {
      const e = S2.edges[i];
      const w = bWeights[i];
      if (Math.abs(w) < 0.01) continue;              // invisible anyway
      const a = nodePos(e.src, W2, H2, pad);
      const b = nodePos(e.tgt, W2, H2, pad);
      const live = typeof e.src === "number" ? fired(e.src) : false;

      // Opacity carries magnitude and reliability; a firing source brightens
      // it so signal flow is visible without changing the encoding.
      const mag = Math.min(1, Math.abs(w));
      bctx.globalAlpha = (0.06 + 0.5 * mag * e.rel) * (live ? 2.0 : 1);
      bctx.strokeStyle = w >= 0 ? C.exc : C.inh;
      bctx.lineWidth = 0.5 + 3 * mag;

      if (e.src === e.tgt) {
        bctx.beginPath();
        bctx.arc(a.x, a.y - r * 1.5, r * 0.85, 0, 6.2832);
        bctx.stroke();
      } else {
        const dx = b.x - a.x, dy = b.y - a.y;
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len;
        const sx = a.x + ux * r, sy = a.y + uy * r;
        const ex = b.x - ux * r, ey = b.y - uy * r;
        // Curve every edge the same way round so opposing pairs stay distinct.
        const cx = (sx + ex) / 2 - (ey - sy) * 0.18;
        const cy = (sy + ey) / 2 + (ex - sx) * 0.18;
        bctx.beginPath();
        bctx.moveTo(sx, sy);
        bctx.quadraticCurveTo(cx, cy, ex, ey);
        bctx.stroke();
        arrowHead(ex, ey, ex - cx, ey - cy, 3 + 3 * mag);
      }
    }
    bctx.globalAlpha = 1;

    // --- input sources ---
    bctx.font = "600 9px Consolas, monospace";
    bctx.textAlign = "center";
    bctx.textBaseline = "middle";
    for (const key of S2.inputs) {
      const p = nodePos("in:" + key, W2, H2, pad);
      const on = (FRAMES[n].sense || {})[key] > 0;
      bctx.fillStyle = on ? C.sensor : C.panelEdge;
      roundRect(p.x - r * 0.95, p.y - r * 0.6, r * 1.9, r * 1.2, 3);
      bctx.fill();
      bctx.fillStyle = on ? "#12160f" : C.dim;
      bctx.fillText(shortSensor(key), p.x, p.y);
    }

    // --- neurons ---
    for (const neu of S2.neurons) {
      const p = nodePos(neu.id, W2, H2, pad);
      const on = fired(neu.id);
      const isCand = (cand >> neu.id & 1) === 1;

      if (isCand) {                                  // candidate for the decision
        bctx.strokeStyle = C.sensor;
        bctx.globalAlpha = 0.9;
        bctx.lineWidth = 1.5;
        bctx.setLineDash([3, 3]);
        bctx.beginPath(); bctx.arc(p.x, p.y, r * 1.45, 0, 6.2832); bctx.stroke();
        bctx.setLineDash([]);
        bctx.globalAlpha = 1;
      }

      bctx.beginPath(); bctx.arc(p.x, p.y, r, 0, 6.2832);
      bctx.fillStyle = on ? C.agent : C.neuronIdle;
      bctx.fill();
      bctx.lineWidth = 1.5;
      bctx.strokeStyle = on ? C.agent : C.neuronEdge;
      bctx.stroke();

      bctx.fillStyle = on ? "#12160f" : C.text;
      bctx.font = "600 " + Math.round(r * 0.95) + "px Consolas, monospace";
      bctx.fillText(String(neu.id), p.x, p.y);

      const action = S2.output_mapping[neu.id];
      if (action) {
        bctx.fillStyle = C.dim;
        bctx.font = "600 9px Consolas, monospace";
        bctx.fillText(ACTION_LABEL[action] || action, p.x, p.y + r * 1.9);
      }
    }
  }

  function shortSensor(key) {
    return { on_food: "\u25cf", food_north: "N", food_east: "E",
             food_south: "S", food_west: "W" }[key] || key.slice(0, 3);
  }

  function arrowHead(x, y, dx, dy, size) {
    const a = Math.atan2(dy, dx);
    bctx.beginPath();
    bctx.moveTo(x, y);
    bctx.lineTo(x - size * Math.cos(a - 0.5), y - size * Math.sin(a - 0.5));
    bctx.lineTo(x - size * Math.cos(a + 0.5), y - size * Math.sin(a + 0.5));
    bctx.closePath();
    bctx.fillStyle = bctx.strokeStyle;
    bctx.fill();
  }

  function roundRect(x, y, w, h, rad) {
    bctx.beginPath();
    bctx.moveTo(x + rad, y);
    bctx.arcTo(x + w, y, x + w, y + h, rad);
    bctx.arcTo(x + w, y + h, x, y + h, rad);
    bctx.arcTo(x, y + h, x, y, rad);
    bctx.arcTo(x, y, x + w, y, rad);
    bctx.closePath();
  }

  // ---------------------------------------------------------------
  // Status panel
  // ---------------------------------------------------------------

  const SENSOR_CELLS = {
    "c-n": "food_north", "c-s": "food_south",
    "c-e": "food_east", "c-w": "food_west", "c-c": "on_food",
  };

  function updateStatus(n) {
    const f = FRAMES[n];
    const frac = Math.max(0, Math.min(1, f.energy / S.energy_capacity));

    const fill = document.getElementById("energy-fill");
    fill.style.width = frac * 100 + "%";
    fill.style.background = !f.alive ? C.dead : (frac < 0.25 ? C.warn : C.agent);

    document.getElementById("energy-text").textContent =
      "ENERGY " + f.energy + "/" + S.energy_capacity;
    const meal = (f.ate !== null && f.ate !== undefined) ? "  ATE" : "";
    document.getElementById("counters").textContent =
      "eats " + f.eats + "   dist " + f.distance + "   pos " + f.y + "," + f.x + meal;

    const state = document.getElementById("alive-state");
    state.textContent = f.alive ? "ALIVE" : "DEAD";
    state.classList.toggle("dead", !f.alive);

    const sense = f.sense || {};
    for (const cls in SENSOR_CELLS) {
      const el = document.querySelector("." + cls);
      const key = SENSOR_CELLS[cls];
      const present = key in sense;
      el.classList.toggle("lit", present && sense[key] > 0);
      el.classList.toggle("absent", !present);
    }

    document.getElementById("tick").innerHTML =
      "<strong>" + f.t + "</strong> / " + FRAMES[N - 1].t +
      (f.next_action ? "   next <strong>" + f.next_action + "</strong>" : "");
  }

  // ---------------------------------------------------------------
  // Transport
  // ---------------------------------------------------------------

  let cur = 0;
  let playing = false;
  let speed = 8;          // ticks per second
  let acc = 0;
  let last = 0;

  const seek = document.getElementById("seek");
  const playBtn = document.getElementById("play");

  function show(n) {
    cur = Math.max(0, Math.min(N - 1, n));
    buildTo(cur);
    draw(cur);
    drawBrain(cur);
    updateStatus(cur);
    seek.value = cur;
  }

  function setPlaying(on) {
    // Restarting from the end would otherwise sit there doing nothing
    if (on && cur >= N - 1) show(0);
    playing = on;
    playBtn.textContent = on ? "❙❙" : "▶";
    playBtn.title = on ? "Pause (space)" : "Play (space)";
    if (on) { last = performance.now(); requestAnimationFrame(tick); }
  }

  function tick(now) {
    if (!playing) return;
    acc += (now - last) / 1000 * speed;
    last = now;
    if (acc >= 1) {
      const step = Math.floor(acc);
      acc -= step;
      if (cur + step >= N - 1) { show(N - 1); setPlaying(false); return; }
      show(cur + step);
    }
    requestAnimationFrame(tick);
  }

  function jumpEvent(dir) {
    const ticks = [...new Set(events.map(e => e.i))].sort((a, b) => a - b);
    const next = dir > 0 ? ticks.find(i => i > cur)
                         : [...ticks].reverse().find(i => i < cur);
    if (next !== undefined) show(next);
  }

  // ---------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------

  seek.min = 0;
  seek.max = N - 1;
  seek.addEventListener("input", function () {
    setPlaying(false);
    show(parseInt(seek.value, 10));
  });

  playBtn.addEventListener("click", function () { setPlaying(!playing); });
  document.getElementById("prev").addEventListener("click", function () {
    setPlaying(false); show(cur - 1);
  });
  document.getElementById("next").addEventListener("click", function () {
    setPlaying(false); show(cur + 1);
  });
  document.getElementById("prev-ev").addEventListener("click", function () {
    setPlaying(false); jumpEvent(-1);
  });
  document.getElementById("next-ev").addEventListener("click", function () {
    setPlaying(false); jumpEvent(1);
  });
  document.getElementById("speed").addEventListener("change", function (e) {
    speed = parseFloat(e.target.value);
  });

  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
    else if (e.key === "ArrowLeft") { setPlaying(false); show(cur - (e.shiftKey ? 10 : 1)); }
    else if (e.key === "ArrowRight") { setPlaying(false); show(cur + (e.shiftKey ? 10 : 1)); }
    else if (e.key === "Home") { setPlaying(false); show(0); }
    else if (e.key === "End") { setPlaying(false); show(N - 1); }
  });

  window.addEventListener("resize", function () { fitCanvas(); draw(cur); drawBrain(cur); });

  resetGrid();
  fitCanvas();
  drawMarkers();
  show(0);
})();
