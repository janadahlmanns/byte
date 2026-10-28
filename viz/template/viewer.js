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
    // Gold and blue are reserved for the sign of a weight. A weight moving is
    // not a statement about sign, so it is marked in neutral white rather than
    // in the same gold as every excitatory edge it is drawn on top of.
    learn: css.getPropertyValue("--text").trim(),
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

  // Beyond this many of one kind they stop being individually interesting.
  const EAT_MARKER_LIMIT = 40;

  const eatTicks = [];
  const moveTicks = [];
  const notable = [];
  FRAMES.forEach(function (f, i) {
    if (f.ate !== null && f.ate !== undefined) eatTicks.push(i);
    // An evolved agent that has learnt to camp stands still for almost the
    // whole run, so the handful of ticks it does move are worth finding.
    if (i > 0 && (f.x !== FRAMES[i - 1].x || f.y !== FRAMES[i - 1].y)) {
      moveTicks.push(i);
    }
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
  const events = notable.slice();
  if (eatTicks.length <= EAT_MARKER_LIMIT) {
    for (const i of eatTicks) events.push({ i: i, kind: "ate" });
  }
  if (moveTicks.length <= EAT_MARKER_LIMIT) {
    for (const i of moveTicks) events.push({ i: i, kind: "move" });
  }

  const EVENT_COLOR = {
    ate: C.food, phase: C.warn, death: C.dead, learn: C.exc,
    move: C.text, think: C.agent, fallback: C.warn,
  };

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

  /* Sized against the canvas box rather than the panel, so the panel keeps
     the freedom to carry a key and a caption without the drawing code needing
     to know their height. */
  function fitCanvas() {
    const boxes = document.querySelectorAll(".canvas-box");
    const wb = boxes[0];
    const size = Math.max(180, Math.min(wb.clientWidth, wb.clientHeight));
    cell = size / Math.max(W, H);
    sizeCanvas(canvas, ctx, W * cell, H * cell);

    const bb = boxes[1];
    if (bb) {
      bCell = {
        w: Math.max(180, bb.clientWidth),
        h: Math.max(180, bb.clientHeight),
      };
      sizeCanvas(document.getElementById("brain"), bctx, bCell.w, bCell.h);
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
  let bModByEdge = null;  // edge index -> modulators acting on it
  let bLearned = null;    // decision index -> set of edges that changed then
  let bThink = null;      // decisions whose activation moved mid-tick
  let bFallback = null;   // decisions with no candidate: the action was random
  // "active": only modulators that are firing, which is the only time they do
  // anything. "all": the full wiring, signal edges pushed back. "off": bare
  // network, nothing drawn on top of it.
  const MOD_MODES = ["active", "all", "off"];
  let modMode = "active";
  // Clean view by default. A late-run network has roughly a third of its edges
  // pruned and a third saturated, and drawing all of that at once buries the
  // handful of connections that are actually carrying the decision. Detail
  // view puts the structural marks back for debugging.
  let detail = false;
  let bCell = 0;          // px per unit, set by fitCanvas

  if (BRAIN) {
    const S2 = BRAIN.static;

    // Rows: input sources, the neurons they feed, everything unlabelled, then
    // the output neurons. Derived from the recording, not hardcoded, so a
    // different brain size still lays out sensibly.
    const byRole = r => S2.neurons.filter(n => n.role === r).map(n => n.id);

    // Order the input row by the neuron each sensor feeds. Sensors are listed
    // in config order, which need not match neuron ids, so without this the
    // five input edges cross each other on the way down for no reason.
    const inputTarget = {};
    for (const e of S2.edges) {
      if (typeof e.src === "string" && !(e.src in inputTarget)) {
        inputTarget[e.src] = e.tgt;
      }
    }
    const inputRow = S2.inputs
      .map(k => "in:" + k)
      .sort((a, b) => (inputTarget[a] ?? 1e9) - (inputTarget[b] ?? 1e9));

    const rows = [
      inputRow,
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

    // Modulators grouped by the edge they act on, so the drawing can ask
    // "what changes this connection" without rescanning the list.
    bModByEdge = new Map();
    for (const m of S2.modulation) {
      if (!bModByEdge.has(m.edge)) bModByEdge.set(m.edge, []);
      bModByEdge.get(m.edge).push(m);
    }

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

    // Which edges learned during each decision. Weight changes arrive indexed
    // by beat; this folds them up to the tick the timeline actually scrubs on.
    bLearned = new Map();
    let di = 0;
    for (const [beat, edge] of BRAIN.weight_changes) {
      while (di + 1 < bOffsets.length && bOffsets[di + 1] <= beat) di++;
      if (!bLearned.has(di)) bLearned.set(di, new Set());
      bLearned.get(di).add(edge);
    }

    // Ticks where at least one weight moved. Learning happens in short bursts,
    // a few percent of ticks in a typical run, so these are worth navigating to.
    // Appended here rather than in the events block above, which runs before
    // the brain data exists.
    for (const d of bLearned.keys()) {
      const frame = d + 1;                  // decision d produces frame d+1
      if (frame < FRAMES.length) events.push({ i: frame, kind: "learn" });
    }

    // A decision whose activation pattern never moves is one the network did
    // not have to resolve: it was already at its fixed point when the tick
    // opened, and all fourteen beats re-asserted it. On a converged elite that
    // is almost every tick. Recorded explicitly so the viewer can say so,
    // rather than leaving a motionless panel to look like a stalled render.
    bThink = new Set();
    for (let d = 0; d < BRAIN.beats_per_tick.length; d++) {
      const start = bOffsets[d], end = start + BRAIN.beats_per_tick[d];
      for (let i = start + 1; i < end; i++) {
        if (BRAIN.activations[i] !== BRAIN.activations[i - 1]) {
          bThink.add(d);
          break;
        }
      }
    }

    // No output neuron survived the propagation phase as a candidate, so no
    // amount of stability checking can yield a decision and decide() falls
    // through to a random action. Rare, and the clearest failure the brain has.
    bFallback = new Set();
    for (let d = 0; d < BRAIN.candidates_per_tick.length; d++) {
      if (BRAIN.candidates_per_tick[d] === 0) bFallback.add(d);
    }

    // On a brain that never settles these cover most of the run, at which
    // point they are the norm rather than an event and jumping between them
    // means nothing. The run-shape lane still carries the density either way.
    if (bThink.size <= EAT_MARKER_LIMIT) {
      for (const d of bThink) {
        if (d + 1 < FRAMES.length) events.push({ i: d + 1, kind: "think" });
      }
    }
    for (const d of bFallback) {
      if (d + 1 < FRAMES.length) events.push({ i: d + 1, kind: "fallback" });
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
    const learned = (bLearned && bLearned.get(d)) || null;
    const beat = lastBeatOf(d);
    if (beat >= 0) weightsTo(beat); else resetWeights();
    const act = beat >= 0 ? BRAIN.activations[beat] : 0;
    const cand = (d >= 0 && d < BRAIN.candidates_per_tick.length)
      ? BRAIN.candidates_per_tick[d] : 0;

    const pad = Math.min(W2, H2) * 0.14;
    const r = Math.max(7, Math.min(W2, H2) * 0.045);
    const fired = id => (act >> id & 1) === 1;

    // --- edges ---
    const mid = new Array(S2.edges.length);          // where modulation attaches
    const shown = new Array(S2.edges.length);        // drawn, so worth annotating
    let nPruned = 0, nFrozen = 0;

    for (let i = 0; i < S2.edges.length; i++) {
      const e = S2.edges[i];
      const w = bWeights[i];
      const a = nodePos(e.src, W2, H2, pad);
      const b = nodePos(e.tgt, W2, H2, pad);
      const live = typeof e.src === "number" ? fired(e.src) : false;

      // Both bounds of the plasticity rule are absorbing: the |w|(1-|w|) factor
      // is zero at 0 and at +/-1 alike, so a weight that reaches either can
      // never move again. A pruned edge is therefore permanently dead wiring,
      // not a connection that happens to be quiet.
      const pruned = Math.abs(w) < 0.01;
      const frozen = Math.abs(w) >= 0.999;
      if (pruned) nPruned++;
      if (frozen) nFrozen++;

      // Clean view drops dead wiring entirely rather than ghosting it. In a
      // late-run network that is roughly a third of the edges, and leaving them
      // in buries the few that are carrying the decision.
      shown[i] = !pruned || detail;

      // Opacity carries magnitude and reliability; a firing source brightens
      // it so signal flow is visible without changing the encoding.
      const mag = Math.min(1, Math.abs(w));
      // The signal wiring is the network itself, so it is the primary layer and
      // never fades below the modulation overlay drawn on top of it. Only the
      // dedicated modulation view pushes it back.
      bctx.globalAlpha = (0.18 + 0.62 * mag * e.rel) * (live ? 1.6 : 1)
                         * (modMode === "all" ? 0.22 : 1);
      bctx.strokeStyle = w >= 0 ? C.exc : C.inh;
      bctx.lineWidth = 0.5 + 3 * mag;

      // Detail view keeps pruned edges as faint ghosts, so the genome's full
      // wiring stays inspectable alongside what survived.
      if (pruned) {
        bctx.globalAlpha = 0.07;
        bctx.strokeStyle = C.dim;
        bctx.lineWidth = 0.8;
        bctx.setLineDash([2, 3]);
      }

      if (e.src === e.tgt) {
        mid[i] = { x: a.x, y: a.y - r * 2.35 };      // top of the loop
        if (shown[i]) {
          bctx.beginPath();
          bctx.arc(a.x, a.y - r * 1.5, r * 0.85, 0, 6.2832);
          bctx.stroke();
        }
      } else {
        const dx = b.x - a.x, dy = b.y - a.y;
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len;
        const sx = a.x + ux * r, sy = a.y + uy * r;
        const ex = b.x - ux * r, ey = b.y - uy * r;
        // Curve every edge the same way round so opposing pairs stay distinct.
        const cx = (sx + ex) / 2 - (ey - sy) * 0.18;
        const cy = (sy + ey) / 2 + (ex - sx) * 0.18;
        // Midpoint of the quadratic curve, not of the straight line, so the
        // marker sits on the stroke that is actually drawn.
        mid[i] = { x: 0.25 * sx + 0.5 * cx + 0.25 * ex,
                   y: 0.25 * sy + 0.5 * cy + 0.25 * ey };
        if (shown[i]) {
          bctx.beginPath();
          bctx.moveTo(sx, sy);
          bctx.quadraticCurveTo(cx, cy, ex, ey);
          bctx.stroke();
          if (!pruned) arrowHead(ex, ey, ex - cx, ey - cy, 3 + 3 * mag);
        }
      }
      bctx.setLineDash([]);

      // Frozen at the bound: a small cap at the midpoint. Structural rather
      // than momentary, so it belongs with the rest of the detail marks.
      if (frozen && detail) {
        bctx.globalAlpha = 0.5;
        bctx.fillStyle = w >= 0 ? C.exc : C.inh;
        bctx.beginPath();
        bctx.arc(mid[i].x, mid[i].y, Math.max(1.5, r * 0.11), 0, 6.2832);
        bctx.fill();
      }

      // Learned on this very tick: a neutral halo over the stroke. This is the
      // one mark that survives into clean view, because watching the network
      // configure itself is the point of the panel.
      if (learned && learned.has(i) && shown[i]) {
        bctx.globalAlpha = 0.9;
        bctx.strokeStyle = C.learn;
        bctx.lineWidth = 1.5;
        bctx.beginPath();
        bctx.arc(mid[i].x, mid[i].y, Math.max(4, r * 0.4), 0, 6.2832);
        bctx.stroke();
      }
    }
    bctx.setLineDash([]);
    bctx.globalAlpha = 1;

    drawModulation(mid, shown, fired, r);
    showBrainStats(S2.edges.length, nPruned, nFrozen, d);

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

  /* Modulatory connections: which neurons CHANGE a connection, as opposed to
     signalling through it.

     A modulator only has an effect on the beat its neuron fires -- that is the
     only time the weight moves. Drawing just those keeps roughly seven links on
     screen instead of all hundred, and what is shown is what is actually acting.
     Press 'm' to see the full wiring instead. */
  function drawModulation(mid, shown, fired, r) {
    if (modMode === "off") return;
    const showAll = modMode === "all";
    const dia = Math.max(2.5, r * 0.3);

    for (const [edgeIndex, mods] of bModByEdge) {
      const m0 = mid[edgeIndex];
      // An edge the signal layer left out is dead wiring; marking what
      // modulates it would annotate a connection that is not on screen.
      if (!m0 || !shown[edgeIndex]) continue;
      const anyActive = mods.some(m => fired(m.by));
      // Clean view marks a connection only while something is acting on it.
      // Asking for the full wiring with 'm' overrides that, as does detail.
      if (!anyActive && !showAll && !detail) continue;

      // A diamond marks every modulated connection, faint until something acts.
      bctx.globalAlpha = anyActive ? 0.95 : (showAll ? 0.6 : 0.16);
      bctx.fillStyle = C.dim;
      diamond(m0.x, m0.y, dia);
      bctx.fill();

      for (const m of mods) {
        const active = fired(m.by);
        if (!active && !showAll) continue;
        const from = nodePos(m.by, bCell.w, bCell.h, Math.min(bCell.w, bCell.h) * 0.14);
        const strength = Math.min(1, Math.abs(m.w) * 2);

        bctx.globalAlpha = active ? 0.3 + 0.6 * strength : 0.25 + 0.35 * strength;
        bctx.strokeStyle = m.w >= 0 ? C.exc : C.inh;
        bctx.lineWidth = active ? 1 + 1.5 * strength : 0.6 + 0.9 * strength;
        bctx.setLineDash(active ? [4, 3] : [3, 3]);
        bctx.beginPath();
        bctx.moveTo(from.x, from.y);
        bctx.lineTo(m0.x, m0.y);
        bctx.stroke();
        bctx.setLineDash([]);

        if (active) {
          bctx.fillStyle = m.w >= 0 ? C.exc : C.inh;
          diamond(m0.x, m0.y, dia * 0.85);
          bctx.fill();
        }
      }
    }
    bctx.globalAlpha = 1;
  }

  function diamond(x, y, s) {
    bctx.beginPath();
    bctx.moveTo(x, y - s);
    bctx.lineTo(x + s, y);
    bctx.lineTo(x, y + s);
    bctx.lineTo(x - s, y);
    bctx.closePath();
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
  // Run shape
  // ---------------------------------------------------------------
  // Four lanes of "did anything happen here" across the whole run. An evolved
  // agent tends to settle within the first few dozen ticks and then repeat one
  // action for thousands more, so the timeline on its own gives no hint that
  // everything worth seeing is a sliver at the left edge.

  const shapeCanvas = document.getElementById("shape");
  const sctx = shapeCanvas.getContext("2d");
  const SHAPE_LABEL_W = 46;
  const LANE_H = 7, LANE_GAP = 4;
  // Half the scrub thumb. The range input's travel is inset by this much at
  // each end, so the lanes have to be inset to match or a mark and the slider
  // position for the same tick would not line up.
  const THUMB = 7.5;

  /* Eating runs solid for the whole life of a camping agent, so its lane is a
     filled bar end to end. Put last it sits directly above the scrub track and
     the two read as one widget, which hides the fact that the scrub is a
     control at all. The sparse lanes go nearest the scrub instead. */
  const LANES = [
    { label: "eats", colour: C.food, ticks: new Set(eatTicks) },
    { label: "learns", colour: C.exc, ticks: decisionTicks(bLearned && bLearned.keys()) },
    { label: "thinks", colour: C.agent, ticks: decisionTicks(bThink) },
    { label: "moves", colour: C.text, ticks: new Set(moveTicks) },
  ];

  const SHAPE_H = LANES.length * LANE_H + (LANES.length - 1) * LANE_GAP;
  let shapeW = 0;

  /* Decision d produces world frame d+1, so a set of decisions has to be
     shifted before it can be drawn against the frame timeline. */
  function decisionTicks(source) {
    const out = new Set();
    if (!source) return out;
    for (const d of source) if (d + 1 < N) out.add(d + 1);
    return out;
  }

  function fitShape() {
    shapeW = Math.max(120, shapeCanvas.parentElement.clientWidth);
    sizeCanvas(shapeCanvas, sctx, shapeW, SHAPE_H);
  }

  function drawShape(n) {
    const x0 = SHAPE_LABEL_W + THUMB;
    const trackW = Math.max(1, shapeW - x0 - THUMB);
    const span = Math.max(1, N - 1);

    sctx.clearRect(0, 0, shapeW, SHAPE_H);
    sctx.font = "8px Consolas, monospace";
    sctx.textAlign = "right";
    sctx.textBaseline = "middle";

    LANES.forEach(function (lane, li) {
      const y = li * (LANE_H + LANE_GAP);
      sctx.fillStyle = C.dim;
      sctx.fillText(lane.label, SHAPE_LABEL_W - 6, y + LANE_H / 2);

      // An empty stretch has to read as "nothing happened here", not as a hole
      // in the drawing, so each lane carries a baseline for its full length.
      sctx.fillStyle = C.edge;
      sctx.fillRect(x0, y + LANE_H / 2 - 0.5, trackW, 1);

      sctx.fillStyle = lane.colour;
      for (const t of lane.ticks) {
        sctx.fillRect(x0 + (t / span) * trackW - 0.75, y, 1.5, LANE_H);
      }
    });

    sctx.globalAlpha = 0.9;
    sctx.fillStyle = C.warn;
    sctx.fillRect(x0 + (n / span) * trackW - 0.5, 0, 1, SHAPE_H);
    sctx.globalAlpha = 1;
  }

  /* Eating carries on for as long as the agent is alive and parked on a
     regrowing cell, so it says nothing about whether the run is still
     developing. Moving, deciding and learning do. */
  function describeShape() {
    const el = document.getElementById("shape-note");
    if (!el) return;

    // Last tick on which the run was still developing. A long quiet stretch
    // after it is the headline fact about an evolved elite; a short one just
    // means the last event happened to land near the end, so it is only worth
    // reporting once it covers a real part of the run.
    let last = -1;
    for (const lane of LANES.slice(1)) {
      for (const t of lane.ticks) if (t > last) last = t;
    }
    const quiet = last >= 0 ? (N - 1) - last : 0;

    const parts = [];
    // Against the decision count, not the frame count: frame 0 predates any
    // decision, so there is always one more frame than there are decisions.
    if (bThink) {
      parts.push("resolves " + bThink.size + "/"
                 + BRAIN.beats_per_tick.length);
    }
    if (quiet > N * 0.1) parts.push(quiet + " quiet");
    el.textContent = parts.join(" · ") || "";
  }

  /* Edge census under the brain panel. Clean view does not draw the pruned and
     saturated marks, so the counts keep that fact on the page rather than
     letting it disappear with the marks. */
  function showBrainStats(total, pruned, frozen, d) {
    const el = document.getElementById("brain-stats");
    if (!el) return;

    // Said in words, because a panel that holds still for a thousand ticks
    // otherwise reads as a broken render rather than as a converged network.
    let state = "";
    if (d >= 0 && bThink) {
      if (bFallback.has(d)) state = " · no candidate, action chosen at random";
      else if (bThink.has(d)) state = " · resolving";
      else state = " · settled, every beat identical";
    }

    el.textContent = total + " edges, " + pruned + " pruned, "
                   + frozen + " saturated" + state;
  }

  /* Keep the view-mode buttons and the key telling the same story as the
     canvas; both modes are also on the keyboard, so neither owns the state. */
  function syncModes() {
    document.getElementById("app").classList.toggle("detail", detail);

    const detailBtn = document.getElementById("detail-btn");
    if (detailBtn) detailBtn.classList.toggle("on", detail);

    const modBtn = document.getElementById("mod-btn");
    if (modBtn) {
      modBtn.textContent = "modulation: " + modMode;
      modBtn.classList.toggle("on", modMode !== "active");
    }
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
    const dIdx = decisionOf(f.t);
    const nLearned = (bLearned && bLearned.get(dIdx)) ? bLearned.get(dIdx).size : 0;
    document.getElementById("counters").textContent =
      "eats " + f.eats + "   dist " + f.distance + "   pos " + f.y + "," + f.x + meal
      + (nLearned ? "   " + nLearned + " weights moved" : "");

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
    drawShape(cur);
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

  function cycleModulation() {
    modMode = MOD_MODES[(MOD_MODES.indexOf(modMode) + 1) % MOD_MODES.length];
    syncModes();
    drawBrain(cur);
  }

  function toggleDetail() {
    detail = !detail;
    syncModes();
    drawBrain(cur);
  }

  document.getElementById("mod-btn").addEventListener("click", cycleModulation);
  document.getElementById("detail-btn").addEventListener("click", toggleDetail);

  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
    else if (e.key === "ArrowLeft") { setPlaying(false); show(cur - (e.shiftKey ? 10 : 1)); }
    else if (e.key === "ArrowRight") { setPlaying(false); show(cur + (e.shiftKey ? 10 : 1)); }
    else if (e.key === "Home") { setPlaying(false); show(0); }
    else if (e.key === "End") { setPlaying(false); show(N - 1); }
    else if (e.key === "m" || e.key === "M") cycleModulation();
    else if (e.key === "d" || e.key === "D") toggleDetail();
  });

  function relayout() {
    fitCanvas();
    fitShape();
    draw(cur);
    drawBrain(cur);
    drawShape(cur);
  }

  /* A canvas is sized from the box it sits in, but the canvas is also part of
     the layout that decides how big every other box is. One pass at startup
     therefore measures a page that has not settled: the shape canvas is still
     at its default 300x150, the transport is 117px too tall, and the world
     canvas ends up a third of the height it should have. Rather than guess an
     order that happens to converge, each box is observed and re-fitted
     whenever it actually changes size. */
  let boxSizes = new Map();
  const observer = new ResizeObserver(function (entries) {
    let changed = false;
    for (const e of entries) {
      const was = boxSizes.get(e.target);
      const now = e.contentRect.width + "x" + e.contentRect.height;
      // Re-fitting resizes the canvases, which can resize their boxes, which
      // would notify us again. Stopping on an unchanged size breaks that loop.
      if (was !== now) { boxSizes.set(e.target, now); changed = true; }
    }
    if (changed) relayout();
  });
  document.querySelectorAll(".canvas-box").forEach(b => observer.observe(b));
  observer.observe(shapeCanvas.parentElement);

  // devicePixelRatio can change without any box changing, when the window
  // moves to a monitor at a different scale.
  window.addEventListener("resize", relayout);

  resetGrid();
  fitShape();
  fitCanvas();
  drawMarkers();
  describeShape();
  syncModes();
  show(0);
})();
