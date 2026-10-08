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

  function fitCanvas() {
    const stage = document.getElementById("stage");
    const avail = Math.min(stage.clientWidth - 20, stage.clientHeight - 20);
    const size = Math.max(200, avail);
    cell = size / Math.max(W, H);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(W * cell * dpr);
    canvas.height = Math.round(H * cell * dpr);
    canvas.style.width = W * cell + "px";
    canvas.style.height = H * cell + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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

  window.addEventListener("resize", function () { fitCanvas(); draw(cur); });

  resetGrid();
  fitCanvas();
  drawMarkers();
  show(0);
})();
