<!doctype html>
<!-- Template, not a page. The __TOKEN__ placeholders below are filled in by
     viz/export_viewer.py, which writes a complete standalone .html elsewhere.
     Opening this file directly shows an unstyled skeleton. -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<div id="app">

  <header>
    <h1>Byte <span class="accent">/</span> replay</h1>
    <div class="meta">__META__</div>
  </header>

  <div id="stage">

    <section class="panel">
      <h2>World</h2>
      <div class="canvas-box"><canvas id="world"></canvas></div>
      <div class="key">
        <span><i class="sw box" style="background:var(--blue)"></i>food</span>
        <span><i class="sw dot" style="background:var(--blue-mid)"></i>regrowing</span>
        <span><i class="sw ring" style="border-color:var(--blue)"></i>eaten</span>
        <span><i class="sw dot" style="background:var(--gold)"></i>agent &amp; trail</span>
        <span><i class="sw box hollow" style="border-color:var(--gold-light)"></i>sensed</span>
        <span class="cov-only"><i class="sw box" style="background:var(--gold-mid)"></i>visited</span>
      </div>
      <p class="key-note">
        Energy is the arc around the agent. <span id="cov-stats"></span>
      </p>
    </section>

    <section class="panel">
      <h2>Brain</h2>
      <div class="canvas-box"><canvas id="brain"></canvas></div>

      <!-- The beats inside the current tick. decide() runs a propagation phase
           and then a stability phase, and only a stability beat can commit an
           action, so which phase a beat belongs to is drawn explicitly. -->
      <div class="beat-row">
        <canvas id="beats"></canvas>
        <div class="beat-note" id="beat-note"></div>
      </div>

      <div class="key" id="brain-key">
        <span><i class="sw line" style="background:var(--gold-light)"></i>excitatory</span>
        <span><i class="sw line" style="background:var(--blue)"></i>inhibitory</span>
        <span><i class="sw dot" style="background:var(--gold)"></i>firing</span>
        <span><i class="sw dia"></i>modulated</span>
        <span><i class="sw line dash"></i>modulator active</span>
        <span><i class="sw ring" style="border-color:var(--text)"></i>weight moved</span>
        <span><i class="sw ring dash" style="border-color:var(--gold-light)"></i>candidate</span>
        <span class="detail-only"><i class="sw line dotted"></i>pruned</span>
        <span class="detail-only"><i class="sw pip"></i>saturated</span>
      </div>
      <p class="key-note">
        Width and opacity carry |weight| x reliability. <span id="brain-stats"></span>
      </p>
    </section>

  </div>

  <div id="transport">
    <button id="play" class="primary" title="Play (space)">&#9658;</button>
    <button id="prev" title="Back one tick (left arrow)">&lsaquo;</button>
    <button id="next" title="Forward one tick (right arrow)">&rsaquo;</button>
    <button id="prev-ev" title="Previous event">|&lsaquo;</button>
    <button id="next-ev" title="Next event">&rsaquo;|</button>

    <!-- Where anything at all happens, across the whole run. Evolved runs tend
         to converge early and then repeat one action for thousands of ticks;
         without this the timeline gives no hint that the interesting part is a
         sliver at the left. The lanes and the scrub share one x-axis, so a
         dense patch sits directly above the stretch of run that produced it. -->
    <div class="timeline">
      <canvas id="shape"></canvas>
      <div class="scrub">
        <div id="markers"></div>
        <input id="seek" type="range" min="0" max="1" value="0" step="1" aria-label="Timeline">
      </div>
    </div>

    <div class="readout" id="tick">0 / 0</div>

    <select id="speed" title="Playback speed" aria-label="Playback speed">
      <option value="2">2 t/s</option>
      <option value="4">4 t/s</option>
      <option value="8" selected>8 t/s</option>
      <option value="20">20 t/s</option>
      <option value="60">60 t/s</option>
    </select>
  </div>

  <div id="status">
    <div class="energy-wrap">
      <div class="energy-track"><div id="energy-fill"></div></div>
      <div class="energy-label">
        <span id="energy-text">ENERGY</span>
        <span id="alive-state">ALIVE</span>
      </div>
    </div>

    <div class="compass" title="The five cells the agent can sense">
      <div class="c-n">N</div>
      <div class="c-w">W</div>
      <div class="c-c">&#9679;</div>
      <div class="c-e">E</div>
      <div class="c-s">S</div>
    </div>

    <div class="readout" id="counters">eats 0   dist 0</div>

    <div class="readout" id="shape-note"></div>

    <div class="modes">
      <button id="cov-btn" title="Shade cells by how often the agent occupied them (c)">coverage</button>
      <button id="detail-btn" title="Show pruned and saturated connections (d)">detail</button>
      <button id="mod-btn" title="Cycle modulation view (m)">modulation: active</button>
    </div>
  </div>

  <div class="hint">
    <kbd>space</kbd> play/pause &nbsp; <kbd>&larr;</kbd><kbd>&rarr;</kbd> step
    (<kbd>shift</kbd> x10) &nbsp; <kbd>&uarr;</kbd><kbd>&darr;</kbd> step beats
    &nbsp; <kbd>home</kbd>/<kbd>end</kbd> jump &nbsp;
    <kbd>c</kbd> coverage &nbsp; <kbd>m</kbd> modulation view &nbsp;
    <kbd>d</kbd> detail
  </div>

</div>

<script>window.RUN_DATA = __DATA__;</script>
<script>__JS__</script>
</body>
</html>
