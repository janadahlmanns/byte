# Reading the replay viewer

Reference card for an exported `.html` page. World on the left, brain on the right,
timeline underneath.

## World

| Mark | Meaning |
|---|---|
| Green square | Food |
| Pale green dot | Eaten, regrowing |
| Green ring | A meal, flashes for 6 ticks |
| Gold dot | The agent, with a fading trail |
| Gold line from centre | Heading |
| Arc around the agent | Energy; amber below 25% |
| 5 gold outlined cells | What the agent senses; filled when that cell holds food |
| Red crossed disc | Dead |
| Gold wash | Coverage — press `c` |

**Coverage** runs pale to dark with time spent on a cell. The palest tint marks cells the
agent reaches only later in the run.

## Brain

| Mark | Meaning |
|---|---|
| Red edge | Excitatory weight |
| Blue edge | Inhibitory weight |
| Grey edge | Sensor wiring — fixed, never learned |
| Edge width | \|weight\| |
| Edge opacity | \|weight\| × reliability; brighter when the source fires |
| Loop above a node | Self-connection |
| Filled black node | Firing on this beat |
| Dashed black ring | Candidate for this decision |
| Solid ring on an edge | Weight moved on this tick |
| Diamond | That connection is modulated |
| Dashed line to a diamond | A modulator acting right now |
| Dotted ghost edge | Pruned (detail view) |
| Small cap on an edge | Saturated at \|w\| = 1 (detail view) |

Colour means **sign only**. Anything that is not a sign — a weight moving, a neuron firing —
is drawn in neutral ink.

## Beats

One tick of the world contains many beats of the brain. `decide()` propagates the network
repeatedly until it settles, and all of that happens before the agent moves.

| Mark | Meaning |
|---|---|
| Pale slots | Propagation — no decision possible yet |
| Grey slots | Stability — a decision can be taken |
| Vertical line | Where propagation ends |
| Last slot, solid | The rule fired here |
| Last slot, hollow | No majority; the action was chosen at random |
| Bar above a slot | The beat you are looking at |

## Timeline

Four lanes sharing one axis with the scrub bar, so a dense patch sits above the stretch of
run that produced it.

| Lane | Meaning |
|---|---|
| eats | A meal was taken |
| learns | A weight moved |
| thinks | The tick resolved through the rule |
| moves | The agent changed cell |

The amber line is the current tick. Ticks on the scrub bar mark single events: meals,
learning, movement, phase switches, death, and ticks that fell back to a random action.

## Readouts

- **Under the world** — energy is the arc; with coverage on, how many cells are reached so
  far, over the whole run, and the busiest.
- **Under the brain** — edge count, how many are pruned and saturated, and whether this tick
  is resolving, settled, or was never resolved.
- **Status bar** — energy, the five sensed cells as a compass, meals, distance, position, and
  how many ticks resolved out of the total.

## Controls

| Key | Action |
|---|---|
| `space` | Play / pause |
| `←` `→` | Step one tick (`shift` for 10) |
| `↑` `↓` | Step one beat inside the tick |
| `home` `end` | Jump to start / end |
| `c` | Coverage |
| `d` | Detail — adds pruned and saturated marks |
| `m` | Modulation: active → all → off |

| Button | Action |
|---|---|
| `▶` `‹` `›` | Play, step back, step forward |
| `\|‹` `›\|` | Previous / next event |
| Speed | Ticks per second |
| save frame | This tick as a high-resolution PNG |
| record | The whole run as an MP4, at the selected speed |

Modulation defaults to **active**, showing only modulators that are firing — the only beat on
which they change anything. `all` shows the full wiring and pushes the network back.
