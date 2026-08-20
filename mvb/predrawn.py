"""Pre-drawn randomness sources for deterministic cross-implementation testing.

Test-only. Nothing here is imported on the normal simulation path unless a config
enables it explicitly (``experiment.predrawn_randomness.enabled``).

Why
---
The tensor/EvoTorch port cannot reproduce the scalar implementation's RNG *stream*:
it draws a dense ``(P, R, n)`` noise tensor every brain tick, while the scalar code
draws per-neuron and only for neurons with ``noise_level > 0``, inside a loop that
exits early. A sequential stream would desynchronise on the first early-exiting
``decide()``.

So instead of replaying a recorded stream, both implementations consume the *same
structurally indexed bundle*:

    neuron_noise[world_tick, brain_tick, neuron_id]     standard normal
    decision_uniform[world_tick]                        uniform [0, 1)
    food_uniform[seed_event, y, x]                      uniform [0, 1)

Indexed by position rather than draw order, so it does not matter how many values
each implementation actually consumes.

How it plugs in
---------------
``PredrawnRandomness`` is a drop-in for ``np.random.Generator`` across the only three
methods the simulation uses:

    .normal(loc, scale)   Neuron.compute_input
    .integers(k)          _stable_outputs_to_decision / _get_random_decision
    .random(shape)        feeding.setup_food_initially

so no production call site changes. One object stands in for all three RNGs; the
values still come from three independently seeded streams, so RNG scopes are
preserved in substance.

The world-tick index is read from ``worm.ticks``, which is incremented *after*
``step_day`` and therefore holds the current 0-based tick throughout ``decide()``.
That is why no hook is needed in ``simulate_run``.
"""

from dataclasses import dataclass

import numpy as np


# ============================================================
# Bundle
# ============================================================

@dataclass
class PredrawnBundle:
    """All randomness one run will consume, indexed structurally."""

    neuron_noise: np.ndarray      # (T, K, n)  standard normal
    decision_uniform: np.ndarray  # (T,)       uniform [0, 1)
    food_uniform: np.ndarray      # (E, H, W)  uniform [0, 1)

    @property
    def max_ticks(self) -> int:
        return self.neuron_noise.shape[0]

    @property
    def max_brain_ticks(self) -> int:
        return self.neuron_noise.shape[1]

    @property
    def n_neurons(self) -> int:
        return self.neuron_noise.shape[2]

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            neuron_noise=self.neuron_noise,
            decision_uniform=self.decision_uniform,
            food_uniform=self.food_uniform,
        )

    @classmethod
    def load(cls, path) -> "PredrawnBundle":
        with np.load(path) as f:
            return cls(
                neuron_noise=f["neuron_noise"],
                decision_uniform=f["decision_uniform"],
                food_uniform=f["food_uniform"],
            )


def make_bundle(
    *,
    run_seed,
    noise_seed,
    decision_seed,
    max_ticks: int,
    max_brain_ticks: int,
    n_neurons: int,
    grid_shape: tuple,
    n_seed_events: int,
) -> PredrawnBundle:
    """Build one run's bundle from the three seeds that run already uses.

    Seeds are taken unchanged from ``run_seeds`` / ``seeds_noise_variant`` /
    ``seeds_decision_variant``, so the common-random-numbers structure is preserved:
    ``run_seed`` is shared across the population, the other two are per-variant.

    ``neuron_noise`` holds *standard normals*; the scale is applied at the use site as
    ``loc + scale * v``. This is bit-identical to ``rng.normal(loc, scale)`` and matches
    the tensor side's ``randn_like(x) * noise_level``.
    """
    if max_ticks <= 0 or max_brain_ticks <= 0 or n_neurons <= 0:
        raise ValueError(
            f"make_bundle needs positive dimensions, got max_ticks={max_ticks}, "
            f"max_brain_ticks={max_brain_ticks}, n_neurons={n_neurons}"
        )
    if n_seed_events <= 0:
        raise ValueError(f"n_seed_events must be positive, got {n_seed_events}")

    rng_noise = np.random.default_rng(noise_seed)
    rng_decision = np.random.default_rng(decision_seed)
    rng_world = np.random.default_rng(run_seed)

    return PredrawnBundle(
        neuron_noise=rng_noise.standard_normal((max_ticks, max_brain_ticks, n_neurons)),
        decision_uniform=rng_decision.random(max_ticks),
        food_uniform=rng_world.random((n_seed_events, *grid_shape)),
    )


# ============================================================
# Source
# ============================================================

class PredrawnRandomness:
    """Drop-in stand-in for ``np.random.Generator`` backed by a ``PredrawnBundle``.

    Must be bound with :meth:`bind` before use. Every failure mode raises loudly
    rather than falling back, per the project's no-silent-defaults rule.
    """

    def __init__(self, bundle: PredrawnBundle):
        self._bundle = bundle
        self._worm = None
        self._drawing = None      # neuron ids that actually draw noise, in id order
        self._world_tick = -1
        self._brain_tick = 0
        self._cursor = 0          # index into self._drawing within the current brain tick
        self._decision_tick = -1  # guards the one-draw-per-world-tick invariant
        self._food_cursor = 0

    # ---------- binding ----------

    def bind(self, worm, brain_state) -> "PredrawnRandomness":
        """Bind to the worm (for the world-tick index) and the brain state.

        Must be called *after* ``init_brain``, since the draw schedule is read off the
        freshly built neurons.
        """
        neurons = brain_state.neurons
        if len(neurons) != self._bundle.n_neurons:
            raise ValueError(
                f"bundle was built for {self._bundle.n_neurons} neurons but the brain "
                f"has {len(neurons)}"
            )

        # Only neurons with noise_level > 0 call .normal(); indexing by real neuron id
        # keeps us aligned with the tensor side's dense (P, R, n) draw even if levels
        # are mixed (see plan Step 0, design decision 6).
        self._drawing = [i for i, n in enumerate(neurons) if n.noise_level > 0.0]

        self._worm = worm
        self._world_tick = -1
        self._brain_tick = 0
        self._cursor = 0
        self._decision_tick = -1
        self._food_cursor = 0
        return self

    def _require_bound(self) -> None:
        if self._worm is None:
            raise RuntimeError("PredrawnRandomness used before bind()")

    # ---------- np.random.Generator surface ----------

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Stand-in for ``rng.normal(0.0, neuron.noise_level)``."""
        self._require_bound()
        if size is not None:
            raise NotImplementedError(
                "PredrawnRandomness.normal() is scalar-only; the simulation never "
                "requests an array"
            )
        if not self._drawing:
            raise RuntimeError(
                "normal() called but no neuron has noise_level > 0 — draw schedule is "
                "empty, so the neuron index cannot be resolved"
            )

        world_tick = self._worm.ticks
        if world_tick != self._world_tick:
            # new world tick: decide() restarts its brain-tick loop from 0
            self._world_tick = world_tick
            self._brain_tick = 0
            self._cursor = 0
        elif self._cursor >= len(self._drawing):
            # every drawing neuron has been served this brain tick -> advance
            self._brain_tick += 1
            self._cursor = 0

        if world_tick >= self._bundle.max_ticks:
            raise IndexError(
                f"world tick {world_tick} exceeds bundle max_ticks="
                f"{self._bundle.max_ticks}"
            )
        if self._brain_tick >= self._bundle.max_brain_ticks:
            raise IndexError(
                f"brain tick {self._brain_tick} at world tick {world_tick} exceeds "
                f"bundle max_brain_ticks={self._bundle.max_brain_ticks}; raise "
                f"predrawn_randomness.max_brain_ticks"
            )

        neuron_id = self._drawing[self._cursor]
        self._cursor += 1
        value = self._bundle.neuron_noise[world_tick, self._brain_tick, neuron_id]
        return loc + scale * value

    def integers(self, low, high=None, size=None, dtype=None, endpoint=False):
        """Stand-in for ``rng.integers(k)`` in the decision path.

        Returns ``low + int(u * (high - low))`` from a stored uniform rather than
        numpy's integer algorithm, so the tensor side can reproduce it with
        ``(u * k).long()``. This deliberately does *not* match ``np.random.Generator``.
        """
        self._require_bound()
        if size is not None or endpoint:
            raise NotImplementedError(
                "PredrawnRandomness.integers() supports only the scalar half-open form "
                "used by the decision path"
            )

        lo, hi = (0, low) if high is None else (low, high)
        span = int(hi) - int(lo)
        if span <= 0:
            raise ValueError(f"integers() needs a positive span, got [{lo}, {hi})")

        world_tick = self._worm.ticks
        if world_tick == self._decision_tick:
            raise RuntimeError(
                f"second decision draw at world tick {world_tick}; the bundle assumes "
                f"decide() draws at most once per world tick"
            )
        if world_tick >= self._bundle.max_ticks:
            raise IndexError(
                f"world tick {world_tick} exceeds bundle max_ticks="
                f"{self._bundle.max_ticks}"
            )
        self._decision_tick = world_tick

        u = self._bundle.decision_uniform[world_tick]
        return int(lo) + int(u * span)

    def random(self, size=None, dtype=None, out=None):
        """Stand-in for ``world.rng_world_run.random(food.shape)`` in food seeding."""
        self._require_bound()
        if out is not None or dtype is not None:
            raise NotImplementedError(
                "PredrawnRandomness.random() supports only the plain shape form used by "
                "setup_food_initially"
            )

        n_events = self._bundle.food_uniform.shape[0]
        if self._food_cursor >= n_events:
            raise IndexError(
                f"food seeding event {self._food_cursor} exceeds the {n_events} events "
                f"the bundle was built for (1 initial + one per phase switch)"
            )

        grid = self._bundle.food_uniform[self._food_cursor]
        self._food_cursor += 1

        if size is not None and tuple(np.shape(size)) == () and isinstance(size, int):
            size = (size,)
        if size is not None and tuple(size) != grid.shape:
            raise ValueError(
                f"food seeding asked for shape {tuple(size)} but the bundle holds "
                f"{grid.shape}"
            )
        return grid
