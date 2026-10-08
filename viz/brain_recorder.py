"""Brain recorder for replay export.

Records the network state once per brain beat, exposing the same interface as
the Qt brain renderer so it can be attached without any change to the
simulation:

    brain_module._brain_renderer = BrainFrameRecorder(worm)

`decide()` then drives it exactly as it would a renderer.

Nothing here mutates brain, world, worm or rng state.
"""

import re


# Written a few lines above the call site in decide().
_PROPAGATION_RE = re.compile(r"PROPAGATION PHASE \((\d+)/(\d+)\)")
_STABILITY_RE = re.compile(r"STABILITY PHASE \((\d+)\): candidates=(\S*)")


class BrainFrameRecorder:
    """Captures the network once per brain beat.

    `decide()` runs many beats per simulation tick -- a propagation phase, then
    a stability phase -- and calls this after computing the next state but
    before committing it. The values read here are therefore the committed
    state at the start of that beat, which is what should be displayed.

    Storage follows what the data does. Activations are binary, so a whole
    network packs into one integer per beat. Weights move rarely: a connection
    only changes while one of its modulators is active, and stops moving
    permanently once it saturates at +/-1, since the plasticity term carries a
    |w|(1-|w|) factor. They are therefore stored as sparse change events rather
    than a snapshot per beat.

    Args:
        worm: Worm instance; read only to detect decision boundaries
    """

    def __init__(self, worm=None):
        self.worm = worm
        self.static = None

        self.beats_per_tick = []       # number of beats in each decision
        self.warmup_per_tick = []      # beats of propagation before stability
        self.candidates_per_tick = []  # bitmask of candidate output neurons
        self.activations = []          # bitmask per beat, flat across all ticks
        self.weight_changes = []       # [beat_index, edge_index, new_weight]

        self._last_weights = None
        self._beats_this_tick = 0
        self._warmup = 0
        self._candidates = 0

    # --- renderer interface ---------------------------------------

    def draw(self, state, brain_tick=0, decision_status="", sense=None):
        if self.static is None:
            self._capture_topology(state)

        # brain_tick restarts at 0 for each decision, so a 0 after beats have
        # been seen means the previous decision just ended.
        if brain_tick == 0 and self._beats_this_tick:
            self._close_tick()

        self._read_status(decision_status)

        mask = 0
        for neuron in state.neurons:
            if neuron.activity > 0.0:
                mask |= 1 << neuron.id
        self.activations.append(mask)

        # Compare at the stored precision: a change too small to survive
        # rounding would otherwise be recorded as a no-op entry.
        beat_index = len(self.activations) - 1
        for edge_index, conn in enumerate(_iter_connections(state)):
            weight = round(float(conn.weight), 4)
            if weight != self._last_weights[edge_index]:
                self._last_weights[edge_index] = weight
                self.weight_changes.append([beat_index, edge_index, weight])

        self._beats_this_tick += 1

    def wait_frame(self):
        """No pacing; recording runs at full speed."""
        return

    def finish(self):
        """Close the final decision; no later beat arrives to trigger it."""
        if self._beats_this_tick:
            self._close_tick()

    # --- internals ------------------------------------------------

    def _close_tick(self):
        self.beats_per_tick.append(self._beats_this_tick)
        self.warmup_per_tick.append(self._warmup)
        self.candidates_per_tick.append(self._candidates)
        self._beats_this_tick = 0
        self._candidates = 0

    def _read_status(self, status):
        """Extract the phase boundary and candidate set from the status string.

        Falls back to leaving the values untouched if the wording ever changes,
        rather than failing the export.
        """
        match = _PROPAGATION_RE.search(status)
        if match:
            self._warmup = int(match.group(2))
            return

        match = _STABILITY_RE.search(status)
        if match:
            names = match.group(2)
            if names and names != "NONE":
                for part in names.split(","):
                    part = part.strip()
                    if part.isdigit():
                        self._candidates |= 1 << int(part)

    def _capture_topology(self, state):
        """Record the wiring once; it does not change during a run."""
        output_map = {int(k): v for k, v in (state.output_mapping or {}).items()}

        sensory_targets = set()
        edges = []
        for conn, target in _iter_connections(state, with_target=True):
            source = conn.source
            if hasattr(source, "key"):
                src_ref = "in:" + str(source.key)
                sensory_targets.add(int(target.id))
            else:
                src_ref = int(source.id)
            edges.append({
                "src": src_ref,
                "tgt": int(target.id),
                "w0": round(float(conn.weight), 4),
                "rel": round(float(conn.reliability), 4),
                "plastic": bool(conn.modulating_inputs),
            })

        modulation = []
        for edge_index, conn in enumerate(_iter_connections(state)):
            for mod_neuron, mod_weight in (conn.modulating_inputs or []):
                modulation.append({
                    "edge": edge_index,
                    "by": int(mod_neuron.id),
                    "w": round(float(mod_weight), 4),
                })

        neurons = []
        for neuron in state.neurons:
            if neuron.id in output_map:
                role = "output"
            elif int(neuron.id) in sensory_targets:
                role = "sensory"
            else:
                role = "hidden"
            neurons.append({
                "id": int(neuron.id),
                "threshold": round(float(neuron.threshold), 4),
                "tonic": round(float(neuron.tonic_level), 4),
                "role": role,
            })

        self.static = {
            "neurons": neurons,
            "inputs": [str(src.key) for src in state.input_sources],
            "edges": edges,
            "modulation": modulation,
            "output_mapping": output_map,
            "eta": round(float(state.eta), 6),
        }
        self._last_weights = [edge["w0"] for edge in edges]

    # --- output ---------------------------------------------------

    def to_dict(self):
        """Return the recording as plain JSON-serialisable data."""
        if self.static is None:
            raise RuntimeError("Nothing recorded; the brain renderer was never called")
        return {
            "static": self.static,
            "beats_per_tick": self.beats_per_tick,
            "warmup_per_tick": self.warmup_per_tick,
            "candidates_per_tick": self.candidates_per_tick,
            "activations": self.activations,
            "weight_changes": self.weight_changes,
        }

    def summary(self):
        """One-line description of the recording."""
        if self.static is None:
            return "BrainFrameRecorder: nothing recorded"
        plastic = sum(1 for edge in self.static["edges"] if edge["plastic"])
        return (
            f"beats={len(self.activations)} "
            f"decisions={len(self.beats_per_tick)} "
            f"neurons={len(self.static['neurons'])} "
            f"edges={len(self.static['edges'])} "
            f"plastic={plastic} "
            f"weight_changes={len(self.weight_changes)} "
            f"eta={self.static['eta']}"
        )


def _iter_connections(state, with_target=False):
    """Connections in a stable order: by target neuron, then incoming order.

    Every edge index used elsewhere refers to this ordering. The wiring is
    built once in init_brain and never altered during a run, so it is stable.
    """
    for neuron in state.neurons:
        for conn in neuron.incoming:
            yield (conn, neuron) if with_target else conn
