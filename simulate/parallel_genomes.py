"""Genome creation and mutation on all allowed cores (plan Step 27).

  generate()  the initial population: genome i = generator(cfg, rng_seed=seeds[i]).
              Each genome has its own seed already, so this is identical to the serial
              loop whatever the number of workers.
  mutate()    children with one seed each (`mutation_rng: per_child`): child i =
              mutate(parents[parent_index[i]], rate, strength, default_rng(seed[i])).
              A child depends only on its parent and its seed, so the result is the
              same for any number of workers and any chunking.

The workers are `spawn`ed (never fork a process that may hold a CUDA context), import
only mvb.genome and numpy, and live for the whole run. Below MIN_PARALLEL items the
work runs serially in this process: the result is identical, only faster. A failure
in a worker raises here, naming the item; nothing is retried.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

MIN_PARALLEL = 64          # fewer items: serial (pickling would cost more than it saves)
CHUNKS_PER_WORKER = 4      # small enough to balance, large enough to amortise pickling


def _generate_chunk(args):
    generator, cfg, start, seeds = args
    out = []
    for k, s in enumerate(seeds):
        try:
            out.append(generator(cfg, rng_seed=int(s)))
        except Exception as e:  # noqa: BLE001 -- re-raised with the genome's index
            raise RuntimeError(f"genome {start + k} (rng_seed {int(s)}): {type(e).__name__}: {e}") from e
    return out


def _mutate_chunk(args):
    mutation_function, parents, start, items, rate, strength = args
    out = []
    for k, (p, seed) in enumerate(items):
        try:
            out.append(mutation_function(parents[p], rate, strength, np.random.default_rng(int(seed))))
        except Exception as e:  # noqa: BLE001 -- re-raised with the child's index
            raise RuntimeError(f"child {start + k} (parent {p}, seed {int(seed)}): "
                               f"{type(e).__name__}: {e}") from e
    return out


def _modules_probe(_):
    import sys
    return sorted(m for m in ("torch", "mvb_torch") if m in sys.modules)


def _chunks(n, workers):
    """Contiguous [start, stop) ranges covering range(n)."""
    k = max(1, min(n, workers * CHUNKS_PER_WORKER))
    bounds = np.linspace(0, n, k + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


class GenomePool:
    """A lazily started process pool of `workers` processes (None or < 2: serial)."""

    def __init__(self, workers):
        self.workers = int(workers or 1)
        self._ex = None

    @property
    def parallel(self):
        return self.workers >= 2

    def _executor(self):
        if self._ex is None:
            self._ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"))
        return self._ex

    def _run(self, fn, jobs):
        try:
            parts = list(self._executor().map(fn, jobs))
        except Exception:
            self.close()
            raise
        return [g for part in parts for g in part]

    def generate(self, generator, cfg, seeds):
        seeds = [int(s) for s in seeds]
        if not self.parallel or len(seeds) < MIN_PARALLEL:
            return _generate_chunk((generator, cfg, 0, seeds))
        return self._run(_generate_chunk, [(generator, cfg, a, seeds[a:b])
                                           for a, b in _chunks(len(seeds), self.workers)])

    def mutate(self, mutation_function, parents, parent_index, seeds, rate, strength):
        items = [(int(p), int(s)) for p, s in zip(parent_index, seeds)]
        if len(items) != len(parent_index) or len(items) != len(seeds):
            raise ValueError("parent_index and seeds must have the same length")
        if not self.parallel or len(items) < MIN_PARALLEL:
            return _mutate_chunk((mutation_function, parents, 0, items, rate, strength))
        # Children are in parent order, so a chunk needs only its own few parents: send
        # just those ({index: genome}), not all of them (large at n = 50).
        jobs = []
        for a, b in _chunks(len(items), self.workers):
            need = {p: parents[p] for p, _s in items[a:b]}
            jobs.append((mutation_function, need, a, items[a:b], rate, strength))
        return self._run(_mutate_chunk, jobs)

    def worker_modules(self):
        """Which of torch / mvb_torch a worker has imported (tests: must be none)."""
        return list(self._executor().map(_modules_probe, [0]))[0]

    def close(self):
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=True)
            self._ex = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def per_child_seeds(rng, offspring_counts):
    """(parent_index, seeds) of one generation's children, in child order: elite 0's
    children first, as in the serial loop. All seeds are drawn here, at once."""
    parent_index = [e for e, c in enumerate(offspring_counts) for _ in range(c)]
    seeds = rng.integers(0, 2 ** 63, size=len(parent_index), dtype=np.uint64)
    return parent_index, seeds
