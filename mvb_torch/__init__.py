"""Batched tensor port of the Minimum Viable Brain simulation (EvoTorch target).

See ``plan_evotorch.md`` for the full design.  This package is additive: nothing
in ``mvb/`` imports from here, and the scalar simulation remains the reference
implementation against which everything in this package is validated.

Binding rule (plan_evotorch.md §5.1) -- the one invariant every module here obeys
-------------------------------------------------------------------------------
The batch has two independent dimensions:

    P   population size   (number of genomes / variants in a generation)
    R   runs per genome   (independent worlds a genome is evaluated on)

**Genome-derived tensors carry the P dimension only and are never expanded to
the run dimension.**  Anything that is a property of the genome -- the modulation
tensor ``Mod``, the sign mask ``Wsign``, the reliability matrix ``Rel``, the
tonic vector, ``eta`` -- is identical across all R runs of a genome, so
materialising it R times would waste memory linearly in R for no gain.  The run
dimension is introduced only by *state*: the live weight matrix, neuron
activities, world grids, worm positions.

Broadcasting carries the rest.  Where a contraction is needed, the ``p`` index is
matched in both operands so the large operand never has to be expanded, e.g.

    torch.einsum('pkij,prk->prij', Mod, act)

which contracts the (P, n, n, n) modulation tensor against (P, R, n) activities
without ever materialising a (P, R, n, n, n) intermediate.

Device / dtype policy (forced by hardware, see reproducibility.md §3)
--------------------------------------------------------------------
    purpose                        device        dtype
    ---------------------------    -----------   ---------
    reference / equivalence tests  cpu           float64    (matches the scalar sim)
    benchmarks / production        mps, cuda     float32

Apple's Metal backend has no float64 type at all, so float64-on-MPS is rejected
loudly rather than silently downcast -- silent downcasting would turn an exact
equivalence test into an approximate one without saying so.

Index convention
----------------
Connection matrices are indexed ``[src, tgt]``, matching ``genome["connection_weights"]``
and the rest of this repository.  (Note this is transposed relative to
``evotorch_example/ctrnn.py``, which uses ``[post, pre]``.)
"""

from .genome_codec import GenomeBatch, encode_genomes

__all__ = ["GenomeBatch", "encode_genomes"]
