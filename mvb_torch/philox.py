"""Counter-based random numbers keyed by each run's own seeds (plan_evotorch.md Step 8, R1).

Why
---
Live mode used to draw from one shared `torch.Generator` and hand the numbers out by
slot position, so the numbers a run got depended on the batch it ran in, and the
per-run seeds stored in the HDF5 played no part. A single run therefore could not be
replayed from the file. Here every random number is a pure function of its address:

    noise     z[run, world tick t, brain tick k, neuron i] = f(noise_seed[run],    t, k, i)
    decision  u[run, world tick t]                         = f(decision_seed[run], t)

so a run gets the same numbers whatever batch, slot, width or schedule it runs in, and
`noise_seed` / `decision_seed` (stored per run in the HDF5) fully determine them.

Philox4x32, 7-10 rounds
-----------------------
`f` is Philox4x32 (Salmon et al. 2011, "Parallel random numbers: as easy as 1, 2, 3";
the Random123 library). The round count is the config key
`experiment.evaluator.philox_rounds` (plan Step 8, R1.1), required in live mode and
limited to 7-10: 7 is the published minimum that passes TestU01 BigCrush, 10 the
authors' default (also cuRAND's, JAX's and PyTorch's CUDA generator's). It changes
every number, so a file can only be replayed with the round count that produced it --
the count is stored with the config in the HDF5, and run_batch's replay checks it.
Every function takes `rounds` explicitly; there is no default.

It is written with plain int64 tensor operations holding 32-bit lanes. Products of two
32-bit values can exceed 2**63; int64 multiplication wraps around identically on CPU and
MPS (verified), and only the low 64 bits are used, so the result is exact.

Layout of the address
---------------------
One Philox call maps a 128-bit counter and a 64-bit key to four 32-bit words.

    noise     key = (noise_seed, 0),    counter = (t, 3*k + block, 0, STREAM_NOISE)
              blocks 0..2 give 12 words -> 12 uniforms -> 6 Box-Muller pairs -> 12
              normals, of which the first n = 11 are used (neuron i <- normal i).
    decision  key = (decision_seed, 0), counter = (t, 0, 0, STREAM_DECISION), word 0.

Uniforms use the top 24 bits of a word: exact in float32 and float64 alike, so the
integer -> uniform step is identical on every device and dtype. Box-Muller's log/cos/sin
are floating-point, so normals can differ in the last bit between device types; that,
like the float32 brain arithmetic, is why exact replay needs the same device type and
dtype as the original run.
"""

from __future__ import annotations

import math

import torch

ALLOWED_ROUNDS = range(7, 11)     # 7..10 -- see the module docstring

_M32 = 0xFFFFFFFF
_MULT_A, _MULT_B = 0xD2511F53, 0xCD9E8D57     # Philox4x32 multipliers
_WEYL_A, _WEYL_B = 0x9E3779B9, 0xBB67AE85     # key schedule increments

STREAM_NOISE = 0
STREAM_DECISION = 1
_BLOCKS_PER_BRAIN_TICK = 3                    # 12 words >= 11 neurons, in pairs
_TWO_PI = 2.0 * math.pi
_INV_2_24 = 2.0 ** -24


def check_rounds(rounds) -> int:
    """The round count, validated: an int in ALLOWED_ROUNDS, or a loud error."""
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds not in ALLOWED_ROUNDS:
        raise ValueError(
            f"[ERROR] philox_rounds must be an int from {ALLOWED_ROUNDS.start} to "
            f"{ALLOWED_ROUNDS.stop - 1}, got {rounds!r}. 7 is the minimum that passes the "
            f"BigCrush randomness tests; 10 is the standard."
        )
    return rounds


def philox4x32(c0, c1, c2, c3, k0, k1, *, rounds: int):
    """Philox4x32 on int64 tensors holding unsigned 32-bit values. Returns 4 words.
    `rounds` is used as given (the known-answer tests also call it outside 7-10)."""
    for _ in range(rounds):
        p0 = c0 * _MULT_A
        p1 = c2 * _MULT_B
        hi0, lo0 = (p0 >> 32) & _M32, p0 & _M32
        hi1, lo1 = (p1 >> 32) & _M32, p1 & _M32
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = (k0 + _WEYL_A) & _M32
        k1 = (k1 + _WEYL_B) & _M32
    return c0, c1, c2, c3


def open_uniform(word: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(0, 1]: never 0, so log() is finite."""
    return ((word >> 8) + 1).to(dtype) * _INV_2_24


def half_open_uniform(word: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[0, 1): the range `int(u * k)` needs, as numpy's random() gives."""
    return (word >> 8).to(dtype) * _INV_2_24


def standard_normals(seed: torch.Tensor, tick: torch.Tensor, K: int, n: int,
                     dtype: torch.dtype, *, rounds: int) -> torch.Tensor:
    """(K, B, 1, n) standard normals for B runs at their own world ticks.

    seed, tick: (B, 1) int64 -- each run's noise seed and current world tick.
    Brain tick k of the result is the address k, so the numbers do not depend on K.
    """
    B = seed.shape[0]
    dev = seed.device
    nb = _BLOCKS_PER_BRAIN_TICK
    k = torch.arange(K, dtype=torch.int64, device=dev).view(K, 1, 1)
    blk = torch.arange(nb, dtype=torch.int64, device=dev).view(1, 1, nb)
    shape = (K, B, nb)
    c0 = tick.view(1, B, 1).expand(shape)
    c1 = (k * nb + blk).expand(shape)
    zero = torch.zeros(shape, dtype=torch.int64, device=dev)
    c3 = torch.full(shape, STREAM_NOISE, dtype=torch.int64, device=dev)
    k0 = seed.view(1, B, 1).expand(shape)
    words = torch.stack(philox4x32(c0, c1, zero, c3, k0, zero, rounds=check_rounds(rounds)),
                        dim=-1)                                            # (K, B, nb, 4)
    u = open_uniform(words.reshape(K, B, 2 * nb, 2), dtype)               # 6 pairs
    r = torch.sqrt(-2.0 * torch.log(u[..., 0]))
    theta = _TWO_PI * u[..., 1]
    z = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=-1)  # (K, B, 6, 2)
    return z.reshape(K, B, 4 * nb)[..., :n].unsqueeze(2)                  # (K, B, 1, n)


def decision_uniforms(seed: torch.Tensor, tick: torch.Tensor,
                      dtype: torch.dtype, *, rounds: int) -> torch.Tensor:
    """(B, 1) uniforms in [0, 1), one per run at its own world tick."""
    zero = torch.zeros_like(seed)
    c3 = torch.full_like(seed, STREAM_DECISION)
    w0, _, _, _ = philox4x32(tick, zero, zero, c3, seed, zero, rounds=check_rounds(rounds))
    return half_open_uniform(w0, dtype)


# --- compiled variants (plan Step 8, R5) -----------------------------------------
# Philox is ~120 small elementwise operations; fused by torch.compile it measured 18-20x
# faster on MPS and CUDA and bit-identical to eager. Compiled with PyTorch's default
# "automatic dynamic" shapes: the first call compiles with fixed sizes; when the batch
# width changes (compaction), ONE recompile makes that size variable, after which no
# more happen. dynamic=True cannot be used: it makes every integer variable, and
# `rounds` must stay a fixed loop count (it does: it never changes).
_COMPILED = {}


def _specialised(call: str, dtype: torch.dtype):
    """A function `f(seed, tick)` whose integer constants are LITERALS in its own code.

    torch.compile tracks Python ints per code object: if one shared function is called
    with rounds=7 and later rounds=10, it decides `rounds` varies and makes it symbolic
    -- and a symbolic loop count cannot be traced. Writing the constants into a separate
    code object per combination keeps them constants. `call` holds only literals.
    """
    namespace = {"standard_normals": standard_normals,
                 "decision_uniforms": decision_uniforms, "DTYPE": dtype}
    exec(f"def f(seed, tick):\n    return {call}\n", namespace)   # noqa: S102 -- literals only
    return namespace["f"]


def compiled_normals(K: int, n: int, dtype: torch.dtype, rounds: int):
    """`standard_normals(seed, tick)` for fixed K, n, dtype, rounds -- compiled once."""
    key = ("normals", int(K), int(n), dtype, check_rounds(rounds))
    if key not in _COMPILED:
        f = _specialised(f"standard_normals(seed, tick, {int(K)}, {int(n)}, DTYPE, "
                         f"rounds={int(rounds)})", dtype)
        _COMPILED[key] = torch.compile(f)
    return _COMPILED[key]


def compiled_uniforms(dtype: torch.dtype, rounds: int):
    """`decision_uniforms(seed, tick)` for fixed dtype, rounds -- compiled once."""
    key = ("uniforms", dtype, check_rounds(rounds))
    if key not in _COMPILED:
        f = _specialised(f"decision_uniforms(seed, tick, DTYPE, rounds={int(rounds)})",
                         dtype)
        _COMPILED[key] = torch.compile(f)
    return _COMPILED[key]
