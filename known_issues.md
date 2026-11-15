# Known issues

## 1. Per-tick tracking fails for large brains (HDF5 header limit)

**Found:** 2026-11-14, on the university PC (RTX 3070), during the behavioural-equivalence
test at n = 50 (plan_evotorch.md, Step 12, part b).

**Symptom.** Writing per-tick tracking data stops with
`ValueError: Unable to synchronously create dataset (object header message is too large)`.

**Cause.** `simulate/hdf5_utils.save_per_tick_to_hdf5` stores each run's per-tick record as
one HDF5 compound dataset with one float32 column per tracked connection, next to ten
base fields (tick, movement, energy, ...). HDF5 keeps a compound type's description in
the dataset's object header, which is limited to 64 KiB.

- Measured on the Mac: 1,000 columns still fit; 1,500 do not.
- Genomes at n = 11 track at most 44 connections.
- Genomes at n = 50 (random generation-0 genomes and evolved elites of the large run)
  track 1,475-1,532.

**Who is affected.** The scalar code, the tensor code and the original code
(8e17e694c7) all write per-tick data with this same function, so the limit is in the
file format, not in either simulation.

| Use | Affected? |
|---|---|
| The EA runs (`run_ea`): all thesis runtime and production runs, including the A100 large run | **No.** `run_ea` never writes per-tick data. |
| Per-run tracking (lifespan, eats, distance, final weights per run) | **No.** One column per run, not per connection. |
| `run_batch` with `enable_per_tick_tracking: true` for genomes with more than about 1,000 tracked connections (in practice n ≳ 35 with evolved genomes) | **Yes**, with both the scalar and the tensor backend. |
| The behavioural-equivalence test at n = 50 | Was affected; worked around (below). |

**Workaround in the equivalence test.** The test replaces the save function in memory,
inside its own processes only, with one that writes the same numbers in a split layout:

- `per_tick` holds the base fields;
- `per_tick_weights` holds a ticks × connections float32 matrix;
- `per_tick_weight_names` holds the column names.

The simulations compute exactly what they computed before; only the write to disk
differs. The workaround is verified by these checks:

- every per-tick run is checked to have been written by the replacement (an error
  otherwise);
- reading a file back gives the original record exactly;
- at n = 50, the scalar and tensor per-tick records (1,510 columns) are identical;
- the original code's files stay unmodified on disk.

This was the only option for the original code, whose files cannot be changed.

**Not fixed in production.** Fixing it means changing the per-tick file format. Every
reader would then have to accept both the old and the new layout: the writer in both
backends, the replay tools and the analysis scripts. It is not needed for the thesis
runs or for the equivalence test. It needs its own plan if per-tick tracking of large
brains is wanted.
