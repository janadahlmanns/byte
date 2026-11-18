# Known issues

## 1. Per-tick tracking fails for large brains (HDF5 header limit) — FIXED 2026-11-18 (see "Fix" at the end of this section)

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

**Fix (2026-11-18, plan_evotorch.md Step 33).**

*Writing.* Records that fit are written exactly as before: one compound dataset, byte
for byte the same file format. Only records above the limit are split:

| Record | Split when | On disk |
|---|---|---|
| `variant_V/run_R/per_tick` | more than 1,000 connection columns (n = 50 brains track ~1,500) | `per_tick_base` (the 10 base fields), `per_tick_weights` (ticks × connections, float32), `per_tick_weight_names`, and the attribute `per_tick_layout = "split"` on the run group |
| `variant_V/wiring` | more than 800 runs (it has one column per run) | `wiring_base`, `wiring_weights` (connections × runs), `wiring_weight_names`, and the attribute `wiring_layout = "split"` |

- **The `wiring` row is the same limit, found while fixing this:** `wiring` fails above
  854 runs at any brain size. No run so far has used more than 300 runs.
- **The scalar code and the tensor code** both write through `simulate/hdf5_utils.py`,
  so both are fixed, and their files stay identical.
- **The first split write** of a run prints one line saying so.

*Reading.* Files from earlier simulations (all n = 11: at most 44 connection columns)
and every new file below the limits read exactly as before with plain h5py. To read
files of any size with the same code, change the line that opens the file:

```python
from simulate.hdf5_utils import open_tracking
with open_tracking("path/to/file.h5") as f:          # instead of h5py.File(path, "r")
    pt = f["variant_0/run_3/per_tick"][()]           # unchanged: the full record, any n
```

- `open_tracking` returns the same structured array the simulation built: every base
  field and every connection column, in the original order.
- It also supports slices, single fields and nested access
  (`f["variant_0"]["run_3"]["per_tick"]`); everything else in the file reads as with
  h5py.
- **Plain h5py on a split record fails** with "object 'per_tick' doesn't exist" (a
  KeyError). It never returns the base fields without the weights.

Tests: `tests/test_hdf5_layout.py`. It covers round trips at 10 / 1,000 / 1,001 / 2,500
columns and 300 / 800 / 801 / 2,000 runs, and checks that `open_tracking` reads the
n = 11 reference files exactly as h5py does. It also runs a real n = 50 run with all
tracking, scalar and tensor, and includes negative controls.
