# MVB Worm Byte

A minimal simulation for exploring the concept of a **Minimum Viable Brain (MVB)**.
Tiny CTRNNs are used to control embedded agents to solve a grid-foraging task. Optimization of parameters is achieved through evolutionary algorithms. The project aims to develop adaptable agents ablet o saolves tasks outside their explicit training scope. 

## Quickstart
```bash
python -m simulate.run_batch --config test_batch      
```

## Overview Folders
1. mvb - to not touch :) Simulation core
2. simulate - runner scripts used to access the simulations
3. configs - WORK HERE: configure Yamls to run different simulations
4. data - goes into here duh
5. analysis_tools, scicom, experimental_design - should not be part of public repo and contain inner workings of the project
6. archive - wel...you're smart enough to figure that one out

## Code principles
Reproducibility and full control via YAML
- hence clean rng scopes and usages
- hence loud fails and no silent fallbacks to defaults

## Running on an NVIDIA machine with Docker
The image holds the code and a pinned software stack: Ubuntu 24.04, Python 3.12 and
torch 2.13.0 with CUDA 13.0. A machine needs neither Python nor CUDA, only Docker, the
GPU driver and git. Design and checks: `plan_evotorch.md`, Step 9.

**Every machine builds its own image** from a clone of this repo. Images are not
copied between machines: the build is small to describe and pinned (base image by
digest, exact torch version), and the image name records the commit it was built from.

**Host requirements**
- An NVIDIA driver that supports CUDA 13.0, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Linux, once per machine:
  - Use the native Docker Engine, not Docker Desktop: Desktop's VM has no GPU.
  - Add yourself to the docker group: `sudo usermod -aG docker $USER`, then log in
    again. Otherwise you get "permission denied ... docker.sock".
  - Register the GPU with Docker:
    ```bash
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```
    Otherwise you get "failed to discover GPU vendor from CDI".
  - Check: `docker run --rm --gpus all ubuntu:24.04 nvidia-smi` must list the GPU.
- Windows: Docker Desktop with the WSL2 backend.
- Not for macOS: Docker there cannot use the GPU (MPS). Run natively instead.

**Build (on each machine)**
```bash
git clone https://github.com/janadahlmanns/byte.git && cd byte          # or `git pull` in an existing clone
git fetch && git checkout evotorch_impl                                 # checkout the pytorch branch
TAG=$(git rev-parse --short=10 HEAD)                                    # the image is named after its code
docker build --build-arg CODE_VERSION=$TAG -t byte-sim:$TAG .           # fails without CODE_VERSION
```
- The first build downloads the base image and torch (a few GB) and takes a while;
  later builds of new commits reuse those layers and only copy the code.
- Run all commands below from the clone's folder: results go to its `data/`.

**For testers: the default test run**

After building, start the container without a command:
```bash
mkdir -p data
docker run -it --rm --gpus all --ipc=host --user $(id -u):$(id -g) \
  -v "$PWD/data:/app/data" byte-sim:$TAG
```
- **A list of every config opens, with the default set already ticked**
  (`configs/presets/tester_default.yaml`). **Press enter.** It runs, in this order:
  1. `ea_from_lookup_tensor_cuda`, 50 generations: the reproducibility check. Its
     result is compared bit for bit with the university PC's.
  2. `neuron_scaling` at 11, 20, 30, 40 and 50 neurons, 10 generations each:
     runtime and memory against neuron count.

  About 1 h on an RTX 3070; slower cards take longer. Cheapest first, so if you have
  to stop (Ctrl-C), what has finished is still useful.
  
- **At the end:** a summary, followed by
  `Send this folder: data/runtime_reports/<time>_<gpu>/`. Please send that whole
  folder. It holds:
  - `system.json`: GPU, driver, CPU, RAM, versions, the code version, and whether the
    default set was run unchanged;
  - `summary.csv`, one row per config:
    - status, total time, generation 0 (includes compiling), the mean of the later
      generations, and microseconds per worm-tick (comparable between configs whose
      worms live for different lengths of time);
    - `matches_reference`: for the lookup run, `identical (...)` or what differs;
    - memory: `vram_peak_mib` (the most GPU memory torch held),
      `gpu_mem_used_peak_mib` (what nvidia-smi saw for the whole card), `runs_queued`
      (`yes` if not all runs fit at once) and `vram_needed_all_runs_est_mib` (then the
      estimated memory for all runs at once; an upper bound, ~3.5x the real need at
      n = 30-50);
    - GPU load before and after, to spot a disturbed run;
  - `generations.csv`: every generation's time and worm-ticks;
  - `results/`: a copy of each run's result file (`.h5`);
  - `configs/`: the expanded configs of sweep points;
  - one log per config.
- **Choosing other configs:**
  - Keys: up/down to move, space to tick or untick, `a` to tick all, enter to start,
    `q` to quit. Configs run in the order you tick them; the number in front shows it.
  - Greyed-out configs cannot run on this machine; the reason is shown next to them.
  - A config that fails is recorded, and the next one still runs.
  - Without a terminal: `... byte-sim:$TAG python -m simulate.run_selected --preset
    tester_default`, or `--configs a,b` (names as in `--list`).
- **Sweeps** (`configs/sweeps/`): one file varies one setting over a base config. Each
  value is its own row, e.g. `neuron_scaling [n_neurons=20]`. Every key in a sweep
  must already exist in the base config, so a typo fails instead of being ignored.
- **Neuron-scaling figure** (natively, outside the container): from one or more
  report folders,
  ```bash
  python -m analysis.plot_neuron_scaling data/runtime_reports/<folder> [...] --out analysis/out
  ```
  writes `neuron_scaling.png`/`.pdf`, `points.csv` and `fits.csv` (the fitted growth
  exponent, t ∝ n^k). Runs from before the Step 18 compile fix or on a busy GPU are
  left out, each with the reason printed. The finer sweep
  `neuron_scaling_fine` (n = 11-50 in steps of 3, ~1.5 h on an RTX 3070) gives a
  denser curve.
- **VRAM vs runtime, emulated:** `python -m tests.width_probe --vram-gib --csv vram.csv`
  runs each n at the width our rule would choose on smaller cards (2, 3, 4, 6, 8, ...
  GiB up to this card's memory), on the largest card available. Pass the CSV to the
  figure with `--vram-csv vram.csv`.
- **Presets** (`configs/presets/`): a named list of configs in run order, plus
  optional reference result files. A preset naming a config that does not exist stops
  the selector before anything runs.

**Run**

The image contains the code and every config in `configs/experiments/` as they were at
build time. Runs read and write only the mounted folders, from the clone's folder:
```
byte/              the clone
├── data/          results land here (can start empty)
└── my_configs/    optional: your own YAML files
```
Pick one of three ways:

1. **A config that is in the image.** No config files are needed on the machine.
   ```bash
   docker run -it --rm --gpus all --ipc=host --user $(id -u):$(id -g) \
     -v "$PWD/data:/app/data" \
     byte-sim:$TAG python -m simulate.run_ea --config ea_from_lookup_tensor_cuda
   ```
2. **Your own config, with the built-in ones still available.** Mount your folder
   somewhere else and pass the file's path:
   ```bash
   docker run -it --rm --gpus all --ipc=host --user $(id -u):$(id -g) \
     -v "$PWD/data:/app/data" -v "$PWD/my_configs:/my_configs" \
     byte-sim:$TAG python -m simulate.run_ea --config /my_configs/my_experiment.yaml
   ```
3. **A whole configs folder of your own** (e.g. a clone of this repo). Mounting it at
   `/app/configs` *replaces* the image's configs: only the files in your
   `configs/experiments/` are visible, and they are run by name.
   ```bash
   docker run -it --rm --gpus all --ipc=host --user $(id -u):$(id -g) \
     -v "$PWD/data:/app/data" -v "$PWD/configs:/app/configs" \
     byte-sim:$TAG python -m simulate.run_ea --config my_experiment
   ```

Config changes in a mounted folder never need a rebuild. Code changes do: `git pull`,
then build again on that machine (the tag follows the commit).
- **`-it`:** the runners ask y/n questions (e.g. "Display plot?" at the end), so they
  need a terminal. To leave a run in the background, start it inside `tmux`/`screen`.
- **`--user`:** results in `data/` belong to you, not root.
- **`--ipc=host`:** recommended for the scalar runner's worker processes. The scalar
  test EA also ran with Docker's default 64 MB of shared memory, but production size is
  untested without the flag.
- **`-v "$PWD/data:/app/data"`:** results land in the host's `data/`. Without it they
  are written inside the container and lost when it ends.
- **No GPU visible?** Without `--gpus all`, a CUDA config stops with "CUDA is not
  available".

**Input files** (e.g. `genome_from_file`)
- A `file_folder` under `data/` works as it is.
- For files stored elsewhere, mount their folder and use the absolute path in the
  config. Mount it read-write: replay and benchmark modes write their result next to
  the source file.
  ```bash
  docker run -it --rm --gpus all --ipc=host --user $(id -u):$(id -g) \
    -v "$PWD/data:/app/data" -v "$PWD/my_configs:/my_configs" \
    -v /path/to/inputs:/inputs \
    byte-sim:$TAG python -m simulate.run_batch --config /my_configs/my_replay.yaml
  # my_replay.yaml: a copy of genome_from_file.yaml with  file_folder: "/inputs/"
  # and visualisation off (the built-in genome_from_file.yaml has it on)
  ```
- Run `git lfs pull` on the host first. Otherwise the container sees LFS pointer files
  ("file signature not found").
- Visualisation is not available in the container (no screen, no Qt). Set
  `viz_enabled: false` and `viz_brain_enabled: false`.

**Tests and GPU diagnostics in the container**
```bash
docker run -it --rm --gpus all --ipc=host --user $(id -u):$(id -g) byte-sim:$TAG python -m tests.run_all
docker run -it --rm --gpus all byte-sim:$TAG python -m tests.gpu_profile
```

**Optional: keep compiled kernels between runs.** Each new container recompiles them,
which cost about 19 s in generation 0 on a GTX 1660 Ti. To avoid that, add
`-v "$PWD/.byte-cache:/cache"`.
