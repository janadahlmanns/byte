# MVB 

A simulation for exploring the concept of a **Minimum Viable Brain (MVB)**.
Tiny CTRNNs are used to control embedded agents to solve a grid-foraging task. Optimization of parameters is achieved through evolutionary algorithms. The project aims to develop adaptable agents able to solve tasks outside their explicit training scope. 

## Quickstart
```bash
python -m simulate.run_batch --config test_batch      
```

## Replay viewer
Turn a saved run into a standalone `.html` and scrub through it in any browser:
```bash
python -m viz.export_viewer --file <h5-name> --genome 0 --run 1
```
`--list` shows what is in a file; omit `--genome`/`--run` to choose interactively.

Experiments stay headless and parallel - the run is replayed afterwards from its stored seeds. The page inlines its own CSS, JS and data, so it needs no server and no network. It is a snapshot: re-export after editing anything in `viz/template/`.

Separate from `viz_enabled` / `viz_brain_enabled` in the experiment YAMLs, which open live Qt windows during a run and force it serial.

## Overview Folders
1. mvb - do not touch :-) Simulation core
2. simulate - runner scripts used to access the simulations
3. viz - replay viewer: a saved run in, one self-contained .html out
4. configs - WORK HERE: configure YAMLs to run different simulations
5. data - goes into here duh
6. analysis_tools, scicom, experimental_design - should not be part of public repo and contain inner workings of the project
7. archive - well...you're smart enough to figure that one out

## Code principles
Reproducibility and full control via YAML
- hence clean rng scopes and usages
- hence loud fails and no silent fallbacks to defaults
