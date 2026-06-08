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
