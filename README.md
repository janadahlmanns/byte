# MVB Worm

A minimal simulation for exploring the concept of a **Minimum Viable Brain (MVB)**.
The worm lives on a grid, moves randomly, eats food if it finds any, and dies if energy reaches zero.

- Configurable via YAML (`configs/default.yaml`)
- Visualization via matplotlib
- Modular design for swapping policies and feeding paradigms

## Quickstart
```bash
python -m mvb.sim --config configs/default.yaml
