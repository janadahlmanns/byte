# MVB Worm Byte

A minimal simulation for exploring the concept of a **Minimum Viable Brain (MVB)**.
The worm Byte lives on a grid, moves randomly, eats food if it finds any, and dies if energy reaches zero. It will develop to have more senses, abilities and decision making skills as its world becomes more and more complex.

- Configurable via YAML (`configs/default.yaml`)
- Visualization via matplotlib
- Modular design for swapping evolutionary steps of Byte and its world

## Quickstart
```bash
python -m mvb.sim --config configs/default.yaml
