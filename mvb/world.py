from dataclasses import dataclass
import numpy as np


@dataclass
class WorldConfig:
    grid_width: int
    grid_height: int
    start_pos: tuple[int, int]
    rng_seed: int

class World:
    def __init__(self, cfg: WorldConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.width = cfg.grid_width
        self.height = cfg.grid_height
        self.ticks = 0
        self.food = np.zeros((self.height, self.width), dtype=np.int8) # food grid: 0/1 per cell for v1
        self.regrow_timer = np.zeros_like(self.food, dtype=np.int16) # regrowth timer map (same shape as food grid)

    def reset_food(self):
        self.food.fill(0)

    def has_food(self, y: int, x: int) -> bool:
        return self.food[y, x] > 0

    def valid_moves_from(self, y: int, x: int):
        """
        Return all four moves with toroidal (wrap-around) topology.
        """
        h, w = self.height, self.width

        return [
            ("up",    ((y - 1) % h, x)),
            ("down",  ((y + 1) % h, x)),
            ("left",  (y, (x - 1) % w)),
            ("right", (y, (x + 1) % w)),
        ]


    def step(self):
        self.ticks += 1
        from .feeding import feeding_tick
        feeding_tick(self, self.feeding_cfg)