import numpy as np


class World:
    def __init__(self, grid_width, grid_height, start_pos, rng_seed):
        self.width = grid_width
        self.height = grid_height
        self.start_pos = start_pos
        self.rng_seed = rng_seed
        self.ticks = 0
        self.food = np.zeros((self.height, self.width), dtype=np.int8) # food grid: 0/1 per cell for v1
        self.regrow_timer = np.zeros_like(self.food, dtype=np.int16) # regrowth timer map (same shape as food grid)
        self.switch_phases = []      # phase transition schedule; set to a fresh list copy per run by eval_variant
        self.rng_world_run = None    # rng_world_run for this run; set per run by eval_variant for phase re-seeding

    def reset_food(self):
        self.food.fill(0)
        self.regrow_timer.fill(0)
        self.ticks = 0

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