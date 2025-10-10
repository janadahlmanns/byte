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
        # food grid: 0/1 per cell for v1
        self.food = np.zeros((self.height, self.width), dtype=np.int8)

    def reset_food(self):
        self.food.fill(0)

    def in_bounds(self, y: int, x: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def has_food(self, y: int, x: int) -> bool:
        return self.food[y, x] > 0

    def eat_one(self, y: int, x: int) -> bool:
        if self.food[y, x] > 0:
            self.food[y, x] -= 1
            return True
        return False

    def valid_moves_from(self, y: int, x: int):
        # edges are hard; only return in-bounds moves
        moves = []
        if self.in_bounds(y - 1, x): moves.append(("up", (y - 1, x)))
        if self.in_bounds(y + 1, x): moves.append(("down", (y + 1, x)))
        if self.in_bounds(y, x - 1): moves.append(("left", (y, x - 1)))
        if self.in_bounds(y, x + 1): moves.append(("right", (y, x + 1)))
        return moves

#test github connection