from dataclasses import dataclass
import numpy as np
from .world import World

@dataclass
class FeedingConfig:
    feeding_paradigm: str          # "only_initial"
    initial_fraction_per_cell: float  # e.g., 0.1

def setup_food_only_initial(world: World, cfg: FeedingConfig, rng: np.random.Generator):
    p = float(cfg.initial_fraction_per_cell)
    # Bernoulli per cell: 0/1 food
    world.food = (rng.random(world.food.shape) < p).astype(world.food.dtype)

# registry (simple for now)
FEEDING_REGISTRY = {
    "only_initial": setup_food_only_initial,
}

def apply_feeding(world: World, cfg: FeedingConfig, rng: np.random.Generator):
    fn = FEEDING_REGISTRY.get(cfg.feeding_paradigm)
    if fn is None:
        raise ValueError(f"Unknown feeding_paradigm: {cfg.feeding_paradigm}")
    fn(world, cfg, rng)
