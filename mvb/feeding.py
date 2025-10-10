from dataclasses import dataclass
import numpy as np
from .world import World

@dataclass
class FeedingConfig:
    feeding_paradigm: dict           # {"initial": True, "regrow": False, ...}
    initial_fraction_per_cell: float
    regrow_time: int


# --- individual behaviors ---

def setup_food_initially(world: World, cfg: FeedingConfig, rng: np.random.Generator):
    """Fill the world with initial food based on fraction."""
    p = cfg.initial_fraction_per_cell
    world.food = (rng.random(world.food.shape) < p).astype(np.int8)
    world.regrow_timer.fill(0)


def tick_regrow(world: World, cfg: FeedingConfig):
    """Handle regrowth of eaten food if enabled."""
    # decrement timers
    world.regrow_timer[world.regrow_timer > 0] -= 1
    # regrow when timer hits 0 (but was active before)
    regrown = (world.regrow_timer == 1)
    world.food[regrown] = 1


def on_eat(world: World, cfg: FeedingConfig, y: int, x: int) -> bool:
    """Called when Byte eats. Removes food and maybe starts regrow timer."""
    if world.food[y, x] > 0:
        world.food[y, x] -= 1
        if cfg.feeding_paradigm.get("regrow", False):
            world.regrow_timer[y, x] = cfg.regrow_time
        return True
    return False


# --- high-level API used by sim/world ---

def seed_food(world: World, cfg: FeedingConfig, rng: np.random.Generator):
    """Seed food according to config (only if 'initial' enabled)."""
    if cfg.feeding_paradigm.get("initial", False):
        setup_food_initially(world, cfg, rng)


def feeding_tick(world: World, cfg: FeedingConfig):
    """Per-tick update. Only regrows if 'regrow' enabled."""
    if cfg.feeding_paradigm.get("regrow", False):
        tick_regrow(world, cfg)
