import numpy as np
from .world import World


# --- individual behaviors ---

def setup_food_initially(world: World, cfg: dict):
    """Fill the world with initial food based on fraction.
    
    Args:
        world: World instance
        cfg: Feed config dict with keys: initial_fraction_per_cell, feeding_paradigm, regrow_time
    """
    p = cfg["initial_fraction_per_cell"]
    world.food = (world.rng_world_run.random(world.food.shape) < p).astype(np.int8)
    world.regrow_timer.fill(0)


def tick_regrow(world: World, cfg: dict):
    """Handle regrowth of eaten food if enabled.
    
    Args:
        world: World instance
        cfg: Feed config dict
    """
    # decrement timers
    world.regrow_timer[world.regrow_timer > 0] -= 1
    # regrow when timer hits 0 (but was active before)
    regrown = (world.regrow_timer == 1)
    world.food[regrown] = 1


def on_eat(world: World, cfg: dict, y: int, x: int) -> bool:
    """Called when Byte eats. Removes food and maybe starts regrow timer.
    
    Args:
        world: World instance
        cfg: Feed config dict with keys: feeding_paradigm, regrow_time
        y: Y coordinate
        x: X coordinate
    
    Returns:
        True if food was eaten, False otherwise
    """
    if world.food[y, x] > 0:
        world.food[y, x] -= 1
        if cfg["feeding_paradigm"].get("regrow", False):
            world.regrow_timer[y, x] = cfg["regrow_time"]
        return True
    return False


# --- high-level API used by sim/world ---

def seed_food(world: World, cfg: dict):
    """Seed food according to config (only if 'initial' enabled).
    
    Args:
        world: World instance
        cfg: Feed config dict with keys: feeding_paradigm, initial_fraction_per_cell
    """
    if cfg["feeding_paradigm"].get("initial", False):
        setup_food_initially(world, cfg)


def feeding_tick(world: World, cfg: dict):
    """Per-tick update. Handles phase transitions and regrowth.

    If a phase transition is due this tick (world.ticks == next phase's phase_from),
    re-seeds the food grid with the new phase config and updates world.feeding_cfg.
    Otherwise runs the normal regrow logic if enabled.

    Args:
        world: World instance
        cfg: Feed config dict for the current phase
    """
    if world.switch_phases and world.ticks == world.switch_phases[0]["phase_from"]:
        new_cfg = world.switch_phases.pop(0)
        seed_food(world, new_cfg)
        world.feeding_cfg = new_cfg
        return

    if cfg["feeding_paradigm"].get("regrow", False):
        tick_regrow(world, cfg)
