from .world import World

# --- individual sensory modules ---
def perceive_current_field(world: World, worm):
    """Checks for food under the worm."""
    return {"on_food": world.has_food(worm.y, worm.x)}

def perceive_chemotaxis_field(world: World, worm):
    """Placeholder for future gradient sensing."""
    return {"chemotaxis_signal": 0.0}

# --- registry of all known senses ---
SENSORY_REGISTRY = {
    "current_field": perceive_current_field,
    "chemotaxis_field": perceive_chemotaxis_field,
}

def perceive(world: World, worm):
    """
    Aggregate sensory data from all active sensors defined for this worm.
    """
    sensors_out = {}
    for name in getattr(worm, "active_sensors", ["current_field"]):
        fn = SENSORY_REGISTRY.get(name)
        if fn is None:
            print(f"Warning: unknown sensor '{name}' — skipping.")
            continue
        sensors_out.update(fn(world, worm))
    return sensors_out
