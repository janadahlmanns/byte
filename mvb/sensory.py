from .world import World

# --- individual sensory modules ---

def perceive_current_field(world: World, worm):
    """Checks for food under the worm."""
    return {
        "on_food": int(world.has_food(worm.y, worm.x))
    }


def perceive_adjacent_binary(world: World, worm):
    """Binary food presence on the four adjacent tiles (N/E/S/W)."""
    y, x = worm.y, worm.x
    h, w = world.height, world.width

    north = ((y - 1) % h, x) # modulo for wrapping when Byte is on the edge
    south = ((y + 1) % h, x)
    west  = (y, (x - 1) % w)
    east  = (y, (x + 1) % w)

    return {
        "food_north": int(world.has_food(*north)),
        "food_south": int(world.has_food(*south)),
        "food_west":  int(world.has_food(*west)),
        "food_east":  int(world.has_food(*east)),
    }


# --- registry of all known senses ---

SENSORY_REGISTRY = {
    "current_field": perceive_current_field,
    "adjacent_binary": perceive_adjacent_binary,
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
