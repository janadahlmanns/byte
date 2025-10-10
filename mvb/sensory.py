from .world import World

def perceive(world: World, worm):
    """
    Collect sensory information for the worm.
    Returns a dictionary with all sensed signals.
    """
    sensors = {}

    # Basic sense: whether there is food on the current cell
    sensors["on_food"] = world.has_food(worm.y, worm.x)

    # Future senses could include:
    # sensors["nearest_food"] = ...
    # sensors["enemy_in_range"] = ...
    # sensors["energy_fraction"] = worm.energy / worm.cfg.energy_capacity

    return sensors
