from .world import World

def act(world: World, worm, action: tuple):
    kind = action[0]

    if kind == "eat":
        if world.eat_one(worm.y, worm.x):
            worm.eats += 1
            # Refill or add energy gain
            worm.energy = min(
                worm.cfg.energy_capacity,
                worm.energy + worm.cfg.energy_capacity
            )

    elif kind == "move":
        new_y, new_x = action[1]
        # Extra movement cost
        move_cost = 1
        worm.energy = max(0, worm.energy - move_cost)

        if (new_y, new_x) != (worm.y, worm.x):
            worm.distance += 1
        worm.y, worm.x = new_y, new_x

    else:
        raise ValueError(f"Unknown action type: {kind}")
