from .world import World
from .feeding import on_eat


def act(world, worm, action):
    """
    Execute the given action tuple.

    Valid actions:
    - ("move", (y, x))
    - ("stay",)
    - ("eat",)   # legacy / backward compatibility
    """
    if action is None:
        return  # explicit no-op

    verb = action[0]
    fn = ACTION_REGISTRY.get(verb)
    if fn is None:
        raise ValueError(f"Unknown action type: {verb}")

    fn(world, worm, *action[1:])


# ------------------------------------------------------------
# Action implementations
# ------------------------------------------------------------

def do_move(world: World, worm, pos, *args):
    """
    Move the worm to a new position.
    """
    ny, nx = pos
    worm.y, worm.x = ny, nx
    worm.energy = max(0, worm.energy - 1)  # movement cost
    worm.distance += 1


def do_stay(world: World, worm, *args):
    """
    Stay in place.
    If there is food, eat it.
    Otherwise do nothing.
    """
    if on_eat(world, world.feeding_cfg, worm.y, worm.x):
        worm.energy = worm.cfg.energy_capacity
        worm.eats += 1


def do_eat(world: World, worm, *args):
    """
    Legacy explicit eat action.
    """
    if on_eat(world, world.feeding_cfg, worm.y, worm.x):
        worm.energy = worm.cfg.energy_capacity
        worm.eats += 1


# ------------------------------------------------------------
# Action registry
# ------------------------------------------------------------

ACTION_REGISTRY = {
    "move": do_move,
    "stay": do_stay,
    "eat": do_eat,   # legacy
}
