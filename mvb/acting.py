from .world import World
from .feeding import on_eat

def act(world, worm, action):
    """Execute the given action tuple ('move', (y,x)) or ('eat',)."""
    verb = action[0]
    fn = ACTION_REGISTRY.get(verb)
    if fn is None:
        raise ValueError(f"Unknown action type: {verb}")
    fn(world, worm, *action[1:])  # consistent call

def do_move(world, worm, pos, *args):   # <— fixed: world first!
    ny, nx = pos
    worm.y, worm.x = ny, nx
    worm.energy = max(0, worm.energy - 1)  # movement cost
    worm.distance += 1

def do_eat(world, worm, *args):
    if on_eat(world, world.feeding_cfg, worm.y, worm.x):
        worm.energy = worm.cfg.energy_capacity
        worm.eats += 1

ACTION_REGISTRY = {
    "move": do_move,
    "eat": do_eat,
    # later: "flee": do_flee, "mate": do_mate, etc.
}
