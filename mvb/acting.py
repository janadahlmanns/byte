from .world import World

def act(world, worm, action):
    """Execute the given action tuple ('move', (y,x)) or ('eat',)."""
    verb = action[0]
    fn = ACTION_REGISTRY.get(verb)
    if fn is None:
        raise ValueError(f"Unknown action type: {verb}")
    fn(world, worm, *action[1:])

def do_move(world, worm, pos):
    ny, nx = pos
    worm.y, worm.x = ny, nx
    worm.energy = max(0, worm.energy - 1)  # movement cost

def do_eat(world, worm):
    if world.eat_one(worm.y, worm.x):
        worm.energy = worm.cfg.energy_capacity
        worm.eats += 1

ACTION_REGISTRY = {
    "move": do_move,
    "eat": do_eat,
    # later: "flee": do_flee, "mate": do_mate, etc.
}
