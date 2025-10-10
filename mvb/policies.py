import numpy as np
from .world import World

def random_valid_policy(world: World, pos: tuple[int,int], rng: np.random.Generator):
    y, x = pos
    moves = world.valid_moves_from(y, x)
    if not moves:
        return pos  # should never happen in rectangular world
    choice = rng.integers(0, len(moves))
    return moves[choice][1]

POLICY_REGISTRY = {
    "random": random_valid_policy,
}

def choose_next_position(policy_name: str, world: World, pos: tuple[int,int], rng: np.random.Generator):
    fn = POLICY_REGISTRY.get(policy_name)
    if fn is None:
        raise ValueError(f"Unknown worm_version/policy: {policy_name}")
    return fn(world, pos, rng)
