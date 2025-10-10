import numpy as np
from .world import World

def random_valid_move(world: World, pos: tuple[int, int], rng: np.random.Generator):
    """Pick a random valid neighboring cell."""
    y, x = pos
    moves = world.valid_moves_from(y, x)
    if not moves:
        return pos  # no valid move (edge case)
    choice = rng.integers(0, len(moves))
    return moves[choice][1]

def decide(world: World, worm, rng: np.random.Generator, inputs: dict):
    """
    Decide what the worm should do next.
    Returns a tuple like ("eat",) or ("move", (new_y, new_x))
    """
    # Step 1: eat if on food always
    on_food = inputs.get("on_food", False)

    # Step 2: if on food, eat; else move
    if on_food:
        return ("eat",)
    else:
        new_pos = random_valid_move(world, (worm.y, worm.x), rng)
        return ("move", new_pos)
