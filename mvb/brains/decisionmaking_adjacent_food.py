import numpy as np
from ..world import World


def random_valid_move(world: World, pos: tuple[int, int], rng: np.random.Generator):
    """Pick a random valid neighboring cell."""
    y, x = pos
    moves = world.valid_moves_from(y, x)
    if not moves:
        return None
    choice = rng.integers(0, len(moves))
    return moves[choice]


def decide(world: World, worm, rng: np.random.Generator, inputs: dict):
    """
    Decision policy:
    1) Eat if on food
    2) Move toward adjacent food if visible
    3) Otherwise roam randomly
    """

    # --- 1) Eat if standing on food ---
    if inputs.get("on_food", 0):
        return ("eat",)

    # --- 2) Check adjacent food ---
    directions = {
        "up": inputs.get("food_north", 0),
        "down": inputs.get("food_south", 0),
        "left": inputs.get("food_west", 0),
        "right": inputs.get("food_east", 0),
    }

    food_dirs = [d for d, has_food in directions.items() if has_food]

    if food_dirs:
        # choose one food direction (random if multiple)
        direction = food_dirs[rng.integers(0, len(food_dirs))]
        y, x = worm.y, worm.x

        move_map = {
            "up":    (y - 1, x),
            "down":  (y + 1, x),
            "left":  (y, x - 1),
            "right": (y, x + 1),
        }

        ny, nx = move_map[direction]
        ny %= world.height
        nx %= world.width

        return ("move", (ny, nx))

    # --- 3) Otherwise: roam ---
    move = random_valid_move(world, (worm.y, worm.x), rng)
    if move is None:
        return ("stay",)
    return ("move", move[1])
