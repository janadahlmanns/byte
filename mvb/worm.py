from dataclasses import dataclass
from .world import World
from .policies import choose_next_position

@dataclass
class WormConfig:
    worm_version: str   # "random"
    speed: int          # cells per tick (must be 1 in v1)
    energy_capacity: int
    metabolic_rate: int # energy per tick

class Worm:
    def __init__(self, cfg: WormConfig, world: World):
        self.cfg = cfg
        self.world = world
        self.reset()

    def reset(self):
        # YAML provides start_pos as [x, y]; convert to (y, x)
        sx_yaml, sy_yaml = self.world.cfg.start_pos
        self.y, self.x = sy_yaml, sx_yaml

        self.energy = self.cfg.energy_capacity
        self.alive = True
        self.eats = 0
        self.distance = 0
        self.ticks = 0


    def death_gate(self) -> bool:
        # If energy is 0 at START of tick, die immediately
        if self.energy <= 0:
            self.alive = False
            return True
        return False

    def step(self, rng, policy_name: str):
        if not self.alive:
            return

        # 1) Death gate at start of tick
        if self.death_gate():
            return

        # 2) Decide action (forced eat if on food)
        on_food = self.world.has_food(self.y, self.x)

        # 3) Drain first
        self.energy = max(0, self.energy - self.cfg.metabolic_rate)

        # 4) Apply action
        if on_food:
            # Eat consumes tick, then refills to cap
            if self.world.eat_one(self.y, self.x):
                self.eats += 1
            self.energy = self.cfg.energy_capacity
        else:
            # Move (speed=1)
            ny, nx = choose_next_position(policy_name, self.world, (self.y, self.x), rng)
            # distance = Manhattan of one step (always 1 here)
            if (ny, nx) != (self.y, self.x):
                self.distance += 1
            self.y, self.x = ny, nx

        self.ticks += 1
