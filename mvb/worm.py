from dataclasses import dataclass
from .world import World
from .brains.decisionmaking_prio_food import decide
from .acting import act
from .sensory import perceive

@dataclass
class WormConfig:
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
        """If energy is 0 at START of tick, die immediately."""
        if self.energy <= 0:
            self.alive = False
            return True
        return False

    def step(self, rng):
        if not self.alive:
            return

        # 1) Baseline metabolism
        self.energy = max(0, self.energy - self.cfg.metabolic_rate)

        # 2) Death gate
        if self.energy <= 0:
            self.alive = False
            return

        # 3) Sense
        sensory_information = perceive(self.world, self)

        # 4) Decide
        action = self.brain.decide(self.world, self, rng, sensory_information)

        # 5) Act
        act(self.world, self, action)

        # 6) Advance time
        self.ticks += 1
