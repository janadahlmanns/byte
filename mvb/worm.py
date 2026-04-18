from dataclasses import dataclass
from .world import World
from .acting import act
from .sensory import perceive

@dataclass
class WormConfig:
    speed: int          # cells per tick (must be 1 in v1)
    energy_capacity: int
    metabolic_rate: int # energy per tick
    movement_cost: int  # energy cost per move

class Worm:
    def __init__(self, speed: int, energy_capacity: int, metabolic_rate: int, movement_cost: int, world: World):
        self.speed = speed
        self.energy_capacity = energy_capacity
        self.metabolic_rate = metabolic_rate
        self.movement_cost = movement_cost
        self.world = world
        self.renderer = None
        self.reset()

    def reset(self):
        # YAML provides start_pos as [x, y]; convert to (y, x)
        sx_yaml, sy_yaml = self.world.start_pos
        self.y, self.x = sy_yaml, sx_yaml

        self.energy = self.energy_capacity
        self.alive = True
        self.eats = 0
        self.distance = 0
        self.ticks = 0
        self.action = None
        self.sensory_information = {}


    def death_gate(self) -> bool:
        """If energy is 0 at START of tick, die immediately."""
        if self.energy <= 0:
            self.alive = False
            return True
        return False
    

    def step_day(self, rng_decision):
        """
        Execute exactly one biological day.
        Brain thinking may internally span multiple beats.
        This function does NOT advance simulation time.
        
        Order:
        1. Act (using action decided last day)
        2. Sense (from NEW situation)
        3. Draw world (showing current position and sensory state)
        4. Decide (store for NEXT day)
        5. Metabolize
        """

        if not self.alive:
            return

        # 1) Act (using action decided last day)
        if self.action is not None:
            act(self.world, self, self.action)

        # 2) Sense (from NEW situation)
        self.sensory_information = perceive(self.world, self)

        # 3) Draw world (after position and sensory update)
        if self.renderer is not None:
            self.renderer.draw()

        # 4) Decide (store for NEXT day)
        #     NOTE: decision-making may block internally (brain beats)
        self.action = self.brain.decide(
            self.world,
            self,
            rng_decision,
            self.sensory_information,
        )

        # 5) Metabolism
        self.energy = max(0, self.energy - self.metabolic_rate)

        # 6) Death gate
        if self.energy <= 0:
            self.alive = False
            return