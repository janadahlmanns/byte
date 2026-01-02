# RNG Usage Analysis

## Current Architecture
Currently there is **ONE single RNG stream** that draws from a single seed across the entire simulation. All stochasticity comes from this single generator, making it impossible to separately control different aspects of randomness.

## All RNG Usage Points

### 1. **World Initialization** - `mvb/world.py`
- **Location**: `World.__init__()`
- **Purpose**: Stores RNG as instance variable
- **Usage**: Passed to all modules needing randomness

### 2. **Food Initialization** - `mvb/feeding.py`
- **Location**: `setup_food_initially()` line 17
- **Call**: `world.food = (rng.random(world.food.shape) < p).astype(np.int8)`
- **Purpose**: Randomly places initial food across world grid
- **When**: Once at simulation start

### 3. **Brain: Neuronal (Primary)** - `mvb/brains/decisionmaking_neuronal_algorithm.py`

#### 3a. Neuron.compute_input()
- **Location**: Line 62
- **Call**: `total += rng.normal(0.0, self.noise_level)`
- **Purpose**: Adds Gaussian noise to neuron input
- **When**: Every brain update tick (multiple per decision)

#### 3b. Neuron.update()
- **Location**: Line 65-66
- **Call**: Calls `compute_input(rng)`
- **Purpose**: Triggers noise addition during neuron update
- **When**: Every brain tick

#### 3c. _output_to_decision()
- **Location**: Line 374
- **Call**: `idx, direction = active[rng.integers(len(active))]`
- **Purpose**: Randomly selects among multiple equally-valid moves
- **When**: When output neurons suggest multiple directions

#### 3d. init()
- **Location**: Line 141
- **Call**: `neuron_params, conn_matrix, sensory_mapping = init_module.build_brain_spec(rng)`
- **Purpose**: Initialize brain spec (future plasticity support)
- **When**: Once at worm initialization

### 4. **Brain: Algorithmic Variants** - `mvb/brains/decisionmaking_prio_food.py` & `decisionmaking_adjacent_food.py`

#### 4a. random_valid_move()
- **Location**: Both files
- **Call**: `choice = rng.integers(0, len(moves))`
- **Purpose**: Randomly selects among valid moves
- **When**: When brain decides to move but multiple directions valid

#### 4b. decide()
- **Location**: Both files (lines 10, 39, 25)
- **Call**: Calls `random_valid_move()`
- **Purpose**: Triggers random move selection
- **When**: On every decision call

## RNG Stream Separation Proposal

To allow consistent comparison between algo vs neuro brains, we should separate RNG into **3+ independent streams**:

### Stream 1: **WORLD_RNG** (World initialization & food dynamics)
- Initial food placement
- Food regrowth (if random)
- World-level randomness
- **Seed**: `rng_seed` (or `world_seed`)

### Stream 2: **BRAIN_DECISION_RNG** (Movement choice randomness)
- Move selection when multiple are valid (neuro output_to_decision)
- Move selection for algorithmic brains
- **Seed**: `rng_seed + 1` (or derived separately)
- **Important**: This stream MUST be identical between algo & neuro for fair comparison

### Stream 3: **NEURON_NOISE_RNG** (Neural computation noise)
- Gaussian noise in neuronal computation
- **Seed**: `rng_seed + 2` (or derived separately)
- **Important**: Can be disabled/controlled per experiment

### Optional Stream 4: **PLASTICITY_RNG** (Future)
- Neural weight changes (if plasticity implemented)
- Connection creation/pruning
- **Seed**: `rng_seed + 3`

## Implementation Strategy

### Option A: Separate RNG objects in World
```python
class World:
    def __init__(self, cfg: WorldConfig, rng_seed: int):
        self.rng_world = np.random.default_rng(rng_seed)
        self.rng_brain_decision = np.random.default_rng(rng_seed + 1)
        self.rng_neuron_noise = np.random.default_rng(rng_seed + 2)
        self.rng_plasticity = np.random.default_rng(rng_seed + 3)
```

**Pros**: Centralized, easy to track
**Cons**: World becomes dependency hub

### Option B: Separate RNG objects in Worm
```python
class Worm:
    def __init__(self, cfg: WormConfig, world: World, seed_offset: int):
        self.rng_decision = np.random.default_rng(world.cfg.rng_seed + seed_offset)
        self.rng_noise = np.random.default_rng(world.cfg.rng_seed + 1000 + seed_offset)
```

**Pros**: Worm-specific randomness (useful for multi-worm comparison)
**Cons**: Splits responsibility

### Option C: RNG Manager Object
```python
class SimulationRNG:
    def __init__(self, seed: int):
        self.world = np.random.default_rng(seed)
        self.decision = np.random.default_rng(seed + 1)
        self.noise = np.random.default_rng(seed + 2)
        self.plasticity = np.random.default_rng(seed + 3)
```

**Pros**: Explicit, testable, clear separation
**Cons**: New abstraction layer

## Files to Modify

1. **mvb/world.py** - Create separate RNG streams
2. **mvb/feeding.py** - Use `rng_world` instead of generic `rng`
3. **mvb/brains/decisionmaking_neuronal_algorithm.py** - Separate `rng_noise` and `rng_decision`
4. **mvb/brains/decisionmaking_prio_food.py** - Use `rng_decision`
5. **mvb/brains/decisionmaking_adjacent_food.py** - Use `rng_decision`
6. **mvb/worm.py** - Pass specific RNG streams where needed
7. **simulate/run_single_rec.py** - Update RNG initialization
8. **simulate/run_single.py** - Update RNG initialization
9. **simulate/run_batch_rec.py** - Update RNG initialization

## Benefits

✅ **Fair Comparison**: algo vs neuro brains see identical move-choice randomness
✅ **Noise Control**: Can disable neuron noise independently for debugging
✅ **Future-Proof**: Ready for plasticity without re-architecting
✅ **Reproducible**: Same seed combinations = same behavior
✅ **Debuggable**: Can isolate which RNG affects behavior
