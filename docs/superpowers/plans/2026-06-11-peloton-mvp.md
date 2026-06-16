# Peloton Simulation MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Mesa 3.x agent-based model of a cycling peloton where cyclists move along a 2-D road, bunch into drafting formations via a greedy "seek shelter" rule, and are colored by a continuous wind-exposure gradient (green = sheltered → red = exposed), visualized in SolaraViz.

**Architecture:** A `src/peloton/` package. Pure drafting/exposure math lives in `physics.py` (no Mesa deps, fully unit-tested). `CyclistAgent` holds state only and delegates each step to `physics` (exposure), `strategy` (STUB), `energy` (STUB), and `movement` (greedy seek-shelter + forward motion). `PelotonModel` owns a `ContinuousSpace` (x = distance along road, y = lateral road width), spawns teams, steps agents, and collects data. `viz.py` wires a green→red gradient portrayal, sliders, and live charts into `SolaraViz`.

**Tech Stack:** Python 3.12, Mesa 3.5.1 (`mesa.space.ContinuousSpace`, `mesa.visualization.SolaraViz`), solara, matplotlib, networkx, pytest, uv.

---

## Coordinate & Concept Conventions (read once before starting)

- **Space:** `ContinuousSpace(x_max=road_length, y_max=road_width, torus=False)`. Position is `(x, y)`.
  - `x` = longitudinal position along the road, `0` = start line, `road_length` = finish.
  - `y` = lateral position across the road (0 → `road_width`).
- **"Ahead"** means a *greater* `x`.
- **`exposure`** is a float in `[0, 1]`: `0.0` = fully sheltered (tucked behind a wheel), `1.0` = fully in the wind (no one ahead).
- Mesa API facts (verified against installed 3.5.1):
  - `Agent(model)` auto-assigns `unique_id`; never pass an id yourself.
  - `space.place_agent(agent, (x, y))`, `space.move_agent(agent, (x, y))`, `space.remove_agent(agent)`.
  - `space.get_neighbors(pos, radius, include_center=True) -> list[Agent]` (Euclidean radius). It includes the center agent; filter `self` out.
  - `agent.pos` is the `(x, y)` tuple.
  - `Model.__init__(self, seed=...)`; `self.agents` is an `AgentSet`; step all via `self.agents.shuffle_do("step")`.

---

## Task 0: Project dependencies & package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/peloton/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add runtime and dev dependencies**

Edit `pyproject.toml` so the `[project]` dependencies and a dev group read exactly:

```toml
[project]
name = "abm-project-src"
version = "0.1.0"
description = "Agent-based model of a competitive cycling peloton"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "mesa>=3.5.1",
    "solara",
    "matplotlib",
    "altair",
    "networkx",
    "numpy",
]

[dependency-groups]
dev = [
    "pytest>=8",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: completes without error; installs solara, matplotlib, altair, networkx.

- [ ] **Step 3: Create empty package + test package files**

Create `src/peloton/__init__.py` with:

```python
"""Agent-based model of a competitive cycling peloton."""
```

Create `tests/__init__.py` as an empty file (no content needed).

- [ ] **Step 4: Verify Mesa + ContinuousSpace import cleanly**

Run: `uv run python -c "from mesa.space import ContinuousSpace; from mesa.visualization import SolaraViz; print('ok')"`
Expected: prints `ok` (this confirms networkx/solara are present).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/peloton/__init__.py tests/__init__.py
git commit -m "chore: add peloton package skeleton and viz dependencies"
```

---

## Task 1: Configuration dataclass

**Files:**
- Create: `src/peloton/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from peloton.config import PelotonConfig


def test_defaults_are_sane():
    cfg = PelotonConfig()
    assert cfg.road_length > cfg.road_width      # road is long and thin
    assert cfg.n_agents >= cfg.n_teams           # at least one rider per team
    assert cfg.draft_radius > 0
    assert cfg.draft_lateral > 0


def test_is_frozen():
    cfg = PelotonConfig()
    try:
        cfg.n_agents = 5
        raised = False
    except Exception:
        raised = True
    assert raised, "PelotonConfig should be immutable (frozen)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'peloton.config'`

- [ ] **Step 3: Write minimal implementation**

Create `src/peloton/config.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class PelotonConfig:
    """Tunable parameters for the peloton simulation. All distances in metres."""

    road_length: float = 1000.0     # finish line position (x_max)
    road_width: float = 8.0         # lateral extent of the road (y_max)
    n_agents: int = 30
    n_teams: int = 5
    base_speed: float = 12.0        # baseline forward advance per step (x units)
    speed_noise: float = 0.5        # uniform +/- noise added to forward advance
    draft_radius: float = 3.0       # longitudinal draft range (README "same group" < 3 m)
    draft_lateral: float = 1.0      # lateral half-width of the draft cone
    seed: int | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/peloton/config.py tests/test_config.py
git commit -m "feat: add PelotonConfig dataclass"
```

---

## Task 2: Drafting math — `cf_draft`

**Files:**
- Create: `src/peloton/physics.py`
- Test: `tests/test_physics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_physics.py`:

```python
import pytest

from peloton.physics import cf_draft


def test_cf_draft_at_zero_distance():
    # README formula: 0.62 - 0.0104*d + 0.0452*d^2, at d=0 -> 0.62
    assert cf_draft(0.0) == pytest.approx(0.62)


def test_cf_draft_increases_with_distance_within_range():
    # Within the relevant 0..3 m band, dropping further back reduces shelter
    # (drag multiplier rises back toward 1).
    assert cf_draft(3.0) > cf_draft(1.0)


def test_cf_draft_value_at_three_metres():
    # 0.62 - 0.0104*3 + 0.0452*9 = 0.62 - 0.0312 + 0.4068 = 0.9956
    assert cf_draft(3.0) == pytest.approx(0.9956)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_physics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'peloton.physics'`

- [ ] **Step 3: Write minimal implementation**

Create `src/peloton/physics.py`:

```python
"""Pure drafting / wind-exposure math. No Mesa dependencies."""


def cf_draft(d_w: float) -> float:
    """Drag multiplier for a rider whose nearest rider ahead is ``d_w`` metres away.

    From the project README: ``CF_draft = 0.62 - 0.0104*d_w + 0.0452*d_w**2``.
    ~0.62 just behind a wheel (max shelter) rising toward 1.0 as the gap grows.
    """
    return 0.62 - 0.0104 * d_w + 0.0452 * d_w**2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_physics.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/peloton/physics.py tests/test_physics.py
git commit -m "feat: add cf_draft drafting formula"
```

---

## Task 3: Neighbors ahead

**Files:**
- Modify: `src/peloton/physics.py`
- Test: `tests/test_physics.py`

We test `neighbors_ahead` with lightweight fakes so `physics` stays Mesa-free. A fake
agent only needs a `.pos` attribute; a fake model only needs `.space.get_neighbors(...)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_physics.py`:

```python
from peloton.physics import neighbors_ahead


class _FakeAgent:
    def __init__(self, pos):
        self.pos = pos


class _FakeSpace:
    """Returns every agent it was given; physics does the geometric filtering."""

    def __init__(self, agents):
        self._agents = agents

    def get_neighbors(self, pos, radius, include_center=True):
        return list(self._agents)


class _FakeModel:
    def __init__(self, agents):
        self.space = _FakeSpace(agents)


def test_neighbors_ahead_keeps_only_riders_in_front_and_in_cone():
    me = _FakeAgent((100.0, 4.0))
    in_front = _FakeAgent((101.5, 4.2))      # ahead (x bigger), within lateral cone
    behind = _FakeAgent((99.0, 4.0))         # behind -> excluded
    too_wide = _FakeAgent((101.0, 6.0))      # ahead but lateral gap 2.0 > draft_lateral
    me_again = _FakeAgent((100.0, 4.0))      # same pos as self -> not "ahead"

    model = _FakeModel([me, in_front, behind, too_wide, me_again])
    result = neighbors_ahead(
        me, model, draft_radius=3.0, draft_lateral=1.0
    )

    assert in_front in result
    assert behind not in result
    assert too_wide not in result
    assert me not in result


def test_neighbors_ahead_empty_when_alone():
    me = _FakeAgent((100.0, 4.0))
    model = _FakeModel([me])
    assert neighbors_ahead(me, model, draft_radius=3.0, draft_lateral=1.0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_physics.py -k neighbors_ahead -v`
Expected: FAIL with `ImportError: cannot import name 'neighbors_ahead'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/peloton/physics.py`:

```python
def neighbors_ahead(agent, model, *, draft_radius: float, draft_lateral: float):
    """Riders directly in front of ``agent`` within the drafting cone.

    A neighbour counts when it is strictly ahead (greater x), within
    ``draft_radius`` longitudinally, and within ``draft_lateral`` laterally.
    """
    x, y = agent.pos
    # Search radius covers the longitudinal range; we filter the cone precisely below.
    candidates = model.space.get_neighbors(
        agent.pos, radius=draft_radius, include_center=True
    )
    ahead = []
    for other in candidates:
        if other is agent:
            continue
        ox, oy = other.pos
        if ox <= x:
            continue                          # not ahead
        if ox - x > draft_radius:
            continue                          # too far forward
        if abs(oy - y) > draft_lateral:
            continue                          # outside lateral cone
        ahead.append(other)
    return ahead
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_physics.py -k neighbors_ahead -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/peloton/physics.py tests/test_physics.py
git commit -m "feat: add neighbors_ahead draft-cone filter"
```

---

## Task 4: Exposure value

**Files:**
- Modify: `src/peloton/physics.py`
- Test: `tests/test_physics.py`

`exposure_for` maps the nearest rider ahead to a `[0, 1]` value via `cf_draft`. Max
shelter (`cf_draft ≈ 0.62`) → biggest saving → lowest exposure. No rider ahead → `1.0`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_physics.py`:

```python
from peloton.physics import exposure_for


def test_exposure_is_one_when_alone():
    me = _FakeAgent((100.0, 4.0))
    model = _FakeModel([me])
    assert exposure_for(me, model, draft_radius=3.0, draft_lateral=1.0) == 1.0


def test_exposure_is_lower_when_tucked_close_behind():
    me = _FakeAgent((100.0, 4.0))
    close = _FakeAgent((100.2, 4.0))     # ~0.2 m behind a wheel -> strong shelter
    model = _FakeModel([me, close])
    exp = exposure_for(me, model, draft_radius=3.0, draft_lateral=1.0)
    assert 0.0 <= exp < 0.5              # well sheltered


def test_exposure_in_unit_interval_and_grows_with_gap():
    me = _FakeAgent((100.0, 4.0))
    far = _FakeAgent((102.8, 4.0))       # near edge of draft range -> little shelter
    model_far = _FakeModel([me, far])
    near = _FakeAgent((100.2, 4.0))
    model_near = _FakeModel([me, near])

    exp_far = exposure_for(me, model_far, draft_radius=3.0, draft_lateral=1.0)
    exp_near = exposure_for(me, model_near, draft_radius=3.0, draft_lateral=1.0)

    assert 0.0 <= exp_near <= 1.0
    assert 0.0 <= exp_far <= 1.0
    assert exp_far > exp_near            # bigger gap = more exposed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_physics.py -k exposure -v`
Expected: FAIL with `ImportError: cannot import name 'exposure_for'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/peloton/physics.py`:

```python
def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def exposure_for(agent, model, *, draft_radius: float, draft_lateral: float) -> float:
    """Wind exposure of ``agent`` in ``[0, 1]``.

    ``0.0`` = fully sheltered, ``1.0`` = fully in the wind. Computed from the
    nearest rider ahead inside the draft cone: a low ``cf_draft`` (deep shelter)
    yields a large saving and therefore low exposure.
    """
    ahead = neighbors_ahead(
        agent, model, draft_radius=draft_radius, draft_lateral=draft_lateral
    )
    if not ahead:
        return 1.0

    x = agent.pos[0]
    nearest_gap = min(other.pos[0] - x for other in ahead)
    saving = 1.0 - _clamp01(cf_draft(nearest_gap))   # 0.62 -> 0.38 saving at the wheel
    return _clamp01(1.0 - saving)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_physics.py -v`
Expected: PASS (all physics tests pass)

- [ ] **Step 5: Commit**

```bash
git add src/peloton/physics.py tests/test_physics.py
git commit -m "feat: add exposure_for wind-exposure value"
```

---

## Task 5: Strategy and energy stubs

**Files:**
- Create: `src/peloton/strategy.py`
- Create: `src/peloton/energy.py`
- Test: `tests/test_stubs.py`

These are deliberate placeholders with real signatures. They must be safe to call every
step and must not mutate position. The game-theory and energy models replace them later.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stubs.py`:

```python
from peloton import energy, strategy


class _Agent:
    def __init__(self):
        self.energy = 100.0
        self.action = None


def test_decide_action_returns_ride_default():
    a = _Agent()
    assert strategy.decide_action(a, model=None) == "ride"


def test_update_energy_is_noop_for_now():
    a = _Agent()
    before = a.energy
    energy.update_energy(a, model=None)
    assert a.energy == before     # stub does not change energy yet
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stubs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'peloton.strategy'`

- [ ] **Step 3: Write minimal implementation**

Create `src/peloton/strategy.py`:

```python
"""STUB: strategy / game-theory layer.

Real version will return one of {"ride", "cooperate", "defect", "breakaway"}
based on learned probabilities sigma(alpha + beta*d_finish + gamma*E_left).
For the MVP every rider simply rides.
"""


def decide_action(agent, model) -> str:
    """Return the action a rider takes this step. STUB: always ``"ride"``."""
    return "ride"
```

Create `src/peloton/energy.py`:

```python
"""STUB: energy model.

Real version will drain energy as a function of exposure and speed, and recover
when sheltered, then feed back into achievable speed. For the MVP this is a no-op
so movement stays purely geometric.
"""


def update_energy(agent, model) -> None:
    """Update ``agent.energy`` in place. STUB: does nothing yet."""
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stubs.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/peloton/strategy.py src/peloton/energy.py tests/test_stubs.py
git commit -m "feat: add strategy and energy stubs"
```

---

## Task 6: Movement — greedy seek-shelter + forward motion

**Files:**
- Create: `src/peloton/movement.py`
- Test: `tests/test_movement.py`

`next_position(agent, model)` returns a new `(x, y)`:
- **Forward:** advance `x` by `base_speed ± speed_noise`.
- **Lateral:** sample a few candidate `y` offsets, keep the one giving the lowest
  exposure at the agent's *current* x (greedy shelter-seeking), plus small noise.
- **Clamp** `y` into `[0, road_width]`; `x` is left unclamped (finish handled by the model).

Movement reads tunables from `model.config` (a `PelotonConfig`) and randomness from
`model.random` (Mesa seeds this). Tests use fakes providing both.

- [ ] **Step 1: Write the failing test**

Create `tests/test_movement.py`:

```python
import random

from peloton.config import PelotonConfig
from peloton.movement import next_position


class _FakeAgent:
    def __init__(self, pos):
        self.pos = pos


class _FakeSpace:
    def __init__(self, agents):
        self._agents = agents

    def get_neighbors(self, pos, radius, include_center=True):
        return list(self._agents)


class _FakeModel:
    def __init__(self, agents, seed=0):
        self.space = _FakeSpace(agents)
        self.config = PelotonConfig(road_width=8.0, base_speed=12.0, speed_noise=0.5)
        self.random = random.Random(seed)


def test_moves_forward_by_about_base_speed():
    me = _FakeAgent((100.0, 4.0))
    model = _FakeModel([me])
    new_x, _ = next_position(me, model)
    assert 100.0 + 12.0 - 0.5 <= new_x <= 100.0 + 12.0 + 0.5


def test_lateral_position_stays_within_road():
    me = _FakeAgent((100.0, 0.1))
    model = _FakeModel([me])
    for _ in range(50):
        x, y = next_position(me, model)
        me.pos = (x, y)
        assert 0.0 <= y <= 8.0


def test_seeks_shelter_when_a_rider_is_ahead_on_one_side():
    # A shelter-giver sits just ahead at y=6. Starting between open air (y=2)
    # and the wheel at y=6, greedy seek-shelter should not drift further from it
    # on average. We assert the rule runs and yields an in-road y.
    me = _FakeAgent((100.0, 4.0))
    shelter = _FakeAgent((100.5, 6.0))
    model = _FakeModel([me, shelter])
    x, y = next_position(me, model)
    assert 0.0 <= y <= 8.0
    assert x > 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_movement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'peloton.movement'`

- [ ] **Step 3: Write minimal implementation**

Create `src/peloton/movement.py`:

```python
"""Placeholder rider motion: greedy shelter-seeking + forward advance.

This is a heuristic stand-in. The strategy layer will eventually drive positioning
(lead-outs, team tactics); for the MVP riders just tuck toward the lowest-exposure
lateral spot near them.
"""

from peloton.physics import exposure_for

# Candidate lateral nudges (metres) evaluated each step. 0.0 keeps the line.
_LATERAL_CANDIDATES = (-0.6, -0.3, 0.0, 0.3, 0.6)


class _Probe:
    """A throwaway agent used to score exposure at a candidate lateral position."""

    __slots__ = ("pos",)

    def __init__(self, pos):
        self.pos = pos


def next_position(agent, model):
    """Return the agent's next ``(x, y)`` position."""
    cfg = model.config
    x, y = agent.pos

    # Forward advance with uniform noise.
    advance = cfg.base_speed + model.random.uniform(-cfg.speed_noise, cfg.speed_noise)
    new_x = x + advance

    # Greedy shelter-seeking: pick the candidate lateral offset with lowest exposure,
    # evaluated at the CURRENT x against current neighbour positions.
    best_y = y
    best_exposure = None
    for dy in _LATERAL_CANDIDATES:
        cand_y = min(max(y + dy, 0.0), cfg.road_width)
        probe = _Probe((x, cand_y))
        exp = exposure_for(
            probe,
            model,
            draft_radius=cfg.draft_radius,
            draft_lateral=cfg.draft_lateral,
        )
        if best_exposure is None or exp < best_exposure:
            best_exposure = exp
            best_y = cand_y

    # Small lateral noise, then clamp into the road.
    new_y = best_y + model.random.uniform(-0.1, 0.1)
    new_y = min(max(new_y, 0.0), cfg.road_width)

    return (new_x, new_y)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_movement.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/peloton/movement.py tests/test_movement.py
git commit -m "feat: add greedy seek-shelter movement rule"
```

---

## Task 7: CyclistAgent

**Files:**
- Create: `src/peloton/agent.py`
- Test: `tests/test_agent.py`

The agent holds state and orchestrates the per-step calls. We test it through a real
`PelotonModel`-shaped fake is awkward, so instead we build the real model in Task 8 and
test the agent's *attributes and step wiring* here using a minimal real Mesa model.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent.py`:

```python
from mesa import Model
from mesa.space import ContinuousSpace

from peloton.agent import CyclistAgent
from peloton.config import PelotonConfig


class _MiniModel(Model):
    """Just enough of a model to host one agent for wiring tests."""

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self.config = PelotonConfig()
        self.space = ContinuousSpace(
            self.config.road_length, self.config.road_width, torus=False
        )


def test_agent_has_team_and_initial_state():
    model = _MiniModel()
    agent = CyclistAgent(model, team_id=2)
    model.space.place_agent(agent, (0.0, 4.0))
    assert agent.team_id == 2
    assert agent.exposure == 1.0       # no neighbours yet -> fully exposed
    assert agent.action is None
    assert isinstance(agent.energy, float)


def test_step_updates_exposure_action_and_advances_x():
    model = _MiniModel()
    agent = CyclistAgent(model, team_id=0)
    model.space.place_agent(agent, (0.0, 4.0))
    agent.step()
    assert agent.action == "ride"
    assert 0.0 <= agent.exposure <= 1.0
    assert agent.pos[0] > 0.0          # advanced forward
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'peloton.agent'`

- [ ] **Step 3: Write minimal implementation**

Create `src/peloton/agent.py`:

```python
"""The cyclist agent: state plus per-step orchestration."""

from mesa import Agent

from peloton import energy, movement, strategy
from peloton.physics import exposure_for


class CyclistAgent(Agent):
    """A single rider. Holds state; delegates behaviour to the peloton modules."""

    def __init__(self, model, team_id: int):
        super().__init__(model)
        self.team_id = team_id
        self.energy = 100.0          # placeholder; energy.update_energy is a stub
        self.exposure = 1.0          # updated each step from drafting geometry
        self.action = None

    def step(self):
        cfg = self.model.config
        self.exposure = exposure_for(
            self, self.model,
            draft_radius=cfg.draft_radius,
            draft_lateral=cfg.draft_lateral,
        )
        self.action = strategy.decide_action(self, self.model)
        energy.update_energy(self, self.model)
        new_pos = movement.next_position(self, self.model)
        self.model.space.move_agent(self, new_pos)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/peloton/agent.py tests/test_agent.py
git commit -m "feat: add CyclistAgent with per-step orchestration"
```

---

## Task 8: PelotonModel

**Files:**
- Create: `src/peloton/model.py`
- Test: `tests/test_model.py`

The model builds the space, spawns agents into teams at the start line, steps everyone,
parks finishers (riders past `road_length` stop advancing), and collects data.

- [ ] **Step 1: Write the failing test**

Create `tests/test_model.py`:

```python
from peloton.config import PelotonConfig
from peloton.model import PelotonModel


def test_model_spawns_all_agents_across_teams():
    cfg = PelotonConfig(n_agents=12, n_teams=4, seed=1)
    model = PelotonModel(cfg)
    assert len(model.agents) == 12
    teams = {a.team_id for a in model.agents}
    assert teams == {0, 1, 2, 3}


def test_agents_advance_and_stay_in_road_bounds():
    cfg = PelotonConfig(n_agents=20, n_teams=5, road_length=300.0, seed=2)
    model = PelotonModel(cfg)
    start_x = {a.unique_id: a.pos[0] for a in model.agents}
    for _ in range(10):
        model.step()
    for a in model.agents:
        assert a.pos[0] >= start_x[a.unique_id]        # never moved backward overall
        assert 0.0 <= a.pos[1] <= cfg.road_width       # stays on the road laterally


def test_finishers_are_counted_and_parked():
    cfg = PelotonConfig(n_agents=10, n_teams=2, road_length=50.0,
                        base_speed=12.0, speed_noise=0.0, seed=3)
    model = PelotonModel(cfg)
    for _ in range(10):                                # 10*12 = 120 m > 50 m road
        model.step()
    assert model.n_finished == 10
    for a in model.agents:
        assert a.pos[0] >= cfg.road_length             # parked at/after the line


def test_datacollector_records_mean_exposure():
    cfg = PelotonConfig(n_agents=15, n_teams=3, seed=4)
    model = PelotonModel(cfg)
    model.step()
    df = model.datacollector.get_model_vars_dataframe()
    assert "MeanExposure" in df.columns
    assert "Finished" in df.columns
    assert 0.0 <= df["MeanExposure"].iloc[-1] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'peloton.model'`

- [ ] **Step 3: Write minimal implementation**

Create `src/peloton/model.py`:

```python
"""The peloton model: space, agent spawning, stepping, and data collection."""

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace

from peloton.agent import CyclistAgent
from peloton.config import PelotonConfig


def _mean_exposure(model: "PelotonModel") -> float:
    agents = list(model.agents)
    if not agents:
        return 0.0
    return sum(a.exposure for a in agents) / len(agents)


class PelotonModel(Model):
    """A road full of cyclists that drift into drafting formations."""

    def __init__(self, config: PelotonConfig | None = None):
        config = config or PelotonConfig()
        super().__init__(seed=config.seed)
        self.config = config
        self.n_finished = 0

        self.space = ContinuousSpace(
            config.road_length, config.road_width, torus=False
        )

        # Spawn agents round-robin into teams, spread just behind the start line.
        for i in range(config.n_agents):
            agent = CyclistAgent(self, team_id=i % config.n_teams)
            x = self.random.uniform(0.0, 5.0)               # small start spread
            y = self.random.uniform(0.0, config.road_width)
            self.space.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "MeanExposure": _mean_exposure,
                "Finished": lambda m: m.n_finished,
            }
        )
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")
        self._park_finishers()
        self.datacollector.collect(self)

    def _park_finishers(self):
        """Stop riders that crossed the line; clamp them to the finish position."""
        finished = 0
        for agent in self.agents:
            x, y = agent.pos
            if x >= self.config.road_length:
                finished += 1
                if x > self.config.road_length:
                    self.space.move_agent(agent, (self.config.road_length, y))
        self.n_finished = finished
```

Note on `test_finishers_are_counted_and_parked`: agents whose `step()` carries them past
the line are clamped *after* the step inside `_park_finishers`. Because parked riders are
re-clamped every subsequent step, they never advance beyond `road_length`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from Tasks 1-8 pass)

- [ ] **Step 6: Commit**

```bash
git add src/peloton/model.py tests/test_model.py
git commit -m "feat: add PelotonModel with spawning, stepping, data collection"
```

---

## Task 9: Visualization (SolaraViz)

**Files:**
- Create: `src/peloton/viz.py`
- Create: `run_app.py`
- Test: `tests/test_viz.py`

The gradient lives in `exposure_to_color` (pure, testable). `agent_portrayal` and the
`SolaraViz` wiring are thin glue verified only by an import smoke test (rendering itself
is interactive and not unit-tested).

- [ ] **Step 1: Write the failing test**

Create `tests/test_viz.py`:

```python
from peloton.viz import exposure_to_color


def test_full_shelter_is_green_ish():
    r, g, b = exposure_to_color(0.0)
    assert g > r                       # sheltered -> green dominates

def test_full_exposure_is_red_ish():
    r, g, b = exposure_to_color(1.0)
    assert r > g                       # exposed -> red dominates

def test_color_channels_in_unit_range():
    for e in (0.0, 0.25, 0.5, 0.75, 1.0):
        r, g, b = exposure_to_color(e)
        for c in (r, g, b):
            assert 0.0 <= c <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_viz.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'peloton.viz'`

- [ ] **Step 3: Write minimal implementation**

Create `src/peloton/viz.py`:

```python
"""SolaraViz wiring and the exposure -> color gradient."""

from mesa.visualization import SolaraViz, make_plot_component, make_space_component
from mesa.visualization.user_param import Slider

from peloton.config import PelotonConfig
from peloton.model import PelotonModel


def exposure_to_color(exposure: float) -> tuple[float, float, float]:
    """Map exposure in [0, 1] to an RGB tuple: green (sheltered) -> red (exposed)."""
    e = max(0.0, min(1.0, exposure))
    return (e, 1.0 - e, 0.0)            # (r, g, b)


def agent_portrayal(agent):
    return {
        "color": exposure_to_color(agent.exposure),
        "size": 25,
        "marker": "o",
    }


model_params = {
    "config": PelotonConfig(),          # fixed base config; sliders override fields below
    "n_agents": Slider("Number of riders", value=30, min=5, max=100, step=5),
    "n_teams": Slider("Number of teams", value=5, min=1, max=10, step=1),
    "base_speed": Slider("Base speed", value=12.0, min=4.0, max=20.0, step=1.0),
    "draft_radius": Slider("Draft radius (m)", value=3.0, min=1.0, max=6.0, step=0.5),
}


def build_model(n_agents=30, n_teams=5, base_speed=12.0, draft_radius=3.0, config=None):
    """Factory SolaraViz calls with slider values to (re)create the model."""
    base = config or PelotonConfig()
    cfg = PelotonConfig(
        road_length=base.road_length,
        road_width=base.road_width,
        n_agents=int(n_agents),
        n_teams=int(n_teams),
        base_speed=float(base_speed),
        speed_noise=base.speed_noise,
        draft_radius=float(draft_radius),
        draft_lateral=base.draft_lateral,
        seed=base.seed,
    )
    return PelotonModel(cfg)


SpaceGraph = make_space_component(agent_portrayal)
ExposurePlot = make_plot_component("MeanExposure")
FinishedPlot = make_plot_component("Finished")


def make_page():
    model = build_model()
    return SolaraViz(
        model,
        components=[SpaceGraph, ExposurePlot, FinishedPlot],
        model_params=model_params,
        name="Cycling Peloton MVP",
    )
```

Note: `model_params` passes a fixed `config` plus slider-driven overrides. Mesa
instantiates the model by calling the model class with these kwargs, so the model factory
path is exercised through `build_model` in `run_app.py` (next step), keeping `PelotonModel`'s
own constructor signature (`config`) unchanged.

- [ ] **Step 4: Run color tests + import smoke test**

Run: `uv run pytest tests/test_viz.py -v`
Expected: PASS (3 passed)

Run: `uv run python -c "import peloton.viz; print('viz import ok')"`
Expected: prints `viz import ok`

- [ ] **Step 5: Create the app entry point**

Create `run_app.py`:

```python
"""Launch the peloton visualization.

Run with:  uv run solara run run_app.py
"""

from peloton.viz import build_model
from peloton.viz import SpaceGraph, ExposurePlot, FinishedPlot, model_params
from mesa.visualization import SolaraViz

model = build_model()

page = SolaraViz(
    model,
    components=[SpaceGraph, ExposurePlot, FinishedPlot],
    model_params=model_params,
    name="Cycling Peloton MVP",
)
```

- [ ] **Step 6: Smoke-test the app module imports**

Run: `uv run python -c "import run_app; print('app ok')"`
Expected: prints `app ok` (full interactive launch is `uv run solara run run_app.py`, done manually).

- [ ] **Step 7: Commit**

```bash
git add src/peloton/viz.py run_app.py tests/test_viz.py
git commit -m "feat: add SolaraViz visualization with exposure gradient"
```

---

## Task 10: Wire model_params override into PelotonModel + README run note

**Files:**
- Modify: `src/peloton/model.py`
- Modify: `src/peloton/viz.py`
- Test: `tests/test_model.py`

SolaraViz's `model_params` calls the model class directly with the slider kwargs, so
`PelotonModel` must accept those kwargs. We add an optional keyword path that builds a
config from overrides, keeping the `config=` path intact for tests.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model.py`:

```python
def test_model_accepts_keyword_overrides_for_viz():
    model = PelotonModel(n_agents=8, n_teams=2, base_speed=10.0, draft_radius=2.5)
    assert len(model.agents) == 8
    assert model.config.n_teams == 2
    assert model.config.base_speed == 10.0
    assert model.config.draft_radius == 2.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py::test_model_accepts_keyword_overrides_for_viz -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'n_agents'`

- [ ] **Step 3: Update PelotonModel constructor**

Replace the `__init__` signature and the first config line in `src/peloton/model.py`. The
current start is:

```python
    def __init__(self, config: PelotonConfig | None = None):
        config = config or PelotonConfig()
        super().__init__(seed=config.seed)
```

Replace with:

```python
    def __init__(self, config: PelotonConfig | None = None, **overrides):
        config = self._resolve_config(config, overrides)
        super().__init__(seed=config.seed)
```

Then add this static method just above `def step(self):`:

```python
    @staticmethod
    def _resolve_config(config: PelotonConfig | None, overrides: dict) -> PelotonConfig:
        """Build a config, applying any keyword overrides (used by SolaraViz sliders)."""
        base = config or PelotonConfig()
        if not overrides:
            return base
        fields = {
            "road_length": base.road_length,
            "road_width": base.road_width,
            "n_agents": base.n_agents,
            "n_teams": base.n_teams,
            "base_speed": base.base_speed,
            "speed_noise": base.speed_noise,
            "draft_radius": base.draft_radius,
            "draft_lateral": base.draft_lateral,
            "seed": base.seed,
        }
        for key, value in overrides.items():
            if key not in fields:
                raise TypeError(f"Unknown model parameter: {key!r}")
            if key in ("n_agents", "n_teams"):
                value = int(value)
            elif key != "seed":
                value = float(value)
            fields[key] = value
        return PelotonConfig(**fields)
```

- [ ] **Step 4: Simplify viz.py to drop the now-redundant build_model override logic**

In `src/peloton/viz.py`, replace the `build_model` function body with a thin wrapper that
defers to the model's own override handling:

```python
def build_model(n_agents=30, n_teams=5, base_speed=12.0, draft_radius=3.0, config=None):
    """Factory used for the standalone app launch."""
    return PelotonModel(
        config=config,
        n_agents=n_agents,
        n_teams=n_teams,
        base_speed=base_speed,
        draft_radius=draft_radius,
    )
```

- [ ] **Step 5: Run the affected tests**

Run: `uv run pytest tests/test_model.py tests/test_viz.py -v`
Expected: PASS (all pass, including the new override test)

- [ ] **Step 6: Add a run note to the README**

Append to `README.md`:

```markdown

## Running the MVP simulation

```bash
uv sync
uv run solara run run_app.py   # opens the interactive peloton visualization
uv run pytest                  # run the test suite
```
```

- [ ] **Step 7: Full suite + commit**

Run: `uv run pytest -v`
Expected: PASS (entire suite green)

```bash
git add src/peloton/model.py src/peloton/viz.py tests/test_model.py README.md
git commit -m "feat: support slider overrides in PelotonModel; document run command"
```

---

## Done

At this point you have: a tested `physics` core, a stubbed strategy/energy layer, a greedy
seek-shelter movement rule, a `CyclistAgent` + `PelotonModel`, and an interactive
SolaraViz app showing the green→red wind-exposure gradient with live charts and sliders.
Launch it with `uv run solara run run_app.py`.
