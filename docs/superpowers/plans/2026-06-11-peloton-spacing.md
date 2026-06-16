# Peloton Physical Spacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give riders a physical footprint (1.8 m × 0.6 m) with a hard no-overlap guarantee and brake-to-follow dynamics, so the simulation forms realistic peloton lines instead of overlapping bunches.

**Architecture:** A pure `overlaps()` predicate joins the existing pure functions in `physics.py`. `movement.next_position` is reworked: lateral candidates that would overlap anyone are discarded, forward advance is braked to the wheel of the nearest rider ahead in the lane, and post-choice jitter is applied only if it keeps the invariant. `model.py` spawns riders on a non-overlapping start grid and removes finishers from the space (so they stop blocking the line), recording `finish_order`. Agents move sequentially (`shuffle_do`) and every move is validated against current positions, making no-overlap a hard invariant.

**Tech Stack:** Python 3.12, Mesa 3.5.1, pytest, uv. Spec: `docs/superpowers/specs/2026-06-11-peloton-spacing-design.md`.

---

## Conventions (read once)

- Position is `(x, y)`: `x` = longitudinal (0 = start, `road_length` = finish), `y` = lateral (0..`road_width`). "Ahead" = greater `x`.
- Two riders **overlap** when `|dx| < rider_length` **and** `|dy| < rider_width` (rectangular footprint, boundary exclusive).
- The current code state: physics has `cf_draft`, `neighbors_ahead`, `exposure_for`, `_clamp01` (exposure normalized `(cf_draft − 0.62)/0.38`). Movement clamps `new_x` at `cfg.road_length`. Model uses `shuffle_do` and counts finishers by position (they currently pin at the line — this plan changes that to removal).
- Run everything with `uv run`. The package is installed editable; `import peloton...` works everywhere.

---

## Task 1: Rider footprint config fields

**Files:**
- Modify: `src/peloton/config.py`
- Modify: `src/peloton/model.py` (the `_resolve_config` fields dict)
- Test: `tests/test_config.py`, `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_rider_footprint_defaults():
    cfg = PelotonConfig()
    assert cfg.rider_length > cfg.rider_width      # bikes are long and narrow
    assert cfg.rider_width > 0
```

Append to `tests/test_model.py`:

```python
def test_resolve_config_preserves_rider_footprint():
    base = PelotonConfig(rider_length=2.5, rider_width=0.9)
    model = PelotonModel(config=base, n_agents=6)
    assert model.config.rider_length == 2.5
    assert model.config.rider_width == 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py::test_rider_footprint_defaults tests/test_model.py::test_resolve_config_preserves_rider_footprint -v`
Expected: FAIL with `AttributeError: 'PelotonConfig' object has no attribute 'rider_length'` (and/or `TypeError` for the unexpected dataclass kwarg).

- [ ] **Step 3: Add the fields**

In `src/peloton/config.py`, add two fields to `PelotonConfig` after `draft_lateral`:

```python
    rider_length: float = 1.8       # longitudinal physical footprint (a bike)
    rider_width: float = 0.6        # lateral physical footprint (shoulders)
```

In `src/peloton/model.py`, inside `_resolve_config`, add to the `fields` dict (after the `"draft_lateral"` entry):

```python
            "rider_length": base.rider_length,
            "rider_width": base.rider_width,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_model.py -v`
Expected: PASS (all, including the two new tests)

- [ ] **Step 5: Commit**

```bash
git add src/peloton/config.py src/peloton/model.py tests/test_config.py tests/test_model.py
git commit -m "feat: add rider physical footprint to config"
```

---

## Task 2: `overlaps` predicate in physics

**Files:**
- Modify: `src/peloton/physics.py`
- Test: `tests/test_physics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_physics.py`:

```python
from peloton.physics import overlaps


def test_overlaps_requires_both_axes_close():
    kw = dict(rider_length=1.8, rider_width=0.6)
    assert overlaps((10.0, 4.0), (11.0, 4.3), **kw)        # close on both axes
    assert not overlaps((10.0, 4.0), (12.0, 4.3), **kw)    # far longitudinally
    assert not overlaps((10.0, 4.0), (11.0, 4.7), **kw)    # far laterally
    assert not overlaps((10.0, 4.0), (12.0, 4.7), **kw)    # far on both


def test_overlaps_boundary_is_exclusive():
    kw = dict(rider_length=1.8, rider_width=0.6)
    assert not overlaps((10.0, 4.0), (11.8, 4.0), **kw)    # exactly one length apart
    assert not overlaps((10.0, 4.0), (10.0, 4.6), **kw)    # exactly one width apart
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_physics.py -k overlaps -v`
Expected: FAIL with `ImportError: cannot import name 'overlaps'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/peloton/physics.py`:

```python
def overlaps(pos_a, pos_b, *, rider_length: float, rider_width: float) -> bool:
    """True when two riders' physical footprints intersect.

    A rider occupies a rectangle ``rider_length`` long (x) and ``rider_width``
    wide (y) centred on its position; two riders overlap when they are closer
    than one footprint on BOTH axes. Boundaries are exclusive: exactly one
    footprint apart is touching, not overlapping.
    """
    dx = abs(pos_a[0] - pos_b[0])
    dy = abs(pos_a[1] - pos_b[1])
    return dx < rider_length and dy < rider_width
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_physics.py -v`
Expected: PASS (all physics tests)

- [ ] **Step 5: Commit**

```bash
git add src/peloton/physics.py tests/test_physics.py
git commit -m "feat: add overlaps footprint predicate"
```

---

## Task 3: Movement rework — feasibility, braking, safe jitter

**Files:**
- Modify: `src/peloton/movement.py` (full rewrite, shown below)
- Test: `tests/test_movement.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_movement.py` (the file already has `_FakeAgent`, `_FakeSpace`, `_FakeModel`; add the import lines at the top of the file):

```python
import pytest

from peloton.physics import overlaps
```

Then append the tests:

```python
def _model_with(agents, **cfg_kwargs):
    model = _FakeModel(agents)
    model.config = PelotonConfig(**cfg_kwargs)
    return model


def test_brakes_to_wheel_when_all_lanes_blocked():
    # A wall of three riders 2 m ahead covers every lateral candidate, so the
    # rider must brake to exactly one rider_length behind the wall.
    me = _FakeAgent((100.0, 4.0))
    wall = [
        _FakeAgent((102.0, 3.4)),
        _FakeAgent((102.0, 4.0)),
        _FakeAgent((102.0, 4.6)),
    ]
    model = _model_with(
        [me] + wall, road_width=8.0, base_speed=12.0, speed_noise=0.0
    )
    new_x, new_y = next_position(me, model)
    assert new_x == pytest.approx(102.0 - model.config.rider_length)
    assert 0.0 <= new_y <= 8.0


def test_never_moves_into_an_occupied_slot():
    # A rider sits ahead-diagonal; sliding right would overlap it. The rider
    # must end up overlap-free and must not have drifted into the occupied side.
    me = _FakeAgent((100.0, 4.0))
    other = _FakeAgent((101.0, 4.7))
    model = _model_with(
        [me, other], road_width=8.0, base_speed=12.0, speed_noise=0.0
    )
    new_pos = next_position(me, model)
    assert not overlaps(
        new_pos, other.pos, rider_length=1.8, rider_width=0.6
    )
    assert new_pos[1] < 4.1          # did not slide toward the occupied slot


def test_lone_rider_still_advances_at_full_speed():
    me = _FakeAgent((100.0, 4.0))
    model = _model_with([me], road_width=8.0, base_speed=12.0, speed_noise=0.0)
    new_x, _ = next_position(me, model)
    assert new_x == pytest.approx(112.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_movement.py -v`
Expected: the three new tests FAIL (current implementation advances through the wall: `new_x == 112.0` in the braking test). The three pre-existing tests still pass.

- [ ] **Step 3: Rewrite the implementation**

Replace the entire contents of `src/peloton/movement.py` with:

```python
"""Placeholder rider motion: greedy shelter-seeking + forward advance.

Riders are physical objects: they never overlap (rectangular footprint
rider_length x rider_width) and brake to follow the wheel ahead instead of
passing through. The shelter-seeking heuristic remains a stand-in for the
future strategy layer; the no-overlap invariant, however, is permanent physics.
"""

from peloton.physics import exposure_for, overlaps

# Candidate lateral nudges (metres) evaluated each step. 0.0 keeps the line.
_LATERAL_CANDIDATES = (-0.6, -0.3, 0.0, 0.3, 0.6)

# Max lateral jitter (m). Applied only when it keeps the no-overlap invariant.
_JITTER = 0.09


class _Probe:
    """A throwaway agent used to score exposure at a candidate lateral position."""

    __slots__ = ("pos",)

    def __init__(self, pos):
        self.pos = pos


def next_position(agent, model):
    """Return the agent's next ``(x, y)`` position (never overlapping anyone).

    For each lateral candidate: discard it if the lane change itself would
    overlap someone (staying in line is always feasible); brake so we never
    advance past the wheel of the nearest rider ahead in that lane; then pick
    the lowest-exposure candidate, tie-broken by most forward progress.
    """
    cfg = model.config
    x, y = agent.pos

    advance = cfg.base_speed + model.random.uniform(-cfg.speed_noise, cfg.speed_noise)
    # Anyone who could block or shelter us is within this radius.
    search_radius = advance + cfg.rider_length + cfg.rider_width
    others = [
        o
        for o in model.space.get_neighbors(
            agent.pos, radius=search_radius, include_center=True
        )
        if o is not agent
    ]

    def _hits_anyone(pos):
        return any(
            overlaps(
                pos, o.pos,
                rider_length=cfg.rider_length, rider_width=cfg.rider_width,
            )
            for o in others
        )

    best_key = None
    best_x = x
    best_y = y
    for dy in _LATERAL_CANDIDATES:
        cand_y = min(max(y + dy, 0.0), cfg.road_width)

        # Feasibility: the lane change itself must not overlap anyone.
        # dy == 0 always survives because the current position is overlap-free.
        if _hits_anyone((x, cand_y)):
            continue

        # Braking: never advance past the wheel of the nearest rider ahead in lane.
        allowed_x = min(x + advance, cfg.road_length)
        for o in others:
            ox, oy = o.pos
            if ox > x and abs(oy - cand_y) < cfg.rider_width:
                allowed_x = min(allowed_x, ox - cfg.rider_length)
        allowed_x = max(allowed_x, x)        # never forced backward

        probe = _Probe((x, cand_y))
        exposure = exposure_for(
            probe, model,
            draft_radius=cfg.draft_radius, draft_lateral=cfg.draft_lateral,
        )
        key = (exposure, -(allowed_x - x))   # lowest exposure, then most progress
        if best_key is None or key < best_key:
            best_key = key
            best_x = allowed_x
            best_y = cand_y

    # Cosmetic lateral jitter — applied only if it keeps the invariant.
    jittered = best_y + model.random.uniform(-_JITTER, _JITTER)
    jittered = min(max(jittered, 0.0), cfg.road_width)
    if not _hits_anyone((best_x, jittered)):
        best_y = jittered

    return (best_x, best_y)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_movement.py -v`
Expected: PASS (all 6: 3 pre-existing + 3 new)

Then run the full suite: `uv run pytest -q`
Expected: **one failure**: `tests/test_model.py::test_finishers_are_counted_and_parked`. Two known causes, both fixed by Task 4: the model still spawns riders at random (overlapping spawns freeze, since every move candidate is infeasible for an already-overlapping rider), and finishers pinned at the line form a blocking wall. Do NOT try to fix this here — confirm it is the only failure and move on. Everything else passes.

- [ ] **Step 5: Commit**

```bash
git add src/peloton/movement.py tests/test_movement.py
git commit -m "feat: no-overlap movement with brake-to-follow dynamics"
```

---

## Task 4: Model — start grid, finisher removal, invariant test

**Files:**
- Modify: `src/peloton/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Update/write the tests**

In `tests/test_model.py`, REPLACE the entire `test_finishers_are_counted_and_parked` function with:

```python
def test_finishers_are_removed_and_counted():
    cfg = PelotonConfig(n_agents=10, n_teams=2, road_length=50.0,
                        base_speed=12.0, speed_noise=0.0, seed=3)
    model = PelotonModel(cfg)
    for _ in range(10):
        model.step()
    assert model.n_finished == 10
    assert len(model.agents) == 0          # finishers leave the road
    finished_ids = [uid for uid, _ in model.finish_order]
    assert len(finished_ids) == 10
    assert len(set(finished_ids)) == 10    # each rider finishes exactly once
```

Then APPEND the invariant test:

```python
def test_no_two_riders_ever_overlap():
    from peloton.physics import overlaps

    cfg = PelotonConfig(n_agents=30, n_teams=5, road_length=400.0, seed=11)
    model = PelotonModel(cfg)

    def assert_no_overlaps(step_no):
        agents = list(model.agents)
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                assert not overlaps(
                    a.pos, b.pos,
                    rider_length=cfg.rider_length, rider_width=cfg.rider_width,
                ), f"step {step_no}: agents {a.unique_id} and {b.unique_id} overlap"

    assert_no_overlaps(0)                  # spawn grid is overlap-free
    for s in range(1, 16):
        model.step()
        assert_no_overlaps(s)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model.py -v`
Expected: `test_finishers_are_removed_and_counted` FAILS (`AttributeError: 'PelotonModel' object has no attribute 'finish_order'`) and `test_no_two_riders_ever_overlap` FAILS at step 0 (random spawn overlaps).

- [ ] **Step 3: Update the model**

In `src/peloton/model.py`, replace the spawn loop in `__init__` (the block starting with the `# Spawn agents round-robin into teams...` comment) with:

```python
        self.finish_order: list[tuple[int, int]] = []

        # Spawn on a start grid with fixed clearance: non-overlapping by
        # construction. Jitter stays strictly under half the clearance so it
        # can never close the gap between neighbouring slots.
        gap = 0.2
        slot_w = config.rider_width + gap
        slot_l = config.rider_length + gap
        per_row = max(1, int(config.road_width // slot_w))
        jitter = gap / 2 - 0.01
        for i in range(config.n_agents):
            agent = CyclistAgent(self, team_id=i % config.n_teams)
            row, col = divmod(i, per_row)
            x = row * slot_l + self.random.uniform(0.0, jitter)
            y = col * slot_w + slot_w / 2 + self.random.uniform(-jitter, jitter)
            self.space.place_agent(agent, (x, y))
```

Then replace the `step` method and delete nothing else:

```python
    def step(self):
        self.agents.shuffle_do("step")
        self._remove_finishers()
        self.datacollector.collect(self)

    def _remove_finishers(self):
        """Riders that crossed the line leave the road (and stop blocking it)."""
        for agent in list(self.agents):
            if agent.pos[0] >= self.config.road_length:
                self.finish_order.append((agent.unique_id, self.steps))
                self.space.remove_agent(agent)
                agent.remove()
        self.n_finished = len(self.finish_order)
```

(Keep `self.n_finished = 0` in `__init__` as is. `self.steps` is Mesa 3's built-in step counter.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS (all model tests, including the rewritten finisher test and the invariant test)

Run the full suite: `uv run pytest -q`
Expected: ALL tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/peloton/model.py tests/test_model.py
git commit -m "feat: start grid spawning and finisher removal with finish order"
```

---

## Task 5: Viz marker size + end-to-end verification

**Files:**
- Modify: `src/peloton/viz.py`

- [ ] **Step 1: Scale the marker to the footprint**

In `src/peloton/viz.py`, in `agent_portrayal`, change `"size": 25,` to:

```python
        "size": 12,   # roughly a bike footprint at the default road scale
```

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -q`
Expected: ALL tests pass.

- [ ] **Step 3: Headless formation check**

Run:

```bash
uv run python -c "
from peloton.config import PelotonConfig
from peloton.model import PelotonModel
from peloton.physics import overlaps
cfg = PelotonConfig(n_agents=30, n_teams=5, road_length=600.0, seed=42)
m = PelotonModel(cfg)
for _ in range(25): m.step()
agents = list(m.agents)
pairs = [(a, b) for i, a in enumerate(agents) for b in agents[i+1:]]
bad = [1 for a, b in pairs if overlaps(a.pos, b.pos, rider_length=cfg.rider_length, rider_width=cfg.rider_width)]
df = m.datacollector.get_model_vars_dataframe()
print('overlapping pairs:', sum(bad))
print('min exposure seen:', round(df.MeanExposure.min(), 3))
print('riders remaining:', len(agents), '| finished:', m.n_finished)
"
```

Expected: `overlapping pairs: 0`, min exposure below 1.0 (drafting still happens), counts consistent.

- [ ] **Step 4: App smoke test**

Run: `uv run python -c "import run_app; print('app ok')"`
Expected: `app ok`. (Interactive check `uv run solara run run_app.py` is done manually by the user.)

- [ ] **Step 5: Commit**

```bash
git add src/peloton/viz.py
git commit -m "feat: scale rider marker to physical footprint"
```

---

## Done

Riders now hold a hard no-overlap invariant, brake to follow wheels (forming drafting
lines), spawn on a race-style grid, and leave the road when they finish (with
`finish_order` recorded for future game-theory payoffs). Launch with
`uv run solara run run_app.py`.
