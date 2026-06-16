# Peloton Bullet-Hell Road View — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing Solara road renderer (`src/peloton/viz.py`) into a scrolling "bullet-hell" view: a fixed-width camera pinned to the leader, a scrolling dashed road texture, and team-hue/exposure-brightness rider coloring.

**Architecture:** In-place edits to `draw_road` and its module in `src/peloton/viz.py`, against the `lorenzo/mvp-attempt` peloton model (Mesa `ContinuousSpace`, `CyclistAgent` with `pos`/`team_id`/`exposure`). Pure-function helpers (`rider_color`, camera-window math) are unit-tested headlessly with the matplotlib `Agg` backend; `RoadView`, the Solara plumbing, and the live `MeanExposure`/`Finished` plots are untouched.

**Tech Stack:** Python 3.12, Mesa 3.5, Solara, matplotlib (`Agg` for tests), `colorsys` (stdlib), pytest, uv.

---

## Task 0: Base the work branch on the real model

The current branch `lorenzo/visualization` is off `master` and has no `src/peloton`.
Merge the model branch in so the viz code and tests exist to edit.

**Files:** none (git only)

- [ ] **Step 1: Confirm clean tree and current branch**

Run: `git status --short && git branch --show-current`
Expected: branch is `lorenzo/visualization`; only `visualization/` shows as untracked (the unrelated genAI dir — leave it).

- [ ] **Step 2: Merge the model branch**

Run: `git merge --no-edit lorenzo/mvp-attempt`
Expected: a merge commit; no conflicts (the branches touch disjoint files — `src/peloton` + `tests` come from mvp-attempt, the spec/plan docs from here).

If conflicts appear, they will only be in shared files like `pyproject.toml`; keep the union of dependencies (mvp-attempt adds `solara`, `matplotlib`, `altair`, `networkx`, `numpy`).

- [ ] **Step 3: Sync dependencies**

Run: `uv sync`
Expected: solara/matplotlib/etc. installed, no error.

- [ ] **Step 4: Verify the existing viz tests pass before changing anything**

Run: `uv run pytest tests/test_viz.py -v`
Expected: PASS (5 tests: 3 `exposure_to_color`, 1 true-scale, 1 camera-follows, 1 empty-race — all green). This is the baseline.

---

## Task 1: `rider_color` — team hue + exposure brightness

**Files:**
- Modify: `src/peloton/viz.py` (add helper + imports)
- Test: `tests/test_viz.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_viz.py`:

```python
from peloton.viz import rider_color


def test_rider_color_differs_by_team():
    c0 = rider_color(team_id=0, n_teams=5, exposure=0.5)
    c1 = rider_color(team_id=1, n_teams=5, exposure=0.5)
    assert c0 != c1                                  # different hue per team


def test_rider_color_brighter_when_exposed():
    sheltered = rider_color(team_id=2, n_teams=5, exposure=0.0)
    exposed = rider_color(team_id=2, n_teams=5, exposure=1.0)
    assert max(exposed) > max(sheltered)             # exposed = brighter


def test_rider_color_channels_in_unit_range():
    for team in range(5):
        for e in (0.0, 0.5, 1.0):
            c = rider_color(team_id=team, n_teams=5, exposure=e)
            assert len(c) == 3
            for ch in c:
                assert 0.0 <= ch <= 1.0


def test_rider_color_handles_zero_teams():
    c = rider_color(team_id=0, n_teams=0, exposure=0.5)   # must not divide by zero
    for ch in c:
        assert 0.0 <= ch <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_viz.py -k rider_color -v`
Expected: FAIL with `ImportError: cannot import name 'rider_color'`.

- [ ] **Step 3: Implement `rider_color`**

In `src/peloton/viz.py`, add `import colorsys` near the top imports, and add this
helper next to `exposure_to_color` (leave `exposure_to_color` in place — the existing
tests still import it):

```python
def rider_color(team_id: int, n_teams: int, exposure: float) -> tuple[float, float, float]:
    """Rider fill color: hue from team, brightness from wind exposure.

    Hue is spread across teams so groups are distinguishable; HSV ``value`` rises
    with exposure (sheltered riders are darker, exposed riders brighter), keeping
    the drafting signal the model actually produces today.
    """
    hue = team_id / max(n_teams, 1)
    value = 0.45 + 0.55 * max(0.0, min(1.0, exposure))
    return colorsys.hsv_to_rgb(hue, 0.85, value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_viz.py -k rider_color -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/peloton/viz.py tests/test_viz.py
git commit -m "feat(viz): team-hue + exposure-brightness rider color"
```

---

## Task 2: Follow-leader camera (replace the symmetric bunch window)

Replace the zoom-to-fit window with a fixed-width window pinned to the leader, and
update the existing camera test to the new semantics.

**Files:**
- Modify: `src/peloton/viz.py` (constants + camera block in `draw_road`)
- Test: `tests/test_viz.py` (rewrite `test_draw_road_camera_follows_the_peloton`)

- [ ] **Step 1: Rewrite the camera test to the leader-pinned semantics**

In `tests/test_viz.py`, replace the body of `test_draw_road_camera_follows_the_peloton`
with:

```python
def test_draw_road_camera_follows_the_peloton():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from peloton.config import PelotonConfig
    from peloton.model import PelotonModel
    from peloton.viz import draw_road, CAMERA_WINDOW, LEADER_MARGIN

    cfg = PelotonConfig(n_agents=20, n_teams=4, road_length=2000.0, seed=6)
    model = PelotonModel(cfg)
    for _ in range(20):
        model.step()
    _, ax = plt.subplots()
    draw_road(model, ax)

    xs = [a.pos[0] for a in model.agents]
    x_lo, x_hi = ax.get_xlim()
    assert abs((x_hi - x_lo) - CAMERA_WINDOW) < 1e-6        # fixed-width window
    assert abs(x_hi - (max(xs) + LEADER_MARGIN)) < 1e-6     # leader pinned near right
    assert x_hi - x_lo < 2000.0                             # not the whole road
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_viz.py::test_draw_road_camera_follows_the_peloton -v`
Expected: FAIL with `ImportError: cannot import name 'CAMERA_WINDOW'`.

- [ ] **Step 3: Add the camera constants**

In `src/peloton/viz.py`, replace the two existing constants

```python
_CAMERA_MARGIN = 10.0   # metres of road shown around the bunch
_MIN_WINDOW = 60.0      # never zoom tighter than this many metres
```

with:

```python
CAMERA_WINDOW = 120.0   # metres of road visible at once (fixed-width follow window)
LEADER_MARGIN = 10.0    # metres of road shown ahead of the leader
```

- [ ] **Step 4: Replace the camera block in `draw_road`**

In `draw_road`, replace this block:

```python
    xs = [a.pos[0] for a in agents]
    x_lo = min(xs) - _CAMERA_MARGIN
    x_hi = max(xs) + _CAMERA_MARGIN
    if x_hi - x_lo < _MIN_WINDOW:
        pad = (_MIN_WINDOW - (x_hi - x_lo)) / 2
        x_lo, x_hi = x_lo - pad, x_hi + pad
    ax.set_xlim(x_lo, x_hi)
```

with:

```python
    leader_x = max(a.pos[0] for a in agents)
    x_hi = leader_x + LEADER_MARGIN
    x_lo = x_hi - CAMERA_WINDOW
    ax.set_xlim(x_lo, x_hi)
```

- [ ] **Step 5: Run the camera test to verify it passes**

Run: `uv run pytest tests/test_viz.py::test_draw_road_camera_follows_the_peloton -v`
Expected: PASS.

- [ ] **Step 6: Run the full viz suite to confirm nothing else broke**

Run: `uv run pytest tests/test_viz.py -v`
Expected: PASS (true-scale and empty-race tests still green; the empty-race path is
untouched because it returns before the camera block).

- [ ] **Step 7: Commit**

```bash
git add src/peloton/viz.py tests/test_viz.py
git commit -m "feat(viz): leader-pinned fixed-width follow camera"
```

---

## Task 3: Use `rider_color` in the draw loop

Switch the rider ellipses from `exposure_to_color` to `rider_color`.

**Files:**
- Modify: `src/peloton/viz.py` (rider-drawing loop in `draw_road`)
- Test: `tests/test_viz.py` (true-scale test already asserts one ellipse per rider; extend it to check the facecolor came from `rider_color`)

- [ ] **Step 1: Extend the true-scale test to assert the new color source**

In `tests/test_viz.py`, in `test_draw_road_renders_one_true_scale_shape_per_rider`,
after the existing `assert shapes[0].height == cfg.rider_width` line, append:

```python
    from peloton.viz import rider_color
    rider0 = list(model.agents)[0]
    expected = rider_color(rider0.team_id, cfg.n_teams, rider0.exposure)
    # matplotlib stores facecolor as RGBA; compare the RGB triple.
    assert tuple(round(c, 6) for c in shapes[0].get_facecolor()[:3]) == \
        tuple(round(c, 6) for c in expected)
```

Note: `draw_road` draws ellipses in `model.agents` order and the test reads
`shapes[0]`, so both index the same rider.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest "tests/test_viz.py::test_draw_road_renders_one_true_scale_shape_per_rider" -v`
Expected: FAIL — facecolor still comes from `exposure_to_color`, so the triple won't match.

- [ ] **Step 3: Switch the draw loop to `rider_color`**

In `draw_road`, in the `for agent in agents:` loop, replace:

```python
                facecolor=exposure_to_color(agent.exposure),
```

with:

```python
                facecolor=rider_color(agent.team_id, cfg.n_teams, agent.exposure),
```

(`cfg` is already bound at the top of `draw_road` as `cfg = model.config`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest "tests/test_viz.py::test_draw_road_renders_one_true_scale_shape_per_rider" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/peloton/viz.py tests/test_viz.py
git commit -m "feat(viz): color riders by team hue + exposure in draw_road"
```

---

## Task 4: Scrolling dashed road texture

Add a dashed centre line at world-x positions so the road visibly slides as the
window scrolls.

**Files:**
- Modify: `src/peloton/viz.py` (constants + dash block in `draw_road`)
- Test: `tests/test_viz.py` (add a dash test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_viz.py`:

```python
def test_draw_road_draws_scrolling_dashes_in_window():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from peloton.config import PelotonConfig
    from peloton.model import PelotonModel
    from peloton.viz import draw_road, DASH_PITCH

    cfg = PelotonConfig(n_agents=15, n_teams=3, road_length=2000.0, seed=8)
    model = PelotonModel(cfg)
    for _ in range(15):
        model.step()
    _, ax = plt.subplots()
    draw_road(model, ax)

    x_lo, x_hi = ax.get_xlim()
    y_mid = cfg.road_width / 2.0
    # Each dash is a Line2D drawn along the centre line; at least one falls in-window.
    centre_dashes = [
        ln for ln in ax.lines
        if len(ln.get_ydata()) and all(abs(y - y_mid) < 1e-6 for y in ln.get_ydata())
    ]
    assert centre_dashes, "expected dashed centre-line segments"
    # Dashes are spaced one DASH_PITCH apart in world-x.
    starts = sorted(ln.get_xdata()[0] for ln in centre_dashes)
    assert all(x_lo - DASH_PITCH <= s <= x_hi for s in starts)
    if len(starts) >= 2:
        gaps = [round(b - a, 6) for a, b in zip(starts, starts[1:])]
        assert all(g == round(DASH_PITCH, 6) for g in gaps)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_viz.py::test_draw_road_draws_scrolling_dashes_in_window -v`
Expected: FAIL with `ImportError: cannot import name 'DASH_PITCH'`.

- [ ] **Step 3: Add the dash constants**

In `src/peloton/viz.py`, below the camera constants, add:

```python
DASH_PITCH = 10.0       # world-metres between dash starts on the centre line
DASH_LEN = 4.0          # length of each centre-line dash (metres)
```

- [ ] **Step 4: Draw the dashes in `draw_road`**

In `draw_road`, after `ax.set_xlim(x_lo, x_hi)` (from Task 2) and before the
`for agent in agents:` loop, add:

```python
    # Scrolling centre-line: dashes at fixed world-x positions. As the window
    # follows the leader, they slide left under the riders.
    y_mid = cfg.road_width / 2.0
    import math
    k0 = math.ceil(x_lo / DASH_PITCH)
    k1 = math.floor(x_hi / DASH_PITCH)
    for k in range(k0, k1 + 1):
        dash_x = k * DASH_PITCH
        ax.plot(
            [dash_x, min(dash_x + DASH_LEN, x_hi)],
            [y_mid, y_mid],
            color="white", linewidth=1.0, solid_capstyle="butt",
        )
```

(Move `import math` to the module top if you prefer; inline is fine and keeps the diff
local.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_viz.py::test_draw_road_draws_scrolling_dashes_in_window -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/peloton/viz.py tests/test_viz.py
git commit -m "feat(viz): scrolling dashed centre-line road texture"
```

---

## Task 5: Full suite + manual smoke check

**Files:** none

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest -v`
Expected: PASS — all viz tests plus the pre-existing model/agent/movement/physics tests.

- [ ] **Step 2: Launch the app and eyeball the scrolling view**

Run: `uv run solara run run_app.py`
Expected: a browser opens; pressing play scrolls the road left with the leader pinned
near the right edge, riders colored by team and brightening when exposed, a dashed
centre line sliding underneath, and droppers leaving the left edge. The `MeanExposure`
and `Finished` plots update live. Stop with Ctrl-C.

- [ ] **Step 3: Commit any final tweaks**

```bash
git add -A
git commit -m "chore(viz): bullet-hell road view smoke-tested"
```

(Skip if nothing changed.)

---

## Self-Review notes

- **Spec coverage:** camera (Task 2), scrolling texture (Task 4), team+exposure color
  (Tasks 1, 3), preserved empty-race/true-scale/`exposure_to_color` (untouched code +
  Task 5 full run), branch merge (Task 0). All spec sections map to a task.
- **Type consistency:** `rider_color(team_id, n_teams, exposure)` signature is identical
  in its definition (Task 1), the draw loop (Task 3), and every test. Constants
  `CAMERA_WINDOW`, `LEADER_MARGIN`, `DASH_PITCH`, `DASH_LEN` are defined in Task 2/4 and
  imported by name in the tests that use them.
- **Action-based fill** is intentionally deferred (no live data); spec records the
  one-line hook.
