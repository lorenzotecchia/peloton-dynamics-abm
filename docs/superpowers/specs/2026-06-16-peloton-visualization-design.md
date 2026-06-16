# Peloton "Bullet-Hell" Visualization — Design

Date: 2026-06-16

## Goal

Live visualization of the Mesa peloton model (`genAI/model.py`, `genAI/cyclist.py`):
a scrolling 2D "road" that slides underneath the riders (bullet-hell style), plus a
separate live matplotlib stats window. Mirrors the existing neural-CA split
(`visualization/visualization.py` pygame view + `visualization/avalanche_view.py`
matplotlib view).

## Model facts the viz relies on

- Riders ride +x toward `finish_x` (default 5000) on a strip of height `road_width` (10).
- Each `Cyclist` exposes: `pos` (`np.array([x, y])`), `energy`, `group_id`,
  `is_cooperating`, `speed`.
- `CyclingRace` exposes: `agents`, `step()`, `road_width`, `finish_x`,
  `detect_groups()`, and a Mesa `datacollector` reporting `MeanEnergy` and `PelotonSize`.

## Components

### 1. `visualization/peloton_view.py` — pygame road view

Horizontal road: race runs left→right, road scrolls left, leader pinned near the
right edge.

- **Camera (follow-leader):** `leader_x = max(rider.pos[0])`. World→screen:
  `screen_x = right_margin - (leader_x - rider_x) * scale`. Leader sits near the
  right edge; slower riders trail left; droppers slide off the left edge.
  `road_width` maps to window height (`screen_y = rider_y / road_width * height`).
- **Scrolling ground:** dashed lane markings drawn at world-x positions offset by
  `leader_x` so the road appears to slide underneath. Finish line drawn once
  `leader_x` nears `finish_x`.
- **Rider glyph:**
  - Color = `group_id`, via a fixed palette indexed by group id (recomputed each
    step from `detect_groups`). Falls back to a default color if `group_id is None`.
  - Filled circle if `is_cooperating`, hollow ring if free-riding.
- **Controls:** Space = single step, Enter = run/stop, R = reset, Esc = quit.
  Same scheme as `visualization.py`.
- **Decoupling:** the renderer reads only the public interface listed above; no
  Mesa internals.

### 2. Live matplotlib stats

Reuse the dataclass pattern from `avalanche_view.py`. Two line plots redrawn every
N steps (`update_interval`, default ~10):

- **MeanEnergy** over time
- **PelotonSize** (largest group) over time

Data pulled directly from `model.datacollector.get_model_vars_dataframe()` — Mesa
already collects it, so no separate history list is kept in the view.
`ponytail:` reads the dataframe each redraw; switch to incremental append only if the
dataframe copy ever shows up as a hotspot.

### 3. Entry point

The `genAI/` model uses bare imports (`from cyclist import Cyclist`) and is not a
package. Add a small `run` entry point that builds `CyclingRace`, creates both
views, and drives the step loop (pygame loop calls `model.step()`, then both views'
`update`). Run from inside `genAI/`, or convert `genAI/` into a package and fix the
imports — decide at implementation time; default to the smaller change (run from
inside the directory).

## Out of scope (deferred)

- Speed trails / motion streaks
- Energy-based rider coloring (group color chosen instead)
- Camera zoom/orbit controls
- Centroid or zoom-to-fit camera modes

## Testing

- One runnable self-check on the world→screen camera transform: assert the leader
  maps to the right margin and a rider `delta_x` behind maps `scale * delta_x` to its
  left. Pure geometry, no pygame surface needed.
