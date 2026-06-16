# Peloton "Bullet-Hell" Road View — Design

Date: 2026-06-16

## Goal

Upgrade the **existing** Solara road renderer (`src/peloton/viz.py`, `draw_road` /
`RoadView`) into a scrolling "bullet-hell" view: a fixed-width window that follows
the leader so the road appears to slide left underneath the riders and droppers
fall off the back edge. Stay in the current Solara + matplotlib stack — no pygame,
no new dependency. The live stat plots (`MeanExposure`, `Finished`) are kept as-is.

This targets the real model on the `lorenzo/mvp-attempt` branch, **not** the
`genAI/` stub. The work branch (`lorenzo/visualization`) is based off `master` and
does not contain `src/peloton`; the first task merges `lorenzo/mvp-attempt` in.

## Model facts the view relies on (src/peloton, mvp-attempt)

- `PelotonModel` uses a Mesa `ContinuousSpace`; finished riders are removed, so the
  agent set shrinks and can reach empty.
- `model.config` (`PelotonConfig`): `road_length` (x_max, finish), `road_width`
  (y_max), `n_teams`, `rider_length`, `rider_width`.
- `CyclistAgent` exposes: `pos` (`(x, y)`), `team_id` (`0..n_teams-1`),
  `exposure` (`0.0` sheltered → `1.0` exposed), `action` (stub, always `"ride"`),
  `energy` (stub).
- `model.datacollector` reports `MeanExposure` and `Finished`; `RoadView`,
  `ExposurePlot = make_plot_component("MeanExposure")`, and
  `FinishedPlot = make_plot_component("Finished")` already exist and are unchanged.

## The three changes to `draw_road`

### 1. Follow-leader camera (replaces the symmetric bunch window)

Current `draw_road` centers a window on the whole bunch (`min..max ± _CAMERA_MARGIN`,
min width `_MIN_WINDOW`), so it zooms to fit everyone. Replace with a **fixed-width
window pinned to the leader**:

- `leader_x = max(a.pos[0] for a in agents)`
- `x_hi = leader_x + LEADER_MARGIN`
- `x_lo = x_hi - CAMERA_WINDOW`
- New module constants: `CAMERA_WINDOW = 120.0` (metres shown), `LEADER_MARGIN = 10.0`
  (metres of road ahead of the leader).

Consequence (intended): the leader sits near the right edge; riders more than
`CAMERA_WINDOW - LEADER_MARGIN` metres behind the leader fall off the left edge.
This is the bullet-hell feel and is a deliberate behaviour change from "whole bunch
always visible". The finish line (`axvline` at `road_length`) stays, drawn only when
`x_lo <= road_length <= x_hi`. The empty-race banner path is unchanged.

### 2. Scrolling road texture

Draw a dashed centre line so motion is visible as the window scrolls:

- Dashes at world-x positions `k * DASH_PITCH` for integer `k` such that the dash
  lies within `[x_lo, x_hi]`, at `y = road_width / 2`, each dash `DASH_LEN` long.
- Constants: `DASH_PITCH = 10.0`, `DASH_LEN = 4.0`.
- Because dash x-positions are world coordinates and the window scrolls left, the
  dashes visibly slide underneath the riders. Drawn with `ax.plot` segments (white,
  thin) on top of the tarmac `axhspan`, beneath the riders.

### 3. Rider color: team hue + exposure brightness

Replace `exposure_to_color` usage with `rider_color(team_id, n_teams, exposure)`:

- Base hue from `team_id`: `hue = team_id / max(n_teams, 1)`.
- Brightness (HSV value) from exposure: sheltered → darker, exposed → brighter,
  e.g. `value = 0.45 + 0.55 * exposure`. Saturation fixed (e.g. `0.85`).
- Convert with `colorsys.hsv_to_rgb` (stdlib). Returns an `(r, g, b)` in `[0, 1]`.

Fill style is left ready for the future strategy layer: riders are drawn filled now
(every `action` is `"ride"`). `ponytail:` one-line hook — when `action` gains
`cooperate`/`defect`, switch free-riders to a hollow `facecolor="none"` ellipse.

`exposure_to_color` is kept (still imported by `tests/test_viz.py`) but is no longer
used by `draw_road`; `rider_color` is the new path.

## Files

- Modify: `src/peloton/viz.py` — add `CAMERA_WINDOW`, `LEADER_MARGIN`, `DASH_PITCH`,
  `DASH_LEN`, `rider_color()`; rewrite the camera block and rider-drawing loop in
  `draw_road`; add the dash-drawing block. `RoadView`, `model_params`, `build_model`,
  the plot components, and `exposure_to_color` are untouched.
- Modify: `tests/test_viz.py` — update the camera test to the leader-pinned
  semantics; add `rider_color` tests. Keep the true-scale, empty-race, and
  `exposure_to_color` tests.

## Out of scope (deferred)

- Pygame / standalone window (staying in Solara).
- Energy-based coloring, speed trails, action-based fill (no live data yet — hook left).
- New live plots beyond the existing `MeanExposure` / `Finished`.

## Testing

Run: `uv run pytest tests/test_viz.py -v` (matplotlib `Agg` backend, as the existing
tests already set).

- **Camera (rewritten):** after stepping a long-road model, assert
  `x_hi - x_lo == CAMERA_WINDOW`, the leader sits within `LEADER_MARGIN` of `x_hi`
  (`x_hi - max(xs) == LEADER_MARGIN`), and the window does not show the whole road.
- **rider_color:** different `team_id` gives different hue; higher exposure gives a
  brighter (larger max channel) color than lower exposure for the same team; all
  channels in `[0, 1]`.
- **Preserved:** true-scale ellipse-per-rider, empty-race no-crash, `exposure_to_color`
  channel/dominance tests still pass.
