# Peloton Physical Spacing — Design

**Date:** 2026-06-11
**Status:** Approved (design phase)
**Builds on:** `2026-06-11-peloton-mvp-design.md` (implemented)

## Goal

Riders are physical objects that occupy space. Replace the current point-agent
behaviour (riders bunch up and overlap on the same low-exposure spot) with a hard
no-overlap guarantee and brake-to-follow dynamics, so the simulation arranges into
realistic peloton formations: single-file lines, arrowhead/diamond shapes, riders
queuing behind wheels.

## Chosen behaviour (decision: brake and follow)

When a rider's path is blocked by a rider ahead, it **brakes to sit just behind that
rider's wheel** rather than passing through or permanently swerving. It only moves
sideways when a clearly better sheltered *and free* slot exists. Soft repulsion
(boids-style) and always-swerve alternatives were rejected: repulsion permits brief
overlaps and looks like sliding; swerving produces bee-swarm weaving, not a peloton.

## Physical footprint

New config fields (`config.py`):

| Field | Default | Meaning |
|---|---|---|
| `rider_length` | `1.8` | longitudinal footprint (a bike), metres |
| `rider_width` | `0.6` | lateral footprint (shoulders/handlebars), metres |

Two riders **overlap** when both `|dx| < rider_length` **and** `|dy| < rider_width`
— a rectangular exclusion zone, long and narrow like a real bike.

New pure function in `physics.py`:

```python
overlaps(pos_a, pos_b, *, rider_length, rider_width) -> bool
```

## Movement rework (`movement.py`)

Each step, for every lateral candidate offset (same `(-0.6, -0.3, 0.0, 0.3, 0.6)`
nudges as today, clamped to the road):

1. **Feasibility** — if the candidate `y` at the rider's *current* `x` would overlap
   someone alongside or behind, discard the candidate. Staying in line (`dy = 0`)
   is always feasible because the current position is overlap-free by invariant.
2. **Braking** — within the chosen line, the rider advances at most to
   `nearest_blocker_x - rider_length`: it brakes to the wheel of the rider ahead
   instead of passing through. No blocker in range → full advance
   (`base_speed ± speed_noise`). The rider is never forced backward
   (`new_x >= x`).
3. **Scoring** — among feasible candidates pick the lowest exposure; tie-break by
   most forward progress. Sitting at a wheel is the low-exposure spot, so blocked
   riders prefer to follow — this is what forms drafting lines.
4. **Jitter** — the small post-choice lateral noise (today ±0.1 m) is applied only
   if the jittered position is still overlap-free; otherwise it is dropped. The
   invariant always wins over cosmetic noise.

**Invariant:** agents move sequentially (`AgentSet.shuffle_do`), and every move is
validated against all current positions, so at no point do two riders overlap. This
is a hard guarantee, not a tendency.

**Calibration note:** the minimum wheel-to-wheel `dx` becomes `rider_length` (1.8 m),
so maximum drafting shelter is `cf_draft(1.8) ≈ 0.75` → exposure ≈ 0.34 (instead of
~0 today). The gradient stays meaningful (followers clearly greener than leaders).
The exact interpretation of `d_w` (centre-to-centre vs wheel-to-wheel) remains the
one-line calibration tweak flagged in the MVP spec.

## Spawning (`model.py`)

Random scatter cannot guarantee a non-overlapping start. Riders now spawn on a
**staggered start grid** with a fixed clearance `gap = 0.2` m: rows behind the
start line, each row holding as many riders as fit laterally
(`road_width // (rider_width + gap)`), rows spaced `rider_length + gap` apart,
with small random jitter that stays inside each slot (< gap/2 per axis).
Non-overlapping by construction; looks like a real race start.

## Finish line (`model.py`)

Today finishers pin at the line. With blocking active, a pinned finisher becomes a
wall and riders behind could never finish. Therefore: **a rider that crosses the
line is removed from the space** (they ride off), and the model keeps:

- `n_finished` — cumulative count (monotonically non-decreasing),
- `finish_order` — list of `(unique_id, step)` in crossing order; free race-results
  data for future game-theory payoffs.

The existing "parked at the line" test changes to assert removal + monotonic count.

## Visualization (`viz.py`)

Marker size scaled down to roughly match the physical footprint so spacing reads
visually. No other changes — gradient, sliders, and charts stay as they are.

## Testing

- `physics`: `overlaps` geometry — inside/outside on each axis independently.
- `movement`: blocked rider brakes to exactly `blocker_x - rider_length`; a lateral
  candidate into an occupied slot is discarded; a lone rider advances at full speed.
- `model`: global invariant — after K steps no pair of riders overlaps; the spawn
  grid is overlap-free; finishers are removed from the space and `n_finished` only
  grows.

## Out of scope

- Energy cost of braking/accelerating (energy model is still a stub).
- Strategy-driven positioning (lead-outs, team tactics) — the movement rule remains
  a placeholder heuristic the strategy layer will later override.
- Continuous collision detection between step endpoints (positions are validated at
  step resolution only, which the sequential update makes sufficient).
