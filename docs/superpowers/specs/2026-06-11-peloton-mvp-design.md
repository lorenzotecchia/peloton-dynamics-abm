# Peloton Simulation — MVP Design

**Date:** 2026-06-11
**Status:** Approved (design phase)

## Goal

An MVP agent-based model of a competitive cycling peloton, built on Mesa 3.x, that
produces a visually meaningful demo: cyclists move along a 2-D road, naturally bunch
into drafting formations, and are colored by a continuous wind-exposure gradient
(green = sheltered, red = exposed). The game-theory and energy layers are scaffolded
as honest stubs to be filled in later.

This is **scope B** ("minimal working movement"): agents really move and drafting is
really computed, while strategy/game-theory and the real energy model stay stubbed.

## Key Decisions

| Topic | Decision |
|---|---|
| Logic scope | Working forward motion + drafting detection; strategy & energy stubbed |
| Geometry | 2-D continuous space (longitudinal distance + lateral road width) |
| Visualization | Mesa `SolaraViz` (browser tab, free controls + live charts) |
| Exposure model | Continuous `CF_draft` from README, normalized to 0..1, green→red colormap |
| Teams | `team_id` attribute carried now, no team *behavior* yet; optional viz channel |
| Lateral movement | Greedy "seek shelter" heuristic (placeholder the strategy layer overrides) |

## Module Layout

```
src/peloton/
  __init__.py
  agent.py        # CyclistAgent: state + step()
  model.py        # PelotonModel: ContinuousSpace, setup, step, DataCollector
  physics.py      # drafting & exposure: cf_draft(), neighbors_ahead(), exposure_for()
  strategy.py     # STUB: decide_action() -> action (returns default "ride")
  energy.py       # STUB: update_energy() (no-op / trivial placeholder)
  movement.py     # greedy "seek shelter" lateral rule + forward motion
  viz.py          # SolaraViz: agent portrayal (green→red gradient), charts, sliders
  config.py       # dataclass of tunable params
run_app.py        # launches SolaraViz (solara run run_app.py)
```

**Design intent:**
- `physics.py` holds pure functions (no Mesa deps) → trivially unit-testable, reused by
  both movement and the future energy model.
- `strategy.py` / `energy.py` are real modules with real signatures but stub bodies —
  the phase-2 scaffold. The agent calls them every step; today they return defaults.
- `movement.py` holds the greedy seek-shelter heuristic so it is obviously a placeholder
  the strategy layer will later override.

## Configuration

```python
@dataclass(frozen=True)
class PelotonConfig:
    road_length: float = 1000.0     # metres, finish line
    road_width: float = 8.0         # lateral extent (a few "lanes")
    n_agents: int = 30
    n_teams: int = 5
    base_speed: float = 12.0        # m/s forward, per step baseline
    speed_noise: float = 0.5
    draft_radius: float = 3.0       # README "same group within 3 m"
    draft_lateral: float = 1.0      # lateral half-width of draft cone
    seed: int | None = None
```

## Agent & Model Logic

`CyclistAgent` carries state only; behavior is delegated to the modules:

```python
class CyclistAgent(Agent):
    # state: pos (via space), team_id, energy (placeholder float), exposure (0..1), action
    def step(self):
        self.exposure = physics.exposure_for(self, self.model)   # continuous CF_draft
        self.action   = strategy.decide_action(self, self.model) # STUB -> "ride"
        energy.update_energy(self, self.model)                   # STUB -> no-op
        new_pos = movement.next_position(self, self.model)       # greedy seek-shelter + forward
        self.model.space.move_agent(self, new_pos)
```

`PelotonModel`:
- builds `ContinuousSpace(road_width, road_length, torus=False)`,
- spawns `n_agents` split round-robin into `n_teams`, random lateral position + small
  longitudinal spread at the start line,
- advances with `AgentSet.shuffle_do("step")` each tick (Mesa 3.x idiom),
- removes/parks agents past `road_length` (finished),
- `DataCollector`: mean exposure, number finished, per-team count — ready for cooperation
  metrics later.

**Step ordering:** exposure is computed at the start of each agent's step from current
positions. Good enough for the MVP; a synchronous compute-all-then-move pass can be added
later if ordering artifacts appear.

## Physics & Exposure

Pure functions, no Mesa state mutation:

```python
def cf_draft(d_w: float) -> float:
    # README formula; valid for d_w in ~[0, draft_radius]
    return 0.62 - 0.0104 * d_w + 0.0452 * d_w**2

def neighbors_ahead(agent, model) -> list[Agent]:
    # riders within draft_radius longitudinally AND draft_lateral laterally,
    # positioned in front (greater longitudinal coord)

def exposure_for(agent, model) -> float:
    # 0.0 = fully sheltered, 1.0 = fully in the wind
    ahead = neighbors_ahead(agent, model)
    if not ahead:
        return 1.0
    d_w = longitudinal distance to nearest rider ahead
    saving = 1.0 - clamp(cf_draft(d_w), 0, 1)   # CF_draft ~0.62 at d_w~0 → ~38% saving
    return clamp(1.0 - saving, 0, 1)
```

**Interpretation note:** the README's `CF_draft` is a drag multiplier (≈0.62 just behind a
wheel, rising toward 1.0 as you drop back). "Exposure" is `CF_draft` normalized so the
gradient reads intuitively (sheltered = low = green). The exact normalization /
colormap thresholds are a one-line tweak the team can calibrate later.

## Visualization

Mesa 3.x `SolaraViz`:
- `agent_portrayal(agent)` → `{"color": green_to_red(agent.exposure), "size": ..., "marker": "o"}`;
  optional team outline/tint as a secondary channel.
- Space rendered with `make_space_component`: long horizontal axis = distance to finish,
  short vertical axis = road width, so it reads like a road.
- `model_params` dict → sliders for `n_agents`, `n_teams`, `base_speed`, `draft_radius`, `seed`.
- Charts via `make_plot_component`: mean exposure over time, number finished. A
  cooperation-rate plot is pre-wired but flat until the strategy layer lands.
- Launch: `solara run run_app.py`.

## Extension Points (the seams)

| Layer | MVP today | Later |
|---|---|---|
| `strategy.decide_action` | returns `"ride"` | game-theory: cooperate/defect/breakaway probs `σ(α + β·d_finish + γ·E_left)` |
| `energy.update_energy` | no-op (or trivial constant) | exposure-driven drain + recovery; feeds speed |
| `movement.next_position` | greedy seek-shelter heuristic | strategy-driven positioning; lead-out / team logic |
| `physics` | exposure for color | same functions feed the energy model |

## Testing (MVP)

- Unit tests on `physics.py` (pure functions): `cf_draft` values, `neighbors_ahead` cone
  geometry, exposure clamping.
- One model smoke test: `N` agents, run `K` steps, assert agents advance and none escape
  the road bounds.
- Stubs need no tests until they have behavior.

## References

- `moped_python` (https://github.com/erickmartins/moped_python) — visualization/motion
  inspiration, taken with a grain of salt.
- Team spec notes in `README.md` (drafting formula, cooperation-probability parameterization).
