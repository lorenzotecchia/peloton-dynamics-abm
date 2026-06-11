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
    rider_length: float = 1.8       # longitudinal physical footprint (a bike)
    rider_width: float = 0.6        # lateral physical footprint (shoulders)
    seed: int | None = None
