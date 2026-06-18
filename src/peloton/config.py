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

    # --- Future knobs: SA targets, not yet read by any active code. ---
    # ponytail: placeholders; consumed when energy/strategy/evolution land.
    # Physiology (rider heterogeneity + stamina model).
    w_max10_mean: float = 400.0     # mean 10-min max power (W)
    w_max10_std: float = 68.0       # ~17% spread weakest->strongest (Trenchard 2009)
    cp_fraction: float = 0.7        # critical power as fraction of W_max10 (lactate threshold)
    recovery_rate: float = 1.0      # stamina recovery multiplier below CP
    # Grouping / pack speed / learning.
    group_radius: float = 3.0       # "<3 m apart => same group" (distinct from draft cone)
    k_s: float = 0.8                # pack speed coefficient (Martins 2013)
    learning_rate: float = 0.1      # eta: across-race coefficient update step
