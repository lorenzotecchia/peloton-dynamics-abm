from dataclasses import dataclass


@dataclass(frozen=True)
class PelotonConfig:
    """Tunable parameters for the peloton simulation. All distances in metres."""

    road_length: float = 2000.0     # finish line position (x_max)
    road_width: float = 8.0         # lateral extent of the road (y_max)
    n_agents: int = 50
    n_teams: int = 1
    rider_length: float = 1.8       # longitudinal footprint, for the viz ellipse + spawn grid
    rider_width: float = 0.6        # lateral footprint, for the viz ellipse + spawn grid
    seed: int | None = None

    # --- Physiology / stamina model (Olds power equation, W' dynamics). ---
    w_max10_mean: float = 400.0     # mean 10-min max power (W)
    w_max10_std: float = 68.0       # ~17% spread weakest->strongest (Trenchard 2009)
    cp_fraction: float = 0.7        # critical power as fraction of W_max10 (lactate threshold)
    recovery_rate: float = 0.1      # r: stamina recovery multiplier below CP
    k_aero: float = 0.18*5            # aerodynamic coefficient in P = k*cf*v^3 + c_roll*v
    c_roll: float = 3.0             # rolling-resistance coefficient
    ref_speed_frac: float = 0.9     # v_hat = ref_speed_frac * s_m, for stamina init
    dt: float = 1.0                 # seconds of race per model step

    # --- Grouping / pack speed / drafting / breakaway. ---
    group_radius: float = 3.0       # "<3 m apart => same group"
    k_s: float = 0.8                # pack speed coefficient, in [0.7, 1] (Martins 2013)
    draft_coefficient: float = 0.62 # air-power multiplier when fully sheltered (vs 1.0 leading)
    breakaway_speed_frac: float = 0.95  # solo speed of a breakaway = frac * s_m (Hoenigman)
    breakaway_cooldown_steps: int = 10  # steps during which recent breakers are kept separate from original packs

    # --- Across-race evolution. ---
    learning_rate: float = 0.1      # eta: coefficient update step
    evo_noise: float = 0.01         # std of Gaussian noise added each generation
    sim_scale: float = 1.0          # rider-similarity bandwidth (in std units)
    # --- Imitation-style evolution parameters (used by new evolution rule).
    evo_bottom_frac: float = 0.2    # fraction of worst riders to update each generation
    evo_top_frac: float = 0.1       # fraction of top riders used as donors
    imitation_mu: float = 0.75      # blend fraction toward donor (1.0 = full copy)
    elite_fraction: float = 0.0     # fraction of top riders preserved unchanged
