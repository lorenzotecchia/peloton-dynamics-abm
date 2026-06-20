"""Strategy layer: cooperation contribution and breakaway/follow decisions.

Each is a logistic of an agent's learned coefficients over race features
(distance to finish, energy left, teammate presence), per Francesca's notes.
The coefficients are fixed at the values below for the first race and then
tuned across races by ``evolution.evolve``.
"""

import copy
import math

# Fixed initial coefficients. alpha = bias, beta = distance, gamma = teammates,
# delta = energy fraction. Mildly cooperative riders that defend a breakaway
# more readily when fresh and near the finish.
DEFAULT_COEFFS = {
    "coop": {"alpha": 0.0, "beta": 0.0, "gamma": 0.3, "delta": 0.5},
    "leave": {"alpha": -2.0, "beta": -0.5, "gamma": -0.5, "delta": 0.0},
    "follow": {"alpha": -1.0, "beta": 0.0, "gamma": 1.0, "delta": 1.0},
}


def default_coeffs() -> dict:
    """A fresh deep copy of the default coefficients (riders must not share dicts)."""
    return copy.deepcopy(DEFAULT_COEFFS)


def sigmoid(z: float) -> float:
    # Branch by sign so math.exp never sees a large positive argument (overflow);
    # evolution can drive coefficients large, so this must stay stable everywhere.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _distance_frac(agent, cfg) -> float:
    """Remaining distance to the finish as a fraction of the course (1 -> 0)."""
    return max(0.0, (cfg.road_length - agent.pos[0]) / cfg.road_length)


def _energy_frac(agent) -> float:
    return agent.w_prime / agent.w_full if agent.w_full else 0.0


def _teammates_in(agent, group) -> int:
    return sum(1 for o in group if o is not agent and o.team_id == agent.team_id)


def contribution(agent, group, cfg) -> float:
    """C_i = sigma(alpha + beta*d/L + gamma*T + delta*W'/W_full), in (0, 1)."""
    c = agent.coeffs["coop"]
    z = (
        c["alpha"]
        + c["beta"] * _distance_frac(agent, cfg)
        + c["gamma"] * _teammates_in(agent, group)
        + c["delta"] * _energy_frac(agent)
    )
    return sigmoid(z)


def breakaway_prob(agent, v_group, cfg) -> float:
    """theta_leave: go solo when the pack is slower than the rider could sustain.

    sigma(alpha + beta*d/L + gamma*(0.7*s_m - v_group) + delta*W'/W_full). The
    speed-deficit term is positive when the group dawdles below the rider's
    anaerobic threshold, making escape attractive.
    """
    c = agent.coeffs["leave"]
    z = (
        c["alpha"]
        + c["beta"] * _distance_frac(agent, cfg)
        + c["gamma"] * (cfg.cp_fraction * agent.s_m - v_group)
        + c["delta"] * _energy_frac(agent)
    )
    return sigmoid(z)


def follow_prob(agent, breakaway, cfg) -> float:
    """theta_follow: chase a breakaway, more readily if a teammate is in it.

    sigma(alpha + beta*d/L + gamma*T + delta*W'/W_full), T = teammates already away.
    """
    c = agent.coeffs["follow"]
    teammates = sum(1 for o in breakaway if o.team_id == agent.team_id)
    z = (
        c["alpha"]
        + c["beta"] * _distance_frac(agent, cfg)
        + c["gamma"] * teammates
        + c["delta"] * _energy_frac(agent)
    )
    return sigmoid(z)
