"""Strategy layer: the effort game and the attack-vs-ride decision.

Every step a rider either *attacks* toward the line (leaving its pack) or
*rides* with the pack, contributing an *effort* in (0, 1) to a public-goods
game that sets pack speed. Both are logistics of an agent's learned
coefficients over race features; the coefficients are sampled once and then
tuned across races by ``evolution.evolve``.

Attacking is a Bernoulli gamble (succeed -> escape and gain; fail -> blow up):
its utility is a mean-standard-deviation expected utility with loss aversion,
which is where risk aversion and loss aversion enter the model.
"""

import math
import random

from peloton import energy

# alpha = bias, beta_dist = distance, gamma_team = teammates, delta_energy =
# fatigue, gamma_match = conditional cooperation (pull toward group's effort).
# sustain: bias + k_gap*(s_sustain - v). weights: the utility tradeoffs;
# real-valued so sign carries meaning (negative lambda_risk = gambler,
# beta_loss > 1 = loss-averse).
DEFAULT_COEFF_MEANS = {
    "effort": {"alpha": 0.0, "beta_dist": 0.0, "gamma_team": 0.0,
               "delta_energy": 0.0, "gamma_match": 0.0},
    "sustain": {"bias": 0.0, "k_gap": 0.5},
    "weights": {"lambda_speed": 1.0, "lambda_energy": 1.0,
                "lambda_risk": 0.5, "beta_loss": 1.5},
}
DEFAULT_COEFF_STDS = {
    "effort": {"alpha": 1.0, "beta_dist": 1.0, "gamma_team": 1.0,
               "delta_energy": 1.0, "gamma_match": 1.0},
    "sustain": {"bias": 1.0, "k_gap": 1.0},
    "weights": {"lambda_speed": 1.0, "lambda_energy": 1.0,
                "lambda_risk": 0.5, "beta_loss": 0.5},
}


def default_coeffs(rng: random.Random | None = None) -> dict:
    """Sample a fresh set of coefficients for one rider."""
    rng = rng or random.Random()
    return {
        group: {
            name: rng.gauss(
                DEFAULT_COEFF_MEANS[group][name], DEFAULT_COEFF_STDS[group][name]
            )
            for name in params
        }
        for group, params in DEFAULT_COEFF_MEANS.items()
    }


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


def sustain_probability(agent, v_target: float, cfg, cf_eff: float = 1.0) -> float:
    """Probability the rider can hold ``v_target`` to the line.

    sigmoid(bias + k_gap*(s_sustain - v_target)): high when its sustainable
    speed beats the pace it is being judged against.
    """
    s = energy.sustainable_speed(agent, cfg, cf_eff)
    c = agent.coeffs["sustain"]
    return sigmoid(c["bias"] + c["k_gap"] * (s - v_target))


def attack_prob(agent, group, v_group: float, cfg) -> float:
    """Probability of leaving the pack to push solo toward the line.

    Mean-standard-deviation expected utility of the attack gamble, framed
    relative to staying (stay = baseline 0). Succeeds with p = p_sustain.
    The std (not variance) risk penalty stays linear in payoff scale, so all
    four weights live on one comparable scale.
    """
    w = agent.coeffs["weights"]
    s_solo = energy.sustainable_speed(agent, cfg, cf_eff=1.0)
    p = sustain_probability(agent, v_group, cfg)

    speed_gain = (s_solo - v_group) / agent.s_m
    drag_cost = w["lambda_energy"] * (
        energy.power_required(s_solo, 1.0, cfg)
        - energy.power_required(v_group, cfg.draft_coefficient, cfg)
    ) * cfg.dt / agent.w_full

    success = w["lambda_speed"] * speed_gain - drag_cost
    fail = -w["beta_loss"] * drag_cost          # spent energy, no prize
    expected = p * success + (1.0 - p) * fail
    std = math.sqrt(max(0.0, p * (1.0 - p))) * abs(success - fail)
    u = expected - w["lambda_risk"] * std
    return sigmoid(u)


def effort(agent, group, cfg) -> float:
    """Public-goods contribution c_i in (0, 1).

    sigmoid over distance, teammates, fatigue, and conditional cooperation
    (pull toward the group's mean effort from the previous step).
    """
    c = agent.coeffs["effort"]
    others = [o for o in group if o is not agent]
    mean_prev = (sum(o.effort for o in others) / len(others)) if others else agent.effort
    z = (
        c["alpha"]
        + c["beta_dist"] * _distance_frac(agent, cfg)
        + c["gamma_team"] * _teammates_in(agent, group)
        + c["delta_energy"] * _energy_frac(agent)   # more energy -> more willing to pull
        + c["gamma_match"] * (mean_prev - agent.effort)
    )
    return sigmoid(z)
