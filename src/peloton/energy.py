"""Rider physiology: power, the speed<->power inversion, and W' stamina.

Pure functions of speed/power and config (no Mesa). Implements Francesca's
energy model: the Olds power equation, critical power, the W' anaerobic-capacity
consume/recover dynamics, and stamina initialisation from the Tlim regression.
"""

import math


def power_required(v: float, cf_eff: float, cfg) -> float:
    """Mechanical power (W) to hold speed ``v`` (m/s) at draft factor ``cf_eff``.

    ``P = k_aero * cf_eff * v**3 + c_roll * v`` (Olds 1998). ``cf_eff`` is 1.0
    when leading (full wind) and as low as ``draft_coefficient`` when sheltered.
    """
    return cfg.k_aero * cf_eff * v**3 + cfg.c_roll * v


def solo_speed(power: float, cfg, cf_eff: float = 1.0) -> float:
    """Speed sustainable at ``power`` — inverts ``power_required`` via Newton.

    The cubic has one positive real root for positive power; the air-only term
    gives a good seed and Newton converges in a handful of steps.
    """
    if power <= 0.0:
        return 0.0
    v = (power / (cfg.k_aero * cf_eff)) ** (1.0 / 3.0)  # air-only seed
    for _ in range(20):
        f = cfg.k_aero * cf_eff * v**3 + cfg.c_roll * v - power
        fp = 3.0 * cfg.k_aero * cf_eff * v**2 + cfg.c_roll
        v -= f / fp
    return max(v, 0.0)


def critical_power(w_max10: float, cfg) -> float:
    """CP = cp_fraction * W_max10 (the ~0.7 lactate threshold)."""
    return cfg.cp_fraction * w_max10


def initial_stamina(w_max10: float, cp: float, cfg) -> float:
    """Full anaerobic work capacity W_full (J).

    Reference solo effort at ``v_hat = ref_speed_frac * s_m``: time to exhaustion
    from the Tlim regression ``ln(Tlim) = -6.351 ln(P/Wmax10) + 2.478``, then
    ``W_full = (P_req - CP) * Tlim``. Clamped at 0 (if the reference effort sits
    below CP there is no anaerobic draw to measure).
    """
    s_m = solo_speed(w_max10, cfg)
    # v_hat = cfg.ref_speed_frac * s_m
    v_hat = s_m

    p_req = power_required(v_hat, 1.0, cfg)
    t_lim = math.exp(-6.351 * math.log(p_req / w_max10) + 2.478)
    return max((p_req - cp) * t_lim, 0.0)


def init_physiology(agent, cfg) -> None:
    """Fill an agent's derived physiology from its ``w_max10`` (set at spawn)."""
    agent.cp = critical_power(agent.w_max10, cfg)
    agent.s_m = solo_speed(agent.w_max10, cfg)  # speed at W_max10 (leading)
    agent.s_cp = solo_speed(agent.cp, cfg)  # sustainable (critical) speed
    agent.w_full = initial_stamina(agent.w_max10, agent.cp, cfg)
    agent.w_prime = agent.w_full


def update_stamina(agent, p_required: float, cfg) -> None:
    """Drain W' above CP, recover (rate ``r``) below it, in place.

    W' floors at 0 (exhausted) and ceils at W_full (fully recovered).
    """
    dt = cfg.dt
    if p_required > agent.cp:
        agent.w_prime -= (p_required - agent.cp) * dt
    else:
        agent.w_prime += cfg.recovery_rate * (agent.cp - p_required) * dt
        agent.w_prime = min(agent.w_prime, agent.w_full)
    agent.w_prime = max(agent.w_prime, 0.0)
