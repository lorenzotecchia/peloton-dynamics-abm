"""Across-race learning of the strategy coefficients.

A single race is one model run. Learning happens *between* races: score each
rider's outcome, then nudge their coefficients toward better-performing, similar
riders. ``run_generations`` is the outer loop carrying a population of
coefficients across races; ``evolve`` is the update rule (Francesca's notes,
normalised for stability):

    w_ij = max(0, U_j - U_i) * sim(i, j)
    delta(theta_i) = eta * ( sum_j w_ij theta_j / sum_j w_ij  -  theta_i ) + noise

i.e. move a fraction ``eta`` of the way toward the similarity-weighted mean of
the peers who did better. With eta <= 1 each update is a convex combination of
current coefficients, so the population spread contracts rather than diverging
(the raw unnormalised sum from the notes blows up: its step scales with peer
count and utility magnitude). A rider still imitates peers who did better *and*
race like them (similar engine), which is how distinct roles can emerge.
"""

import copy
import math
import statistics
import numpy as np

from peloton.model import PelotonModel

# def _assign_utilities(agents, model) -> None:
#     """Utility = finishing position (winner highest); DNF scores 0 (worst)."""
#     rank = {uid: pos for pos, (uid, _step) in enumerate(model.finish_order)}
#     n = len(agents)
#     for a in agents:
#         a.utility = (n - rank[a.unique_id]) if a.unique_id in rank else 0.0


def _assign_utilities(agents, model, cfg) -> None:
    """Utility decays exponentially by finishing position; DNF = 0."""
    rank = {uid: pos for pos, (uid, _step) in enumerate(model.finish_order)}

    decay = model.config.utility_decay  # lambda; larger -> steeper decay

    if not model.finish_order:
        return None  # nessun vincitore

    individual_utility = {}

    for a in agents:
        if a.unique_id in rank:
            pos = rank[a.unique_id]
            individual_utility[a.unique_id] = math.exp(-decay * pos)
        else:
            individual_utility[a.unique_id] = 0.0

    team_utility = np.zeros(cfg.n_teams)

    for a in agents:
        team_utility[a.team_id] += individual_utility[a.unique_id]

    # 3. Assegna a ogni agente la utility del proprio team
    for a in agents:
        a.utility = team_utility[a.team_id]


def _similarity(a, b, cfg) -> float:
    """Gaussian on the engine difference. s_m is a monotone function of w_max10,
    so w_max10 alone captures 'races like me' — no need to weight both."""
    z = (a.w_max10 - b.w_max10) / (cfg.sim_scale * cfg.w_max10_std)
    return math.exp(-0.5 * z * z)


def evolve(agents, model) -> None:
    """Update every agent's coefficients in place from this race's outcomes."""
    cfg = model.config
    rng = model.random
    _assign_utilities(agents, model, cfg)
    # New rule: a fraction of the worst riders copy/blend coefficients from
    # high-performing donors. Donor selection uses stochastic acceptance
    # (roulette-wheel) restricted to the top fraction. This drops the old
    # similarity-weighted pull in favor of a simpler imitation mechanic.
    n = len(agents)
    if n == 0:
        return

    snapshot = [copy.deepcopy(a.coeffs) for a in agents]

    # Determine donor / recipient sets by utility ranking
    idxs = sorted(range(n), key=lambda i: agents[i].utility, reverse=True)
    top_count = max(1, int(cfg.evo_top_frac * n))
    bottom_count = max(1, int(cfg.evo_bottom_frac * n))
    donor_idxs = idxs[:top_count]
    recipient_idxs = idxs[-bottom_count:]

    # Remove any accidental overlap (can happen for very small n)
    donor_set = set(donor_idxs)
    recipient_idxs = [i for i in recipient_idxs if i not in donor_set]
    if not recipient_idxs:
        return

    donors = [(j, agents[j]) for j in donor_idxs]

    def _roulette_pick(donor_list):
        # stochastic acceptance: pick uniformly then accept with prob w/max_w
        weights = [max(0.0, d.utility) for _, d in donor_list]
        max_w = max(weights) if weights else 0.0
        if max_w <= 0.0:
            return rng.choice(donor_list)[0]
        while True:
            j, d = rng.choice(donor_list)
            w = max(0.0, d.utility)
            if rng.random() < (w / max_w):
                return j

    mu = getattr(cfg, "imitation_mu", 1.0)

    for i in recipient_idxs:
        a = agents[i]
        donor_j = _roulette_pick(donors)
        for key, params in a.coeffs.items():
            for param in params:
                theta_i = snapshot[i][key][param]
                theta_d = snapshot[donor_j][key][param]
                # Blend toward donor and add small Gaussian mutation
                a.coeffs[key][param] = (
                    (1 - mu) * theta_i + mu * theta_d + rng.gauss(0.0, cfg.evo_noise)
                )


def _coeff_stats(riders) -> dict:
    """Flat ``{key.param_mean, key.param_std}`` across riders.

    Flat keys drop straight into a DataFrame; plotting the means per generation
    shows whether coefficients converge, and the stds show how much the
    population spreads into distinct roles.
    """
    stats = {}
    for key, params in riders[0].coeffs.items():  # coop / leave / follow
        for param in params:
            vals = [r.coeffs[key][param] for r in riders]
            stats[f"{key}.{param}_mean"] = statistics.mean(vals)
            stats[f"{key}.{param}_std"] = statistics.pstdev(vals)
    return stats


def _utility_stats(riders) -> dict:
    """Utility statistics: mean, std, min, max across riders after a race.

    Captures how spread out performance was and whether learning improves
    the average population utility (better strategy -> higher utility).
    """
    utilities = [r.utility for r in riders]
    if not utilities:
        return {}

    return {
        "utility_mean": statistics.mean(utilities),
        "utility_std": statistics.pstdev(utilities) if len(utilities) > 1 else 0.0,
        "utility_min": min(utilities),
        "utility_max": max(utilities),
        "utility_range": max(utilities) - min(utilities),
    }


def run_generations(n_generations: int, max_steps: int, config=None) -> list[dict]:
    """Run ``n_generations`` races in sequence, learning between them.

    Coefficients persist across races in ``population`` (one dict per spawn
    slot); each race is seeded from it and ``evolve`` writes the updates back.
    Returns a per-generation history: ``n_finished`` plus the mean/std of every
    coefficient *as it raced that generation* (so generation 0 is the initial
    population, before any learning), plus performance metrics (utilities).
    """
    population: list[dict] | None = None
    history: list[dict] = []

    for gen in range(n_generations):
        model = PelotonModel(config=config, population=population)
        for _ in range(max_steps):
            if not model.running:
                break
            model.step()

        entry = {"generation": gen, "n_finished": model.n_finished}
        # Emergent metrics averaged over this generation's race, for SA targets.
        # (Race-mean, not final step: once everyone finishes the last step is empty.)
        entry.update(model.datacollector.get_model_vars_dataframe().mean().to_dict())
        entry.update(_coeff_stats(model.riders))  # coeffs that raced this generation

        # Call evolve, which assigns utilities internally and updates coefficients
        evolve(model.riders, model)
        entry.update(_utility_stats(model.riders))  # performance metrics after race
        history.append(entry)

        # Deep copy so the next generation's agents never alias each other's dicts.
        population = [copy.deepcopy(rider.coeffs) for rider in model.riders]

    return history
