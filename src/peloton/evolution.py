"""Across-race learning of the strategy coefficients.

A single race is one model run. Learning happens *between* races: score each
rider's outcome, then let the weaker riders imitate a donor sampled by a logit
choice rule over utility advantage. ``run_generations`` is the outer loop
carrying a population of coefficients across races; ``evolve`` is the update
rule. The recipient blends toward the chosen donor and then receives small
Gaussian mutation.

The logit temperature (``config.logit_lambda``) is the bounded-rationality
knob: large values make donor choice nearly deterministic, while small values
flatten the choice distribution.
"""

import copy
import math
import statistics

from peloton.model import PelotonModel


def _assign_utilities(agents, model) -> None:
    """Utility decays exponentially by finishing position; DNF = 0."""
    rank = {uid: pos for pos, (uid, _step) in enumerate(model.finish_order)}

    decay = 0.4  # smaller -> steeper decay

    if not model.finish_order:
        return None  # nessun vincitore

    winner_uid = model.finish_order[0][0]
    winner = next((a for a in agents if a.unique_id == winner_uid), None)
    team_winner = getattr(winner, "team_id", None)

    for a in agents:
        if a.unique_id in rank:
            pos = rank[a.unique_id]
            a.utility = 2 * math.exp(-decay * pos)
        else:
            a.utility = 0.0
        if getattr(a, "team_id", None) == team_winner:
            a.utility += 0.0


def evolve(agents, model) -> None:
    """Update every agent's coefficients in place from this race's outcomes."""
    cfg = model.config
    rng = model.random
    _assign_utilities(agents, model)
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

    donors = donor_idxs
    mu = getattr(cfg, "imitation_mu", 1.0)
    logit_lambda = max(getattr(cfg, "logit_lambda", 1.0), 1e-12)

    def _logit_pick(recipient_idx):
        recipient_utility = agents[recipient_idx].utility
        logits = [
            (agents[j].utility - recipient_utility) * logit_lambda
            for j in donors
        ]
        max_logit = max(logits)
        weights = [math.exp(value - max_logit) for value in logits]
        return rng.choices(donors, weights=weights, k=1)[0]

    for i in recipient_idxs:
        a = agents[i]
        donor_j = _logit_pick(i)
        for key, params in a.coeffs.items():
            for param in params:
                theta_i = snapshot[i][key][param]
                theta_d = snapshot[donor_j][key][param]
                # Blend toward donor and add small Gaussian mutation
                a.coeffs[key][param] = (1 - mu) * theta_i + mu * theta_d + rng.gauss(0.0, cfg.evo_noise)


def _coeff_stats(riders) -> dict:
    """Flat ``{key.param_mean, key.param_std}`` across riders.

    Flat keys drop straight into a DataFrame; plotting the means per generation
    shows whether coefficients converge, and the stds show how much the
    population spreads into distinct roles.
    """
    stats = {}
    for key, params in riders[0].coeffs.items():       # coop / leave / follow
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


def _mean_finishing_time(model) -> float | None:
    """Mean finishing time in seconds for riders that completed the race."""
    if not model.finish_order:
        return None
    finish_times = [step * model.config.dt for _uid, step in model.finish_order]
    return statistics.mean(finish_times)


def run_generations(n_generations: int, max_steps: int, config=None) -> list[dict]:
    """Run ``n_generations`` races in sequence, learning between them.

    Coefficients persist across races in ``population`` (one dict per spawn
    slot); each race is seeded from it and ``evolve`` writes the updates back.
    Returns a per-generation history: ``n_finished``, ``mean_finishing_time``,
    plus the mean/std of every coefficient *as it raced that generation* (so
    generation 0 is the initial population, before any learning), plus
    performance metrics (utilities).
    """
    population: list[dict] | None = None
    history: list[dict] = []

    for gen in range(n_generations):
        model = PelotonModel(config=config, population=population)
        for _ in range(max_steps):
            if not model.running:
                break
            model.step()

        entry = {
            "generation": gen,
            "n_finished": model.n_finished,
            "mean_finishing_time": _mean_finishing_time(model),
        }
        entry.update(_coeff_stats(model.riders))       # coeffs that raced this generation
        
        # Call evolve, which assigns utilities internally and updates coefficients
        evolve(model.riders, model)
        entry.update(_utility_stats(model.riders))     # performance metrics after race
        history.append(entry)

        # Deep copy so the next generation's agents never alias each other's dicts.
        population = [copy.deepcopy(rider.coeffs) for rider in model.riders]

    return history
