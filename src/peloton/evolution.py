"""Across-race learning of the strategy coefficients.

A single race is one model run. Learning happens *between* races: score each
rider's outcome, then nudge their coefficients toward better-performing, similar
riders. ``run_generations`` is the outer loop carrying a population of
coefficients across races; ``evolve`` is the update rule (Francesca's notes):

    delta(theta_i) = eta * sum_j max(0, U_j - U_i) (theta_j - theta_i) sim(i, j) + noise

so a rider imitates peers who did better *and* race like them (similar engine),
which is how distinct roles can emerge from a homogeneous start.
"""

import copy
import math
import statistics

from peloton.model import PelotonModel


def _assign_utilities(agents, model) -> None:
    """Utility = finishing position (winner highest); DNF scores 0 (worst)."""
    rank = {uid: pos for pos, (uid, _step) in enumerate(model.finish_order)}
    n = len(agents)
    for a in agents:
        a.utility = (n - rank[a.unique_id]) if a.unique_id in rank else 0.0


def _similarity(a, b, cfg) -> float:
    """Gaussian on the engine difference. s_m is a monotone function of w_max10,
    so w_max10 alone captures 'races like me' — no need to weight both."""
    z = (a.w_max10 - b.w_max10) / (cfg.sim_scale * cfg.w_max10_std)
    return math.exp(-0.5 * z * z)


def evolve(agents, model) -> None:
    """Update every agent's coefficients in place from this race's outcomes."""
    cfg = model.config
    rng = model.random
    _assign_utilities(agents, model)

    # Read coefficients from a frozen snapshot so updates don't feed back mid-pass.
    snapshot = [copy.deepcopy(a.coeffs) for a in agents]
    for i, a in enumerate(agents):
        for key, params in a.coeffs.items():           # coop / leave / follow
            for param in params:
                delta = 0.0
                for j, b in enumerate(agents):
                    advantage = b.utility - a.utility
                    if j == i or advantage <= 0.0:
                        continue
                    pull = snapshot[j][key][param] - snapshot[i][key][param]
                    delta += advantage * pull * _similarity(a, b, cfg)
                noise = rng.gauss(0.0, cfg.evo_noise)
                a.coeffs[key][param] = snapshot[i][key][param] + cfg.learning_rate * delta + noise


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


def run_generations(n_generations: int, max_steps: int, config=None) -> list[dict]:
    """Run ``n_generations`` races in sequence, learning between them.

    Coefficients persist across races in ``population`` (one dict per spawn
    slot); each race is seeded from it and ``evolve`` writes the updates back.
    Returns a per-generation history: ``n_finished`` plus the mean/std of every
    coefficient *as it raced that generation* (so generation 0 is the initial
    population, before any learning).
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
        entry.update(_coeff_stats(model.riders))       # coeffs that raced this generation
        history.append(entry)

        evolve(model.riders, model)
        # Deep copy so the next generation's agents never alias each other's dicts.
        population = [copy.deepcopy(rider.coeffs) for rider in model.riders]

    return history
