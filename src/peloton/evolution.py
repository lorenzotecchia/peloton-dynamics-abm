"""Across-race learning of the strategy coefficients.

A single race is one model run. Learning happens *between* races by a simple
genetic algorithm:

  1. score each rider (``_assign_utilities`` — exponential decay by finishing
     position, summed within a team);
  2. average each rider's score over ``evo_replicates`` races so fitness reflects
     strategy rather than that race's lucky engine draw / grid slot / RNG;
  3. ``_next_generation`` keeps the top half unchanged (elitism) and refills the
     bottom half with mutated copies of survivors (truncation selection +
     Gaussian mutation).

Truncation + mutation replaced an earlier similarity-weighted imitation rule
that only nudged the worst 20% and never mutated the rest, so it barely moved
the population even when there was a clear fitness gradient.
"""

import copy
import math
import random
import statistics

from peloton.config import PelotonConfig
from peloton.model import PelotonModel


def _assign_utilities(agents, model) -> None:
    """Score each rider, then set its utility to the rider's *team total*.

    Individual points decay exponentially by finishing position (DNF = 0); a
    rider's utility is the sum of its teammates' points, so evolution rewards
    coefficients that help the whole team rather than the lone rider. With one
    rider per team this collapses back to the individual score.
    """
    rank = {uid: pos for pos, (uid, _step) in enumerate(model.finish_order)}
    decay = 0.4  # smaller -> steeper decay

    points = {
        a.unique_id: (2.0 * math.exp(-decay * rank[a.unique_id])
                      if a.unique_id in rank else 0.0)
        for a in agents
    }
    team_points: dict = {}
    for a in agents:
        team_points[a.team_id] = team_points.get(a.team_id, 0.0) + points[a.unique_id]
    for a in agents:
        a.utility = team_points[a.team_id]


def _mutate(coeffs: dict, std: float, rng: random.Random) -> None:
    """Add zero-mean Gaussian noise to every gene, in place."""
    if std <= 0.0:
        return
    for params in coeffs.values():
        for name in params:
            params[name] += rng.gauss(0.0, std)


def _next_generation(coeffs_list, fitness, cfg, rng) -> list[dict]:
    """Truncation selection: top half survive unchanged, bottom half are mutated
    copies of random survivors. Returns the new population (one dict per slot)."""
    n = len(coeffs_list)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: fitness[i], reverse=True)
    survivors = order[: max(1, n // 2)]
    new = [copy.deepcopy(coeffs_list[i]) for i in survivors]  # elitism
    while len(new) < n:
        child = copy.deepcopy(coeffs_list[rng.choice(survivors)])
        _mutate(child, cfg.evo_noise, rng)
        new.append(child)
    return new


def evolve(agents, model) -> None:
    """Score one race and replace the population in place (truncation + mutation).

    Convenience wrapper around ``_assign_utilities`` + ``_next_generation`` for a
    single race; ``run_generations`` does the multi-race fitness averaging.
    """
    _assign_utilities(agents, model)
    new = _next_generation(
        [a.coeffs for a in agents], [a.utility for a in agents], model.config, model.random
    )
    for a, coeffs in zip(agents, new):
        a.coeffs = coeffs


def _coeff_stats(riders) -> dict:
    """Flat ``{key.param_mean, key.param_std}`` across riders, ready for a DataFrame."""
    stats = {}
    for key, params in riders[0].coeffs.items():
        for param in params:
            vals = [r.coeffs[key][param] for r in riders]
            stats[f"{key}.{param}_mean"] = statistics.mean(vals)
            stats[f"{key}.{param}_std"] = statistics.pstdev(vals)
    return stats


def _utility_stats(riders) -> dict:
    """Mean/std/min/max/range of utility across riders after a race."""
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


def run_generations(n_generations: int, max_steps: int, config=None) -> tuple[list[dict], list[dict] | None]:
    """Run ``n_generations`` of (race(s) -> select -> mutate), learning between them.

    Each generation runs ``evo_replicates`` races with the current population
    (different seeds), averages each rider's utility across them, and breeds the
    next population. Returns a per-generation history (coeff mean/std, finish
    times, utility stats) and the final population (one coeff dict per slot).
    """
    cfg = config or PelotonConfig()
    rng = random.Random(cfg.seed)
    reps = max(1, cfg.evo_replicates)

    population: list[dict] | None = None
    history: list[dict] = []

    for gen in range(n_generations):
        fitness: list[float] | None = None
        finish_steps: list[int] = []
        last = None
        for _ in range(reps):
            model = PelotonModel(config=cfg, population=population,
                                 seed=rng.randint(0, 2**31 - 1))
            for _ in range(max_steps):
                if not model.running:
                    break
                model.step()
            _assign_utilities(model.riders, model)
            if fitness is None:
                fitness = [0.0] * len(model.riders)
            for i, rider in enumerate(model.riders):
                fitness[i] += rider.utility
            finish_steps.extend(step for _uid, step in model.finish_order)
            last = model
        assert last is not None and fitness is not None  # reps >= 1

        entry = {
            "generation": gen,
            "n_finished": last.n_finished,
            "mean_finish_step": statistics.mean(finish_steps) if finish_steps else float("nan"),
            "min_finish_step": min(finish_steps) if finish_steps else float("nan"),
        }
        entry.update(_coeff_stats(last.riders))  # coeffs that raced this generation
        for i, rider in enumerate(last.riders):  # expose averaged fitness for stats
            rider.utility = fitness[i] / reps
        entry.update(_utility_stats(last.riders))
        history.append(entry)

        population = _next_generation([r.coeffs for r in last.riders], fitness, cfg, rng)

    return history, population
