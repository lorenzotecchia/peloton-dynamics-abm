"""Across-race learning via discrete-time replicator dynamics.

Each agent carries a mixed strategy σ = (p_C, p_D, p_B).  During the race,
EGT payoffs accumulate from pairwise group interactions (strategy.group_payoffs).
Between races, σ is updated by the standard discrete-time replicator rule:

    Δσ_s = η · σ_s · (f_s − f̄) / |f̄|

where f_s is the population mean payoff of strategy s this race, f̄ = σ · f is
the agent's expected payoff, and η = learning_rate.  A mutation term (evo_noise)
blends each σ toward the uniform distribution to prevent strategy extinction.

Distinct roles (cooperator, defector, breakaway specialist) emerge when the
payoff matrix creates selective pressure for strategy differentiation.
"""

import copy
import statistics

from peloton import strategy as strategy_module
from peloton.model import PelotonModel
from peloton.strategy import N_STRATEGIES, STRATEGY_NAMES, Strategy


def evolve(agents, model) -> None:
    """Update every agent's mixed strategy in-place via replicator dynamics.

    Steps:
      1. Set each agent's utility to their accumulated EGT payoff this race.
      2. Compute population mean fitness f_s for each pure strategy.
      3. Apply the replicator update to σ_i, then add mutation.
      4. Draw a new pure strategy from the updated σ_i for the next race.
    """
    cfg = model.config
    rng = model.random
    n = len(agents)
    if n == 0:
        return

    # Step 1 — record actual payoffs as utility so _utility_stats can read them.
    for a in agents:
        a.utility = a.egt_payoff

    # Step 2 — population fitness per pure strategy (mean payoff across agents
    # who played that strategy this race; strategies with no players default to 0).
    payoff_lists: list[list[float]] = [[] for _ in range(N_STRATEGIES)]
    for a in agents:
        payoff_lists[int(a.strategy)].append(a.egt_payoff)

    f: list[float] = [
        sum(vals) / len(vals) if vals else 0.0
        for vals in payoff_lists
    ]

    # Step 3 — replicator update + mutation for each agent.
    for a in agents:
        sigma = list(a.mixed_strategy)

        # Expected fitness under the current mixed strategy: f̄ = σ · f
        f_bar = sum(sigma[s] * f[s] for s in range(N_STRATEGIES))

        # Additive replicator: Δσ_s = η · σ_s · (f_s − f̄) / |f̄|
        # The |f̄| denominator normalises the step size so η is scale-invariant.
        denom = max(abs(f_bar), 1e-6)
        new_sigma = [
            sigma[s] + cfg.learning_rate * sigma[s] * (f[s] - f_bar) / denom
            for s in range(N_STRATEGIES)
        ]

        # Mutation: blend toward uniform to prevent strategy extinction.
        mu = cfg.evo_noise
        uniform = 1.0 / N_STRATEGIES
        new_sigma = [
            (1.0 - mu) * max(v, 0.0) + mu * uniform
            for v in new_sigma
        ]

        # Normalise to a valid probability vector.
        total = sum(new_sigma)
        a.mixed_strategy = [v / total for v in new_sigma]

        # Step 4 — sample pure strategy for the next race.
        a.strategy = strategy_module.sample_strategy(a.mixed_strategy, rng)


def _strategy_stats(riders) -> dict:
    """Population statistics of mixed strategies and pure-strategy frequencies.

    Returns flat keys ``{C,D,B}_mean``, ``{C,D,B}_std``, ``{C,D,B}_freq``
    ready to write into a DataFrame row and plot with plot_learning.py.
    """
    stats: dict = {}
    n = len(riders)
    for i, name in enumerate(STRATEGY_NAMES):
        probs = [r.mixed_strategy[i] for r in riders]
        stats[f"{name}_mean"] = statistics.mean(probs)
        stats[f"{name}_std"]  = statistics.pstdev(probs)
        stats[f"{name}_freq"] = sum(1 for r in riders if int(r.strategy) == i) / n
    return stats


def _utility_stats(riders) -> dict:
    """Distribution of EGT payoffs (= utilities) across all riders."""
    utilities = [r.utility for r in riders]
    if not utilities:
        return {}
    return {
        "utility_mean":  statistics.mean(utilities),
        "utility_std":   statistics.pstdev(utilities) if len(utilities) > 1 else 0.0,
        "utility_min":   min(utilities),
        "utility_max":   max(utilities),
    }


def run_generations(n_generations: int, max_steps: int, config=None) -> list[dict]:
    """Run ``n_generations`` races in sequence, evolving mixed strategies between them.

    Returns a per-generation history containing:
    - ``generation``, ``n_finished``
    - ``{C,D,B}_mean``, ``{C,D,B}_std``, ``{C,D,B}_freq`` — mixed strategy stats
    - ``utility_{mean,std,min,max}`` — EGT payoff distribution this race

    Generation 0 is the initial population (uniform σ), before any learning.
    """
    population: list[list[float]] | None = None
    history: list[dict] = []

    for gen in range(n_generations):
        model = PelotonModel(config=config, population=population)
        for _ in range(max_steps):
            if not model.running:
                break
            model.step()

        entry: dict = {"generation": gen, "n_finished": model.n_finished}
        entry.update(_strategy_stats(model.riders))

        evolve(model.riders, model)
        entry.update(_utility_stats(model.riders))
        history.append(entry)

        population = [list(rider.mixed_strategy) for rider in model.riders]

    return history
