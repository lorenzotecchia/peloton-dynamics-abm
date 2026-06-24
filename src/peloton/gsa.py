"""GSA evaluation: run the full evolutionary learning loop and return final-generation metrics.

Mirrors ``main.py learn``: run ``n_generations`` races in sequence with
strategy learning between them (via ``evolution.evolve``), then extract
emergence metrics from the **last** generation.  The final generation
captures the state after evolutionary pressure has acted — that is what
we care about for global sensitivity analysis.

Three metrics capture emergent behaviour:
  n_finished        — agents that crossed the finish line in the last race
  sum_finish_time   — sum of finishing times (seconds) across all finishers
  sum_stamina_spent — sum of W' consumed (J) across ALL agents
"""

import copy
from dataclasses import asdict

from peloton.config import PelotonConfig
from peloton.evolution import evolve
from peloton.model import PelotonModel

METRICS = ["n_finished", "sum_finish_time", "sum_stamina_spent"]

# PelotonConfig fields that must be cast to int when sampled as floats by SALib.
_INT_FIELDS = frozenset({"n_agents", "n_teams", "breakaway_cooldown_steps"})


def run_one(
    params: dict,
    n_generations: int = 200,
    max_steps: int = 10_000,
    seed: int = 42,
) -> dict:
    """Run the full learning loop and return metrics from the final generation.

    Matches ``main.py learn --generations N --max-steps M``.
    All values in ``params`` override PelotonConfig defaults; integer fields
    are rounded automatically so SALib can treat them as continuous.
    """
    base = asdict(PelotonConfig())
    for k, v in params.items():
        base[k] = int(round(v)) if k in _INT_FIELDS else float(v)
    base["seed"] = seed
    config = PelotonConfig(**base)

    population: list[dict] | None = None
    model: PelotonModel | None = None

    for _ in range(n_generations):
        model = PelotonModel(config=config, population=population)
        for _step in range(max_steps):
            if not model.running:
                break
            model.step()
        evolve(model.riders, model)
        population = [copy.deepcopy(rider.coeffs) for rider in model.riders]

    assert model is not None  # n_generations >= 1 guaranteed by callers
    sum_finish_time = sum(step * config.dt for _, step in model.finish_order)
    sum_stamina_spent = sum(a.w_full - a.w_prime for a in model.riders)

    return {
        "n_finished": float(model.n_finished),
        "sum_finish_time": sum_finish_time,
        "sum_stamina_spent": sum_stamina_spent,
    }
