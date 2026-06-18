"""STUB: across-race learning of game-theory coefficients.

A single race is one model run. Learning happens *between* races: after a race
finishes, score each rider's outcome, then nudge their coefficients toward the
better performers. ``run_generations`` is the outer loop that carries a
population of coefficients forward across races; ``evolve`` is the update rule.

Both are inert for now. ``evolve`` is a no-op, so coefficients never change and
every generation is an identical replicate — but the wiring (population ->
model -> race -> score -> evolve -> population) exists for the real rule to
drop into.
"""

from peloton.model import PelotonModel


def evolve(agents, model) -> None:
    """Update each agent's ``coeffs`` from race outcomes. STUB: does nothing.

    Real version reads ``model.finish_order`` / ``agent.utility`` and shifts
    each rider's coefficients toward better-performing, similar riders
    (learning rate ``model.config.learning_rate``).
    """
    return None


def run_generations(n_generations: int, max_steps: int, config=None) -> list[dict]:
    """Run ``n_generations`` races in sequence, learning between them.

    Coefficients persist across races in ``population`` (one dict per spawn
    slot); each race is seeded from it and ``evolve`` writes the updates back.
    Returns a per-generation history of summary stats.
    """
    population: list[dict] | None = None
    history: list[dict] = []

    for gen in range(n_generations):
        model = PelotonModel(config=config, population=population)
        for _ in range(max_steps):
            if not model.running:
                break
            model.step()

        evolve(model.riders, model)

        # Carry this generation's coefficients into the next race.
        population = [dict(rider.coeffs) for rider in model.riders]
        history.append({"generation": gen, "n_finished": model.n_finished})

    return history
