"""Peloton ABM entry point.

    uv run python main.py run        # one race, headless, prints finish order
    uv run python main.py learn      # many races in sequence, learning between them
    uv run python main.py solara     # interactive Solara visualization
    uv run python main.py test       # run the test suite

Parameter sweeps / sensitivity analysis live in their own module:
    uv run python -m peloton.sweep --runs 256 --out results.csv
"""

import argparse
import subprocess
import sys

from peloton.config import PelotonConfig
from peloton.evolution import run_generations
from peloton.model import PelotonModel


def run_headless(max_steps: int) -> None:
    model = PelotonModel()
    for _ in range(max_steps):
        if not model.running:
            break
        model.step()

    print(f"Race over after {model.steps} steps; {model.n_finished} riders finished.")
    for rank, (unique_id, step) in enumerate(model.finish_order, start=1):
        print(f"  {rank:>2}. rider {unique_id} (step {step})")


def run_learning(generations: int, max_steps: int, seed: int | None, out: str, teams: int) -> None:
    """Run the across-race learning loop and dump the per-generation trajectory.

    Also write the final population (per-rider coefficient dicts) to
    ``population.json`` so the last trained population can be reloaded for
    visualization with the `solara` command.

    ``teams`` must be > 1: with a single team the team-sum utility is identical
    for every rider, so there is no fitness variance and nothing to select on.
    """
    import pandas as pd
    import json

    cfg = PelotonConfig(seed=seed, n_teams=teams)
    history, population = run_generations(generations, max_steps, cfg)
    pd.DataFrame(history).to_csv(out, index=False)
    print(f"Ran {generations} generations; wrote coefficient trajectory to {out}.")

    if population is not None:
        with open("population.json", "w") as fh:
            json.dump(population, fh, indent=2)
        print("Wrote final population to population.json (load it via the solara command)")

    first, last = history[0], history[-1]
    key = "effort.delta_energy"
    print(f"  {key} mean: {first[key + '_mean']:.3f} -> {last[key + '_mean']:.3f}"
          f"  (std {first[key + '_std']:.3f} -> {last[key + '_std']:.3f})")

    # Did riders learn to finish faster? Mean steps-to-finish + how many finished.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = [h["generation"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(gens, [h["mean_finish_step"] for h in history], "b-o", ms=3, label="mean")
    ax1.plot(gens, [h["min_finish_step"] for h in history], "c--", label="best")
    ax1.set_xlabel("generation")
    ax1.set_ylabel("steps to finish", color="b")
    ax2 = ax1.twinx()
    ax2.plot(gens, [h["n_finished"] for h in history], "r:", label="riders finished")
    ax2.set_ylabel("riders finished", color="r")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("learning_finish_steps.png", dpi=120)
    print("Wrote learning_finish_steps.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cycling peloton agent-based model")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    run_p = sub.add_parser("run", help="run one race headless and print finish order")
    run_p.add_argument("--max-steps", type=int, default=200)

    learn_p = sub.add_parser("learn", help="run many races in sequence, learning between them")
    learn_p.add_argument("--generations", type=int, default=100)
    learn_p.add_argument("--max-steps", type=int, default=400)
    learn_p.add_argument("--seed", type=int, default=None)
    learn_p.add_argument("--out", default="learning.csv")
    learn_p.add_argument("--teams", type=int, default=50,
                         help="number of teams (>1 needed for fitness variance; =n_agents -> per-rider)")

    sub.add_parser("solara", help="launch the interactive Solara visualization")
    sub.add_parser("test", help="run the test suite (pytest)")

    args = parser.parse_args()

    match args.command:
        case "run":
            run_headless(args.max_steps)
        case "learn":
            run_learning(args.generations, args.max_steps, args.seed, args.out, args.teams)
        case "solara":
            # solara is a server: this blocks until the user stops it (Ctrl-C).
            subprocess.run(["solara", "run", "run_app.py"], check=True)
        case "test":
            sys.exit(subprocess.run(["pytest"]).returncode)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
