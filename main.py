"""Peloton ABM entry point.

    uv run python main.py run        # one race, headless, prints finish order
    uv run python main.py dump       # one race, dump full per-step per-agent data for analysis
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


def run_learning(generations: int, max_steps: int, seed: int | None, out: str) -> None:
    """Run the across-race learning loop and dump the per-generation trajectory."""
    import pandas as pd

    from pathlib import Path

    history = run_generations(generations, max_steps, PelotonConfig(seed=seed))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out, index=False)
    print(f"Ran {generations} generations; wrote coefficient trajectory to {out}.")
    first, last = history[0], history[-1]
    print(
        f"  coop.delta mean: {first['coop.delta_mean']:.3f} -> {last['coop.delta_mean']:.3f}"
        f"  (std {first['coop.delta_std']:.3f} -> {last['coop.delta_std']:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cycling peloton agent-based model")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    run_p = sub.add_parser("run", help="run one race headless and print finish order")
    run_p.add_argument("--max-steps", type=int, default=200)

    dump_p = sub.add_parser(
        "dump", help="run one race and dump full per-agent data for analysis"
    )
    dump_p.add_argument("--max-steps", type=int, default=2000)
    dump_p.add_argument("--seed", type=int, default=None)
    dump_p.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: analysis_output/<timestamp>)",
    )
    dump_p.add_argument(
        "--parquet",
        action="store_true",
        help="write Parquet instead of CSV (needs pyarrow)",
    )

    learn_p = sub.add_parser(
        "learn", help="run many races in sequence, learning between them"
    )
    learn_p.add_argument("--generations", type=int, default=100)
    learn_p.add_argument("--max-steps", type=int, default=2000)
    learn_p.add_argument("--seed", type=int, default=None)
    learn_p.add_argument("--out", default="data/learning.csv")

    sub.add_parser("solara", help="launch the interactive Solara visualization")
    sub.add_parser("test", help="run the test suite (pytest)")

    args = parser.parse_args()

    match args.command:
        case "run":
            run_headless(args.max_steps)
        case "dump":
            from datetime import datetime
            from peloton.recorder import dump_run

            out_dir = args.out_dir or f"analysis_output/{datetime.now():%Y%m%d-%H%M%S}"
            dump_run(
                PelotonConfig(seed=args.seed), args.max_steps, out_dir, args.parquet
            )
        case "learn":
            run_learning(args.generations, args.max_steps, args.seed, args.out)
        case "solara":
            # solara is a server: this blocks until the user stops it (Ctrl-C).
            subprocess.run(["solara", "run", "run_app.py"], check=True)
        case "test":
            sys.exit(subprocess.run(["pytest"]).returncode)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
