"""Peloton ABM entry point.

    uv run python main.py run       # one race, headless, prints finish order
    uv run python main.py solara    # interactive Solara visualization
    uv run python main.py test      # run the test suite

Parameter sweeps / sensitivity analysis live in their own module:
    uv run python -m peloton.sweep --runs 256 --out results.csv
"""

import argparse
import subprocess
import sys

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Cycling peloton agent-based model")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    run_p = sub.add_parser("run", help="run one race headless and print finish order")
    run_p.add_argument("--max-steps", type=int, default=200)

    sub.add_parser("solara", help="launch the interactive Solara visualization")
    sub.add_parser("test", help="run the test suite (pytest)")

    args = parser.parse_args()

    match args.command:
        case "run":
            run_headless(args.max_steps)
        case "solara":
            # solara is a server: this blocks until the user stops it (Ctrl-C).
            subprocess.run(["solara", "run", "run_app.py"], check=True)
        case "test":
            sys.exit(subprocess.run(["pytest"]).returncode)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
