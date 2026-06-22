"""Run many peloton simulations in parallel and dump results to CSV.

Thin wrapper around ``mesa.batch_run`` (multiprocessing pool, per-run seeding,
datacollector harvesting). Designed for one fat Snellius CPU node.

    python -m peloton.sweep --runs 256 --max-steps 500 --out results.csv

``PARAMETERS`` is empty for now, so runs are stochastic replicates of the
default scenario, differing only by seed. To sweep parameters later, add
slider-name keys with list values (Cartesian product) — that is the only
change needed for a sensitivity analysis.
"""

import argparse
import os

import pandas as pd
from mesa import batch_run

from peloton.model import PelotonModel

# ponytail: empty = pure replicates. SA later fills this with utility knobs, e.g.
# {"k_s": [0.8, 0.9, 1.0], "draft_coefficient": [0.5, 0.62, 0.75], "evo_noise": [0.05, 0.1]}.
PARAMETERS: dict = {}


def run(runs: int, max_steps: int, processes: int | None) -> pd.DataFrame:
    records = batch_run(
        PelotonModel,
        PARAMETERS,
        rng=range(runs),  # reproducible, distinct seed per replicate
        number_processes=processes,
        max_steps=max_steps,
        data_collection_period=-1,  # final step only
        display_progress=True,
    )
    return pd.DataFrame(records)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=64, help="number of replicate simulations")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--processes", type=int, default=os.cpu_count(),
                   help="worker processes (set to $SLURM_CPUS_PER_TASK on Slurm)")
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()

    df = run(args.runs, args.max_steps, args.processes)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} runs to {args.out}")


if __name__ == "__main__":
    main()
