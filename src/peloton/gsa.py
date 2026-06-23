"""Global sensitivity analysis of the peloton model (SALib).

Morris (cheap screening) and Sobol (variance-based first/total-order indices).
Each parameter sample is run for several seed replicates and the emergent
metrics are averaged, so simulation stochasticity doesn't leak into the indices.

    # screen first, then the gold-standard decomposition
    python -m peloton.gsa --method both --samples 512 --replicates 5 \
        --max-steps 1000 --processes $SLURM_CPUS_PER_TASK

Writes one CSV per method to <out-dir>/gsa_<method>.csv (long form:
metric, param, index columns). Edit PROBLEM below to change which knobs are
varied or their ranges.
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.analyze.morris import analyze as morris_analyze
from SALib.sample import sobol as sobol_sample
from SALib.sample.morris import sample as morris_sample

from peloton.model import PelotonModel

# The knobs under analysis and their ranges (around PelotonConfig defaults).
PROBLEM = {
    "num_vars": 4,
    "names": ["recovery_rate", "breakaway_speed_frac", "w_max10_mean", "k_s"],
    "bounds": [
        [0.05, 0.20],    # recovery_rate   (default 0.1)
        [0.85, 1.00],    # breakaway_speed_frac (default 0.95)
        [350.0, 450.0],  # w_max10_mean    (default 400)
        [0.70, 1.00],    # k_s             (default 0.8)
    ],
}

# Emergent metrics read off the model's datacollector at race end.
METRICS = ["MeanStamina", "NumGroups", "Breakaways", "MeanExposure"]

_MORRIS_LEVELS = 4


def _evaluate(args: tuple) -> np.ndarray:
    """Run one parameter sample for `replicates` seeds; return seed-mean metrics."""
    row, max_steps, replicates = args
    overrides = dict(zip(PROBLEM["names"], (float(v) for v in row)))
    out = np.empty((replicates, len(METRICS)))
    for s in range(replicates):
        model = PelotonModel(seed=s, **overrides)
        for _ in range(max_steps):
            if not model.running:
                break
            model.step()
        last = model.datacollector.get_model_vars_dataframe().iloc[-1]
        out[s] = [last[m] for m in METRICS]
    return out.mean(axis=0)


def _simulate(X: np.ndarray, max_steps: int, replicates: int, processes) -> np.ndarray:
    """Evaluate every sample row in parallel -> Y of shape (n_samples, n_metrics)."""
    tasks = [(row, max_steps, replicates) for row in X]
    print(f"  evaluating {len(tasks)} samples x {replicates} replicates ...", flush=True)
    with Pool(processes) as pool:
        return np.array(pool.map(_evaluate, tasks))


def run_method(method: str, n: int, max_steps: int, replicates: int, processes) -> pd.DataFrame:
    """Sample, simulate, and estimate indices for one method. Returns long-form df."""
    if method == "morris":
        X = morris_sample(PROBLEM, n, num_levels=_MORRIS_LEVELS)
    elif method == "sobol":
        X = sobol_sample.sample(PROBLEM, n, calc_second_order=False)
    else:
        raise ValueError(f"unknown method {method!r}")

    Y = _simulate(X, max_steps, replicates, processes)

    rows = []
    for j, metric in enumerate(METRICS):
        if method == "morris":
            Si = morris_analyze(PROBLEM, X, Y[:, j], num_levels=_MORRIS_LEVELS)
            cols = {"mu_star": Si["mu_star"], "mu_star_conf": Si["mu_star_conf"],
                    "sigma": Si["sigma"]}
        else:
            Si = sobol_analyze.analyze(PROBLEM, Y[:, j], calc_second_order=False)
            cols = {"S1": Si["S1"], "S1_conf": Si["S1_conf"],
                    "ST": Si["ST"], "ST_conf": Si["ST_conf"]}
        for i, param in enumerate(PROBLEM["names"]):
            rows.append({"metric": metric, "param": param,
                         **{k: v[i] for k, v in cols.items()}})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["morris", "sobol", "both"], default="both")
    p.add_argument("--samples", type=int, default=512,
                   help="SALib N: Morris trajectories (~N*(D+1) runs) / Sobol base "
                        "(~N*(D+2) runs); use a power of 2 for Sobol")
    p.add_argument("--replicates", type=int, default=5,
                   help="seed replicates averaged per sample (tames ABM noise)")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--processes", type=int, default=os.cpu_count())
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ["morris", "sobol"] if args.method == "both" else [args.method]
    for method in methods:
        print(f"[gsa] {method} (N={args.samples})")
        df = run_method(method, args.samples, args.max_steps, args.replicates, args.processes)
        out = out_dir / f"gsa_{method}.csv"
        df.to_csv(out, index=False)
        print(f"[gsa] wrote {out}")


if __name__ == "__main__":
    main()
