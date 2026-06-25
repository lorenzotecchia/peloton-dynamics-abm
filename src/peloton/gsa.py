"""Global sensitivity analysis of the peloton model (SALib).

Morris (cheap screening) and Sobol (variance-based first/total-order indices).
Each sample runs the across-race *evolution* loop (so utility_decay/lambda bites)
for several seed replicates; the final generation's emergent metrics are averaged,
so simulation stochasticity doesn't leak into the indices.

    # screen first, then the gold-standard decomposition
    python -m peloton.gsa --method both --samples 512 --replicates 5 \
        --generations 30 --max-steps 1000 --processes $SLURM_CPUS_PER_TASK

Writes one CSV per method to <out-dir>/gsa_<method>.csv (long form:
metric, param, index columns). Edit PROBLEM below to change which knobs are
varied or their ranges.

Parameters under analysis (default -> range; see PROBLEM_* for the live values):
    recovery_rate         0.1  -> [0.05, 0.20]   W' recovery rate below CP
    k_s                   0.8  -> [0.70, 1.00]   pack-speed coefficient (Martins 2013)
    breakaway_speed_frac  0.95 -> [0.85, 1.00]   solo speed as fraction of threshold
    utility_decay (lambda)0.4  -> [0.10, 1.00]   steepness of position->utility decay
    n_agents (Sobol only) 96   -> [24, 192]      pack size, int (2-16 riders/team)
Targets (race-mean of the final generation): MeanStamina, NumGroups,
Breakaways, MeanExposure. Fixed scenario knobs (road_length, dt, group_radius)
are not SA-varied; set them with --road-length / --dt / --group-radius.
"""

import argparse
import os
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.analyze.morris import analyze as morris_analyze
from SALib.sample import sobol as sobol_sample
from SALib.sample.morris import sample as morris_sample

from peloton.config import PelotonConfig
from peloton.evolution import run_generations

# The knobs under analysis and their ranges (around PelotonConfig defaults).
# Morris screens the 4 continuous physiology/strategy knobs; Sobol additionally
# decomposes pack size (n_agents), an integer knob rounded from the float sample.
PROBLEM_MORRIS = {
    "num_vars": 4,
    "names": ["recovery_rate", "breakaway_speed_frac", "utility_decay", "k_s"],
    "bounds": [
        [0.05, 0.20],    # recovery_rate        (default 0.1)
        [0.85, 1.00],    # breakaway_speed_frac (default 0.95)
        [0.10, 1.00],    # utility_decay/lambda (default 0.4)
        [0.70, 1.00],    # k_s                  (default 0.8)
    ],
}

PROBLEM_SOBOL = {
    "num_vars": PROBLEM_MORRIS["num_vars"] + 1,
    "names": [*PROBLEM_MORRIS["names"], "n_agents"],
    "bounds": [
        *PROBLEM_MORRIS["bounds"],
        [24, 192],       # n_agents (default 96; 2-16 riders/team at n_teams=12)
    ],
}

PROBLEMS = {"morris": PROBLEM_MORRIS, "sobol": PROBLEM_SOBOL}

# Knobs that count/index things: round the float sample to int before it reaches
# PelotonConfig (replace() would otherwise leave e.g. range(144.0) -> TypeError).
INT_PARAMS = {"n_agents"}

# Emergent metrics, race-averaged over the final generation's race.
METRICS = ["MeanStamina", "NumGroups", "Breakaways", "MeanExposure"]

_MORRIS_LEVELS = 4


def _evaluate(args: tuple) -> np.ndarray:
    """Run one sample's evolution for `replicates` seeds; return seed-mean final metrics."""
    row, names, generations, max_steps, replicates, base = args
    overrides = {n: (int(round(v)) if n in INT_PARAMS else float(v))
                 for n, v in zip(names, row)}
    out = np.empty((replicates, len(METRICS)))
    for s in range(replicates):
        # `base` holds the fixed scenario (road_length, dt, group_radius, ...);
        # `overrides` are the SA knobs this sample varies. SA knobs win on overlap.
        cfg = replace(PelotonConfig(seed=s), **base, **overrides)
        last = run_generations(generations, max_steps, cfg)[-1]
        out[s] = [last[m] for m in METRICS]
    return out.mean(axis=0)


def _simulate(X, names, generations, max_steps, replicates, processes, y_path, base) -> np.ndarray:
    """Evaluate every sample row in parallel -> Y of shape (n_samples, n_metrics).

    Checkpoints each completed sample to ``y_path`` as it finishes, so a killed
    or timed-out job resumes from disk instead of recomputing. Append-only +
    flush, so the worst case loses at most the one sample in flight.
    """
    done = []
    if y_path.exists():
        done = list(np.loadtxt(y_path, delimiter=",", ndmin=2))
    start = len(done)
    if start:
        print(f"  resuming from checkpoint: {start}/{len(X)} samples done", flush=True)

    tasks = [(row, names, generations, max_steps, replicates, base) for row in X[start:]]
    print(f"  evaluating {len(tasks)} samples x {replicates} reps x {generations} gens ...",
          flush=True)
    with Pool(processes) as pool, open(y_path, "a") as fh:
        for row in pool.imap(_evaluate, tasks):   # ordered: row k appended after k-1
            fh.write(",".join(repr(float(v)) for v in row) + "\n")
            fh.flush()
            done.append(row)
    return np.array(done)


def run_method(method, n, generations, max_steps, replicates, processes,
               out_dir, base=None) -> pd.DataFrame:
    """Sample, simulate, and estimate indices for one method. Returns long-form df."""
    problem = PROBLEMS[method]  # Morris and Sobol vary different knob sets.
    # Pin the design to disk: Morris sampling isn't reproducible across runs, so a
    # resume must reuse the exact X that the checkpointed Y rows were computed for.
    x_path = Path(out_dir) / f".gsa_{method}_X.npy"
    if x_path.exists():
        X = np.load(x_path)
    else:
        if method == "morris":
            X = morris_sample(problem, n, num_levels=_MORRIS_LEVELS)
        elif method == "sobol":
            X = sobol_sample.sample(problem, n, calc_second_order=False)
        else:
            raise ValueError(f"unknown method {method!r}")
        np.save(x_path, X)

    y_path = Path(out_dir) / f".gsa_{method}_Y.csv"
    Y = _simulate(X, problem["names"], generations, max_steps, replicates, processes,
                  y_path, base or {})

    rows = []
    for j, metric in enumerate(METRICS):
        if method == "morris":
            Si = morris_analyze(problem, X, Y[:, j], num_levels=_MORRIS_LEVELS)
            cols = {"mu_star": Si["mu_star"], "mu_star_conf": Si["mu_star_conf"],
                    "sigma": Si["sigma"]}
        else:
            Si = sobol_analyze.analyze(problem, Y[:, j], calc_second_order=False)
            cols = {"S1": Si["S1"], "S1_conf": Si["S1_conf"],
                    "ST": Si["ST"], "ST_conf": Si["ST_conf"]}
        for i, param in enumerate(problem["names"]):
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
    p.add_argument("--generations", type=int, default=30,
                   help="races per evolution run (lambda only bites across generations)")
    p.add_argument("--max-steps", type=int, default=1000,
                   help="steps per race; must be high enough for riders to finish "
                        "(~600+) or utility is degenerate and lambda can't bite")
    p.add_argument("--processes", type=int, default=os.cpu_count())
    p.add_argument("--out-dir", default="data")
    # Fixed scenario knobs (not SA-varied): override PelotonConfig for every sample.
    # Defaults None = keep config defaults. For the long-road/coarse-dt experiment
    # pass e.g. --road-length 200000 --dt 60 --group-radius 300 (and bump --max-steps).
    p.add_argument("--road-length", type=float, default=None, help="finish line in m (e.g. 200000)")
    p.add_argument("--dt", type=float, default=None, help="seconds of race per step (e.g. 60)")
    p.add_argument("--group-radius", type=float, default=None,
                   help="longitudinal pack radius in m; scale with v*dt for coarse dt")
    args = p.parse_args()

    base = {k: v for k, v in (("road_length", args.road_length), ("dt", args.dt),
                              ("group_radius", args.group_radius)) if v is not None}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ["morris", "sobol"] if args.method == "both" else [args.method]
    for method in methods:
        print(f"[gsa] {method} (N={args.samples}) base={base or 'defaults'}")
        df = run_method(method, args.samples, args.generations, args.max_steps,
                        args.replicates, args.processes, out_dir, base)
        out = out_dir / f"gsa_{method}.csv"
        df.to_csv(out, index=False)
        print(f"[gsa] wrote {out}")
        # Indices written: drop the checkpoint so a fresh re-run resamples.
        (out_dir / f".gsa_{method}_X.npy").unlink(missing_ok=True)
        (out_dir / f".gsa_{method}_Y.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
