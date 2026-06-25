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

Parameters under analysis (see PROBLEM_MORRIS / PROBLEM_SOBOL for live values):
    Morris screens *every* live knob (18 of them) to find what matters: the
        physiology block (w_max10_mean/std, cp_fraction, recovery_rate, k_aero,
        c_roll, ref_speed_frac), grouping/breakaway (k_s, breakaway_speed_frac,
        breakaway_cooldown_steps), evolution (utility_decay, evo_noise, sim_scale,
        evo_bottom_frac, evo_top_frac, imitation_mu) and structure (n_agents,
        n_teams). Dead/scenario/viz knobs are excluded (see PROBLEM_MORRIS).
    Sobol decomposes a focused subset: recovery_rate, breakaway_speed_frac,
        utility_decay, k_s, n_agents.
Integer knobs (n_agents, n_teams, breakaway_cooldown_steps) are rounded from the
float sample before reaching PelotonConfig.
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
#
# Morris (cheap elementary-effects screening) varies *every* live model knob, so
# we can see which of the full parameter set actually move the emergent metrics.
# Omitted on purpose: learning_rate/elite_fraction (dead — unused by the model);
# road_length/dt/group_radius (fixed scenario knobs, set per experiment via
# --road-length/--dt/--group-radius, not screened); road_width/rider_length/
# rider_width (viz-only geometry — riders are points, no dynamics) and seed (RNG).
_MORRIS_KNOBS = [
    # name                       lo      hi      # default
    ("w_max10_mean",            350.0,  550.0),  # 450  mean 10-min max power (W)
    ("w_max10_std",              34.0,  102.0),  # 68   power spread across the field
    ("cp_fraction",               0.60,   0.80), # 0.7  critical-power fraction
    ("recovery_rate",             0.05,   0.20), # 0.1  W' recovery below CP
    ("k_aero",                    0.60,   1.20), # 0.9  aerodynamic coefficient
    ("c_roll",                    2.0,    5.0),  # 3.6  rolling-resistance coefficient
    ("ref_speed_frac",            0.80,   0.95), # 0.9  v_hat fraction for stamina init
    ("k_s",                       0.70,   1.00), # 0.9  pack-speed coefficient
    ("breakaway_speed_frac",      0.85,   1.00), # 0.9  solo speed / threshold
    ("breakaway_cooldown_steps",  2,     20),    # 10   breaker re-merge cooldown (int)
    ("utility_decay",             0.10,   1.00), # 0.8  lambda: position->utility decay
    ("evo_noise",                 0.0,    0.10), # 0.02 per-generation Gaussian noise
    ("sim_scale",                 0.50,   2.00), # 1.0  rider-similarity bandwidth
    ("evo_bottom_frac",           0.10,   0.40), # 0.2  fraction of worst updated
    ("evo_top_frac",              0.05,   0.30), # 0.1  fraction of top used as donors
    ("imitation_mu",              0.30,   1.00), # 0.75 blend toward donor (1=copy)
    ("n_agents",                 24,    192),    # 96   pack size (int, 2-16/team)
    ("n_teams",                   4,     24),    # 12   number of teams (int)
]
PROBLEM_MORRIS = {
    "num_vars": len(_MORRIS_KNOBS),
    "names": [name for name, _lo, _hi in _MORRIS_KNOBS],
    "bounds": [[lo, hi] for _name, lo, hi in _MORRIS_KNOBS],
}

# Sobol (variance-based, expensive) decomposes a focused subset — keep D small.
# n_agents is integer, rounded from the float sample.
PROBLEM_SOBOL = {
    "num_vars": 5,
    "names": ["recovery_rate", "breakaway_speed_frac", "utility_decay", "k_s", "n_agents"],
    "bounds": [
        [0.05, 0.20],    # recovery_rate        (default 0.1)
        [0.85, 1.00],    # breakaway_speed_frac (default 0.9)
        [0.10, 1.00],    # utility_decay/lambda (default 0.8)
        [0.70, 1.00],    # k_s                  (default 0.9)
        [24, 192],       # n_agents (default 96; 2-16 riders/team at n_teams=12)
    ],
}

PROBLEMS = {"morris": PROBLEM_MORRIS, "sobol": PROBLEM_SOBOL}

# Knobs that count/index things: round the float sample to int before it reaches
# PelotonConfig (replace() would otherwise leave e.g. range(144.0) -> TypeError).
INT_PARAMS = {"n_agents", "n_teams", "breakaway_cooldown_steps"}

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
