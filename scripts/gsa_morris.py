"""Morris method (elementary effects) sensitivity analysis for the peloton model.

Evaluates elementary effects for ALL float/int parameters of PelotonConfig
against three emergence metrics:
  - n_finished        (agents that crossed the finish line)
  - sum_finish_time   (sum of finishing times in seconds)
  - sum_stamina_spent (sum of W' consumed across all agents, joules)

Outputs per metric:
  morris_<metric>.csv  — mu, mu_star, sigma, mu_star_conf per parameter
  raw.csv              — all sample rows + metric values
  morris_indices.json  — all indices in one JSON

Usage:
    uv run python scripts/gsa_morris.py [--trajectories 20] [--out-dir DIR]
"""

import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

from peloton.gsa import METRICS, run_one

# All PelotonConfig parameters except `seed` (random seed, not a model knob)
# and `dt` (numerical integration step — varies physics non-physically).
# Integer fields (n_agents, n_teams, breakaway_cooldown_steps) are sampled
# as continuous and rounded inside run_one.
PROBLEM: dict = {
    "num_vars": 24,
    "names": [
        # Race geometry
        "road_length", "road_width",
        # Rider count and teams
        "n_agents", "n_teams",
        # Physical footprint (affects spawn spacing only)
        "rider_length", "rider_width",
        # Physiology
        "w_max10_mean", "w_max10_std", "cp_fraction", "recovery_rate",
        # Aerodynamics & rolling resistance
        "k_aero", "c_roll", "ref_speed_frac",
        # Grouping & pack dynamics
        "group_radius", "k_s", "breakaway_speed_frac",
        "breakaway_cooldown_steps",
        # Learning / evolution
        "learning_rate", "evo_noise", "sim_scale",
        "evo_bottom_frac", "evo_top_frac",
        "imitation_mu", "logit_lambda",
    ],
    "bounds": [
        [5_000.0, 20_000.0],   # road_length  (m): 5–20 km
        [5.0,     12.0],       # road_width   (m)
        [40,      200],        # n_agents
        [5,       25],         # n_teams
        [1.0,     2.5],        # rider_length (m)
        [0.4,     0.9],        # rider_width  (m)
        [300.0,   600.0],      # w_max10_mean (W)
        [30.0,    120.0],      # w_max10_std  (W)
        [0.55,    0.85],       # cp_fraction
        [0.02,    0.40],       # recovery_rate
        [0.10,    0.30],       # k_aero
        [1.0,     6.0],        # c_roll
        [0.70,    1.00],       # ref_speed_frac
        [1.0,     6.0],        # group_radius (m)
        [0.70,    1.00],       # k_s  (Martins 2013: 0.7–1.0)
        [0.70,    1.10],       # breakaway_speed_frac
        [5,       20],         # breakaway_cooldown_steps
        [0.01,    0.30],       # learning_rate
        [0.005,   0.10],       # evo_noise
        [0.1,     3.0],        # sim_scale
        [0.05,    0.50],       # evo_bottom_frac
        [0.50,    1.00],       # evo_top_frac
        [0.50,    1.00],       # imitation_mu
        [0.10,    5.00],       # logit_lambda
    ],
}


def _eval_row(args):
    row, n_generations, max_steps, seed = args
    params = dict(zip(PROBLEM["names"], row))
    return run_one(params, n_generations=n_generations, max_steps=max_steps, seed=seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectories", type=int, default=20,
                   help="Morris trajectories N (default 20); "
                        "total samples = N × (k+1)")
    p.add_argument("--levels", type=int, default=4,
                   help="grid levels for Morris sampling (default 4)")
    p.add_argument("--generations", type=int, default=200,
                   help="learning generations per sample (mirrors 'main.py learn')")
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--processes", type=int, default=os.cpu_count())
    p.add_argument("--out-dir", default="results/morris")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    k = PROBLEM["num_vars"]
    total = args.trajectories * (k + 1)
    print(f"[morris] {args.trajectories} trajectories × (k+1)={k+1} = {total} samples")
    print(f"[morris] parameters: {PROBLEM['names']}")

    samples = morris_sample.sample(
        PROBLEM, args.trajectories,
        num_levels=args.levels,
        seed=args.seed,
    )
    print(f"[morris] sample matrix: {samples.shape}")

    with mp.Pool(processes=args.processes) as pool:
        results = pool.map(_eval_row,
                           [(row, args.generations, args.max_steps, args.seed)
                            for row in samples])
    print(f"[morris] evaluated {len(results)} samples")

    raw_df = pd.DataFrame(
        [{**dict(zip(PROBLEM["names"], row)), **res}
         for row, res in zip(samples, results)]
    )
    raw_path = os.path.join(args.out_dir, "raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"[morris] raw results → {raw_path}")

    all_indices: dict = {}
    for metric in METRICS:
        Y = np.array([r[metric] for r in results])
        Si = morris_analyze.analyze(
            PROBLEM, samples, Y,
            num_levels=args.levels,
            print_to_console=False,
        )
        all_indices[metric] = {
            "mu":          Si["mu"].tolist(),
            "mu_star":     Si["mu_star"].tolist(),
            "sigma":       Si["sigma"].tolist(),
            "mu_star_conf": Si["mu_star_conf"].tolist(),
        }
        df = pd.DataFrame({
            "parameter":   PROBLEM["names"],
            "mu":          Si["mu"],
            "mu_star":     Si["mu_star"],
            "sigma":       Si["sigma"],
            "mu_star_conf": Si["mu_star_conf"],
        }).sort_values("mu_star", ascending=False)
        out = os.path.join(args.out_dir, f"morris_{metric}.csv")
        df.to_csv(out, index=False)
        print(f"[morris] {metric} → {out}")

    json_path = os.path.join(args.out_dir, "morris_indices.json")
    with open(json_path, "w") as f:
        json.dump(all_indices, f, indent=2)
    print(f"[morris] all indices → {json_path}")
    print("[morris] done.")


if __name__ == "__main__":
    main()
