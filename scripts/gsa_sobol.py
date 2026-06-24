"""Sobol global sensitivity analysis (S1, S2, ST) for the peloton model.

Parameters under study: recovery_rate, breakaway_speed_frac, logit_lambda, k_s
Base sample size: N=128
  - S1 / ST only:  N × (k+2) = 128 × 6 = 768 model runs
  - S1 / S2 / ST:  N × (2k+2) = 128 × 10 = 1280 model runs  ← used here

Outputs per metric:
  sobol_S1_ST_<metric>.csv  — first-order and total-order indices
  sobol_S2_<metric>.csv     — second-order indices (upper triangle)
  raw.csv                   — all sample rows + metric values
  sobol_indices.json        — all indices in one JSON

Usage:
    uv run python scripts/gsa_sobol.py [--n-base 128] [--out-dir DIR]
"""

import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from peloton.gsa import METRICS, run_one

PROBLEM: dict = {
    "num_vars": 4,
    "names": ["recovery_rate", "breakaway_speed_frac", "logit_lambda", "k_s"],
    "bounds": [
        [0.02, 0.40],   # recovery_rate
        [0.70, 1.10],   # breakaway_speed_frac
        [0.10, 5.00],   # logit_lambda
        [0.70, 1.00],   # k_s  (Martins 2013: 0.7–1.0)
    ],
}

N_BASE = 128  # Saltelli base; total runs = N*(2k+2) = 1280 with second order


def _eval_row(args):
    row, n_generations, max_steps, seed = args
    params = dict(zip(PROBLEM["names"], row))
    return run_one(params, n_generations=n_generations, max_steps=max_steps, seed=seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-base", type=int, default=N_BASE,
                   help=f"Saltelli base N (default {N_BASE}; "
                        f"total runs = N×(2k+2) = {N_BASE*(2*PROBLEM['num_vars']+2)})")
    p.add_argument("--generations", type=int, default=200,
                   help="learning generations per sample (mirrors 'main.py learn')")
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--processes", type=int, default=os.cpu_count())
    p.add_argument("--out-dir", default="results/sobol")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    k = PROBLEM["num_vars"]
    total_runs = args.n_base * (2 * k + 2)
    print(f"[sobol] N={args.n_base}, k={k}, total runs={total_runs} (S1+S2+ST)")
    print(f"[sobol] parameters: {PROBLEM['names']}")

    samples = sobol_sample.sample(
        PROBLEM, args.n_base,
        calc_second_order=True,
        seed=args.seed,
    )
    print(f"[sobol] sample matrix: {samples.shape}")

    with mp.Pool(processes=args.processes) as pool:
        results = pool.map(_eval_row,
                           [(row, args.generations, args.max_steps, args.seed)
                            for row in samples])
    print(f"[sobol] evaluated {len(results)} samples")

    raw_df = pd.DataFrame(
        [{**dict(zip(PROBLEM["names"], row)), **res}
         for row, res in zip(samples, results)]
    )
    raw_path = os.path.join(args.out_dir, "raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"[sobol] raw results → {raw_path}")

    all_indices: dict = {}
    for metric in METRICS:
        Y = np.array([r[metric] for r in results])
        Si = sobol_analyze.analyze(
            PROBLEM, Y,
            calc_second_order=True,
            print_to_console=False,
            seed=args.seed,
        )

        all_indices[metric] = {
            "S1":      Si["S1"].tolist(),
            "S1_conf": Si["S1_conf"].tolist(),
            "S2":      Si["S2"].tolist(),
            "S2_conf": Si["S2_conf"].tolist(),
            "ST":      Si["ST"].tolist(),
            "ST_conf": Si["ST_conf"].tolist(),
        }

        # S1 and ST table
        df_1st = pd.DataFrame({
            "parameter": PROBLEM["names"],
            "S1":        Si["S1"],
            "S1_conf":   Si["S1_conf"],
            "ST":        Si["ST"],
            "ST_conf":   Si["ST_conf"],
        })
        out_1st = os.path.join(args.out_dir, f"sobol_S1_ST_{metric}.csv")
        df_1st.to_csv(out_1st, index=False)
        print(f"[sobol] {metric} S1/ST → {out_1st}")

        # S2 upper-triangle table
        s2_rows = []
        for i in range(k):
            for j in range(i + 1, k):
                s2_rows.append({
                    "param_i":  PROBLEM["names"][i],
                    "param_j":  PROBLEM["names"][j],
                    "S2":       Si["S2"][i, j],
                    "S2_conf":  Si["S2_conf"][i, j],
                })
        out_s2 = os.path.join(args.out_dir, f"sobol_S2_{metric}.csv")
        pd.DataFrame(s2_rows).to_csv(out_s2, index=False)
        print(f"[sobol] {metric} S2 → {out_s2}")

    json_path = os.path.join(args.out_dir, "sobol_indices.json")
    with open(json_path, "w") as f:
        json.dump(all_indices, f, indent=2)
    print(f"[sobol] all indices → {json_path}")
    print("[sobol] done.")


if __name__ == "__main__":
    main()
