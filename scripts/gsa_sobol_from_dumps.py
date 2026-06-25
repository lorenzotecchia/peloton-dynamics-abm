#!/usr/bin/env python3
"""Rebuild a Sobol GSA CSV from an existing per-run agent-dump tree — no re-simulation.

This reuses the *raw data* already on disk: it reads each run's
``model_timeseries.parquet``, race-averages the four emergent METRICS over steps,
averages those over the seed replicates, and feeds the resulting Y matrix straight
into SALib's Sobol estimator — the same final step as ``peloton.gsa``.

    uv run python scripts/gsa_sobol_from_dumps.py <DUMP_DIR> [--out FILE] [--processes N]

<DUMP_DIR> is the directory that holds ``sample_index.csv`` and the
``s{NNNNN}_r{RR}/`` run directories, e.g. on Snellius:

    uv run python scripts/gsa_sobol_from_dumps.py \
        /gpfs/work5/0/prjs2142/gsa-agent-dump-per-run/24187588-2fd8a6c-sobol \
        --out data/gsa_sobol_from_dumps.csv

IMPORTANT — this does NOT reproduce the original gsa_sobol.csv bit-for-bit.
The GSA driver's Y came from ``history[-1]`` (the final-generation race, *before*
that generation's evolve()), whereas dump_run recorded a fresh race using the
*post-final-evolve* population (one generation further on). The scored quantity was
never persisted, so the indices here describe that one-generation-later population.
They are a valid, internally-consistent Sobol analysis, just not identical numbers.
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze

# Reuse the canonical problem/metric definitions so this can't drift from the model.
from peloton.gsa import METRICS, PROBLEM


def _race_mean(run_dir: Path) -> np.ndarray:
    """Race-average the METRICS over steps for one run dir; NaN row if unreadable."""
    path = run_dir / "model_timeseries.parquet"
    df = pd.read_parquet(path, columns=METRICS)
    return df[METRICS].mean().to_numpy(dtype=float)


def _eval_one(args: tuple) -> tuple:
    """Worker: mean the METRICS over a sample's replicate run dirs."""
    sample_idx, dump_dir, replicates = args
    acc = np.zeros((replicates, len(METRICS)), dtype=float)
    for r in range(replicates):
        run_dir = Path(dump_dir) / f"s{sample_idx:05d}_r{r:02d}"
        if not (run_dir / "model_timeseries.parquet").exists():
            raise FileNotFoundError(f"missing dump: {run_dir}/model_timeseries.parquet")
        acc[r] = _race_mean(run_dir)
    return sample_idx, acc.mean(axis=0)


def _detect_replicates(dump_dir: Path) -> int:
    """Count r## dirs for sample 0 (assumes a rectangular sample x replicate grid)."""
    reps = sorted(dump_dir.glob("s00000_r*"))
    if not reps:
        raise SystemExit(f"no s00000_r* run dirs under {dump_dir}")
    return len(reps)


def build_Y(dump_dir: Path, n_samples: int, replicates: int, processes: int) -> np.ndarray:
    """Y of shape (n_samples, len(METRICS)): replicate-mean of per-run race-means."""
    tasks = [(s, str(dump_dir), replicates) for s in range(n_samples)]
    Y = np.empty((n_samples, len(METRICS)), dtype=float)
    print(f"[from-dumps] aggregating {n_samples} samples x {replicates} replicates "
          f"x {len(METRICS)} metrics using {processes} processes ...", flush=True)
    done = 0
    with Pool(processes) as pool:
        for sample_idx, row in pool.imap_unordered(_eval_one, tasks, chunksize=8):
            Y[sample_idx] = row
            done += 1
            if done % 256 == 0 or done == n_samples:
                print(f"  {done}/{n_samples} samples", flush=True)
    return Y


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump_dir", help="dir holding sample_index.csv and s{NNNNN}_r{RR}/ run dirs")
    p.add_argument("--out", default="data/gsa_sobol_from_dumps.csv",
                   help="output CSV (default: data/gsa_sobol_from_dumps.csv)")
    p.add_argument("--replicates", type=int, default=None,
                   help="seed replicates per sample (default: auto-detect from s00000_r* dirs)")
    p.add_argument("--processes", type=int, default=min(8, os.cpu_count() or 1),
                   help="parallel parquet readers (default: min(8, ncpu); raise on a compute node)")
    args = p.parse_args()

    dump_dir = Path(args.dump_dir)
    index_csv = dump_dir / "sample_index.csv"
    if not index_csv.exists():
        raise SystemExit(f"no sample_index.csv under {dump_dir}")

    # The design matrix order IS the sample order; we only need its row count and a
    # sanity check that it matches the model's PROBLEM definition.
    X = pd.read_csv(index_csv)
    n_samples = len(X)
    param_cols = [c for c in X.columns if c != "sample_idx"]
    if param_cols != PROBLEM["names"]:
        print(f"[warn] sample_index columns {param_cols} != PROBLEM['names'] "
              f"{PROBLEM['names']}; ensure this dump matches the current model.",
              file=sys.stderr)

    replicates = args.replicates or _detect_replicates(dump_dir)

    # SALib (calc_second_order=False) expects n_samples == N*(D+2).
    d = PROBLEM["num_vars"]
    if n_samples % (d + 2) != 0:
        print(f"[warn] n_samples={n_samples} is not a multiple of D+2={d + 2}; "
              f"sobol_analyze will likely reject it.", file=sys.stderr)
    else:
        print(f"[from-dumps] N={n_samples // (d + 2)} (n_samples={n_samples}, D={d}, replicates={replicates})")

    Y = build_Y(dump_dir, n_samples, replicates, args.processes)

    if not np.isfinite(Y).all():
        bad = np.argwhere(~np.isfinite(Y))
        raise SystemExit(f"non-finite Y values at (sample, metric) {bad[:10].tolist()} ...")

    rows = []
    for j, metric in enumerate(METRICS):
        Si = sobol_analyze.analyze(PROBLEM, Y[:, j], calc_second_order=False)
        cols = {"S1": Si["S1"], "S1_conf": Si["S1_conf"],
                "ST": Si["ST"], "ST_conf": Si["ST_conf"]}
        for i, param in enumerate(PROBLEM["names"]):
            rows.append({"metric": metric, "param": param,
                         **{k: v[i] for k, v in cols.items()}})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[from-dumps] wrote {out} ({len(rows)} rows, "
          f"{len(METRICS)} metrics x {len(PROBLEM['names'])} params)")


if __name__ == "__main__":
    main()
