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
"""

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime
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
from peloton.recorder import dump_run

# The knobs under analysis and their ranges (around PelotonConfig defaults).
PROBLEM = {
    "num_vars": 4,
    "names": ["recovery_rate", "breakaway_speed_frac", "utility_decay", "k_s"],
    "bounds": [
        [0.05, 0.20],    # recovery_rate        (default 0.1)
        [0.85, 1.00],    # breakaway_speed_frac (default 0.95)
        [0.10, 1.00],    # utility_decay/lambda (default 0.4)
        [0.70, 1.00],    # k_s                  (default 0.8)
    ],
}

# Emergent metrics, race-averaged over the final generation's race.
METRICS = ["MeanStamina", "NumGroups", "Breakaways", "MeanExposure"]

_MORRIS_LEVELS = 4


def _evaluate(args: tuple) -> np.ndarray:
    """Run one sample's evolution for `replicates` seeds; return seed-mean final metrics."""
    row, generations, max_steps, replicates, sample_idx, dump_dir, parquet = args
    overrides = dict(zip(PROBLEM["names"], (float(v) for v in row)))
    out = np.empty((replicates, len(METRICS)))
    for s in range(replicates):
        cfg = replace(PelotonConfig(seed=s), **overrides)
        history, population = run_generations(generations, max_steps, cfg,
                                              return_population=True)
        out[s] = [history[-1][m] for m in METRICS]
        if dump_dir is not None:
            run_dir = os.path.join(dump_dir, f"s{sample_idx:05d}_r{s:02d}")
            dump_run(cfg, max_steps, run_dir, parquet, population=population)
    return out.mean(axis=0)


def _simulate(X, generations, max_steps, replicates, processes,
              dump_dir=None, parquet=False, row_offset=0) -> np.ndarray:
    """Evaluate every sample row in parallel -> Y of shape (n_samples, n_metrics).

    row_offset shifts sample indices so dump directory names stay globally unique
    when X is a chunk of a larger design matrix.
    """
    tasks = [
        (row, generations, max_steps, replicates, row_offset + i, dump_dir, parquet)
        for i, row in enumerate(X)
    ]
    print(f"  evaluating rows {row_offset}..{row_offset + len(X) - 1} "
          f"({len(tasks)} samples x {replicates} reps x {generations} gens) ...",
          flush=True)
    with Pool(processes) as pool:
        return np.array(pool.map(_evaluate, tasks))


def run_method(method, n, generations, max_steps, replicates, processes,
               dump_dir=None, parquet=False) -> pd.DataFrame:
    """Sample, simulate, and estimate indices for one method. Returns long-form df."""
    if method == "morris":
        X = morris_sample(PROBLEM, n, num_levels=_MORRIS_LEVELS)
    elif method == "sobol":
        X = sobol_sample.sample(PROBLEM, n, calc_second_order=False)
    else:
        raise ValueError(f"unknown method {method!r}")

    if dump_dir is not None:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        # Write the design matrix so analysts can look up parameter values by sample index.
        pd.DataFrame(X, columns=PROBLEM["names"]).to_csv(
            os.path.join(dump_dir, "sample_index.csv"), index_label="sample_idx"
        )

    Y = _simulate(X, generations, max_steps, replicates, processes,
                  dump_dir=dump_dir, parquet=parquet)

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
    p.add_argument("--mode", choices=["full", "sample", "evaluate", "merge"], default="full",
                   help="full=sample+evaluate+analyze in one process (default); "
                        "sample=generate X.npy only; evaluate=simulate a row chunk; "
                        "merge=combine Y chunks and compute indices")
    p.add_argument("--method", choices=["morris", "sobol", "both"], default="both")
    p.add_argument("--samples", type=int, default=512,
                   help="SALib N: Morris trajectories (~N*(D+1) runs) / Sobol base "
                        "(~N*(D+2) runs); use a power of 2 for Sobol")
    p.add_argument("--replicates", type=int, default=5,
                   help="seed replicates averaged per sample (tames ABM noise)")
    p.add_argument("--generations", type=int, default=30,
                   help="races per evolution run (lambda only bites across generations)")
    p.add_argument("--max-steps", type=int, default=2000,
                   help="steps per race; must be high enough for riders to finish "
                        "across the full sweep range. With road_length=200000 m and "
                        "dt=30 s, k_s=0.70 (sweep minimum) needs ~1470 steps for "
                        "mean-power riders; 2000 gives safe headroom for slow/exhausted riders")
    p.add_argument("--processes", type=int, default=os.cpu_count())
    p.add_argument("--out-dir", default="data")
    p.add_argument("--dump-dir", default=None,
                   help="root directory for per-run agent-state dumps; "
                        "subdirs are created as <dump-dir>/<job_id>/<method>/s{N:05d}_r{R:02d}/")
    p.add_argument("--parquet", action="store_true",
                   help="write agent dumps as parquet instead of csv (recommended for GSA scale)")
    # --- split-job args (used by sample / evaluate / merge modes) ---
    p.add_argument("--x-file",    default=None, help="path to X.npy (sample output / evaluate+merge input)")
    p.add_argument("--y-out",     default=None, help="path to save Y chunk .npy (evaluate mode)")
    p.add_argument("--row-start", type=int, default=0,   help="first row of X to evaluate (evaluate mode)")
    p.add_argument("--row-end",   type=int, default=None, help="exclusive end row of X (evaluate mode)")
    p.add_argument("--merge-dir", default=None, help="directory containing Y_*.npy chunks (merge mode)")
    args = p.parse_args()

    # ── sample mode: generate Sobol design matrix and save ───────────────────
    if args.mode == "sample":
        X = sobol_sample.sample(PROBLEM, args.samples, calc_second_order=False)
        Path(args.x_file).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.x_file, X)
        pd.DataFrame(X, columns=PROBLEM["names"]).to_csv(
            str(args.x_file).replace(".npy", "_index.csv"), index_label="sample_idx"
        )
        print(f"[gsa] saved X {X.shape} → {args.x_file}")
        return

    # ── evaluate mode: simulate a contiguous chunk of X rows ─────────────────
    if args.mode == "evaluate":
        X      = np.load(args.x_file)
        r_end  = args.row_end if args.row_end is not None else len(X)
        X_chunk = X[args.row_start:r_end]
        dump_dir = args.dump_dir  # caller passes fully constructed path
        if dump_dir:
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
        Y_chunk = _simulate(X_chunk, args.generations, args.max_steps, args.replicates,
                            args.processes, dump_dir=dump_dir, parquet=args.parquet,
                            row_offset=args.row_start)
        Path(args.y_out).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.y_out, Y_chunk)
        print(f"[gsa] saved Y chunk {Y_chunk.shape} → {args.y_out}")
        return

    # ── merge mode: combine Y chunks and compute Sobol indices ───────────────
    if args.mode == "merge":
        X           = np.load(args.x_file)
        chunk_files = sorted(Path(args.merge_dir).glob("Y_*.npy"),
                             key=lambda f: int(f.stem.split("_")[1]))
        Y = np.concatenate([np.load(f) for f in chunk_files], axis=0)
        assert Y.shape == (len(X), len(METRICS)), \
            f"Y shape {Y.shape} does not match X rows {len(X)} x metrics {len(METRICS)}"
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for j, metric in enumerate(METRICS):
            Si   = sobol_analyze.analyze(PROBLEM, Y[:, j], calc_second_order=False)
            cols = {"S1": Si["S1"], "S1_conf": Si["S1_conf"],
                    "ST": Si["ST"], "ST_conf": Si["ST_conf"]}
            for i, param in enumerate(PROBLEM["names"]):
                rows.append({"metric": metric, "param": param,
                             **{k: v[i] for k, v in cols.items()}})
        out = out_dir / "gsa_sobol.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[gsa] merged {len(chunk_files)} chunks ({Y.shape[0]} rows) → {out}")
        return

    # ── full mode (default): sample + evaluate + analyze in one process ───────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ["morris", "sobol"] if args.method == "both" else [args.method]

    job_dump_dir = None
    if args.dump_dir:
        job_id   = os.environ.get("SLURM_JOB_ID", f"local_{datetime.now():%Y%m%d_%H%M%S}")
        job_dump_dir = os.path.join(args.dump_dir, job_id)
        Path(job_dump_dir).mkdir(parents=True, exist_ok=True)
        meta = {
            "job_id":       job_id,
            "created":      datetime.now().isoformat(timespec="seconds"),
            "method":       args.method,
            "samples":      args.samples,
            "generations":  args.generations,
            "replicates":   args.replicates,
            "max_steps":    args.max_steps,
            "problem":      PROBLEM,
            "metrics":      METRICS,
            "layout": (
                "{dump_dir}/{job_id}/meta.json  — this file\n"
                "{dump_dir}/{job_id}/{method}/sample_index.csv  — design matrix (row=sample)\n"
                "{dump_dir}/{job_id}/{method}/s{N:05d}_r{R:02d}/  — final-gen race dump\n"
                "    config.json, agent_timeseries, model_timeseries, agent_meta, finish_order, manifest.json"
            ),
        }
        with open(os.path.join(job_dump_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[gsa] dump root: {job_dump_dir}/")

    for method in methods:
        print(f"[gsa] {method} (N={args.samples})")
        method_dump = os.path.join(job_dump_dir, method) if job_dump_dir else None
        df = run_method(method, args.samples, args.generations, args.max_steps,
                        args.replicates, args.processes,
                        dump_dir=method_dump, parquet=args.parquet)
        out = out_dir / f"gsa_{method}.csv"
        df.to_csv(out, index=False)
        print(f"[gsa] wrote {out}")


if __name__ == "__main__":
    main()
