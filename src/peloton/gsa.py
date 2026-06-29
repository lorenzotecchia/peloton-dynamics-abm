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
Targets, read from the final generation's race: MeanStamina, NumGroups,
Breakaways, MeanExposure (per-step race-means) plus the race-end aggregates
TotalTime / MeanTime and TotalStaminaSpent / MeanStaminaSpent (time and W'
consumed, summed over the field and per-rider). Fixed scenario knobs
(road_length, dt, group_radius) are not SA-varied; set them with
--road-length / --dt / --group-radius.
"""

import argparse
import os
import time
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
    ("w_max10_mean", 350.0, 550.0),  # 450  mean 10-min max power (W)
    ("w_max10_std", 34.0, 102.0),  # 68   power spread across the field
    ("cp_fraction", 0.60, 0.80),  # 0.7  critical-power fraction
    ("recovery_rate", 0.05, 0.20),  # 0.1  W' recovery below CP
    ("k_aero", 0.60, 1.20),  # 0.9  aerodynamic coefficient
    ("c_roll", 2.0, 5.0),  # 3.6  rolling-resistance coefficient
    ("ref_speed_frac", 0.80, 0.95),  # 0.9  v_hat fraction for stamina init
    ("k_s", 0.70, 1.00),  # 0.9  pack-speed coefficient
    ("breakaway_speed_frac", 0.85, 1.00),  # 0.9  solo speed / threshold
    ("breakaway_cooldown_steps", 2, 20),  # 10   breaker re-merge cooldown (int)
    ("utility_decay", 0.10, 1.00),  # 0.8  lambda: position->utility decay
    ("evo_noise", 0.0, 0.10),  # 0.02 per-generation Gaussian noise
    ("sim_scale", 0.50, 2.00),  # 1.0  rider-similarity bandwidth
    ("evo_bottom_frac", 0.10, 0.40),  # 0.2  fraction of worst updated
    ("evo_top_frac", 0.05, 0.30),  # 0.1  fraction of top used as donors
    ("imitation_mu", 0.30, 1.00),  # 0.75 blend toward donor (1=copy)
    ("n_agents", 24, 192),  # 96   pack size (int, 2-16/team)
    ("n_teams", 4, 24),  # 12   number of teams (int)
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
    "names": [
        "recovery_rate",
        "breakaway_speed_frac",
        "utility_decay",
        "k_s",
        "n_agents",
    ],
    "bounds": [
        [0.05, 0.20],  # recovery_rate        (default 0.1)
        [0.85, 1.00],  # breakaway_speed_frac (default 0.9)
        [0.10, 1.00],  # utility_decay/lambda (default 0.8)
        [0.70, 1.00],  # k_s                  (default 0.9)
        [24, 192],  # n_agents (default 96; 2-16 riders/team at n_teams=12)
    ],
}

PROBLEMS = {"morris": PROBLEM_MORRIS, "sobol": PROBLEM_SOBOL}

# Knobs that count/index things: round the float sample to int before it reaches
# PelotonConfig (replace() would otherwise leave e.g. range(144.0) -> TypeError).
INT_PARAMS = {"n_agents", "n_teams", "breakaway_cooldown_steps"}

# Emergent SA targets read from the final generation's race. The first four are
# race-means (per-step, via the DataCollector); the Total*/Mean* time & stamina
# metrics are race-end aggregates over the whole field (see _race_totals).
METRICS = [
    "MeanStamina",
    "NumGroups",
    "Breakaways",
    "MeanExposure",
    "TotalTime",
    "MeanTime",
    "TotalStaminaSpent",
    "MeanStaminaSpent",
]

_MORRIS_LEVELS = 4

# Sobol second-order (pairwise S2) indices. One switch drives both the sampling
# and the analysis so they can never disagree: True makes the Sobol design
# N*(2D+2) rows (vs N*(D+2)) and emits a gsa_sobol_S2.csv of pairwise indices
# alongside the first/total-order gsa_sobol.csv. Morris is unaffected.
SOBOL_SECOND_ORDER = True


def _evaluate(args: tuple) -> np.ndarray:
    """Run one sample's evolution for `replicates` seeds; return seed-mean final metrics."""
    row, names, generations, max_steps, replicates, base = args
    overrides = {
        n: (int(round(v)) if n in INT_PARAMS else float(v)) for n, v in zip(names, row)
    }
    out = np.empty((replicates, len(METRICS)))
    for s in range(replicates):
        # `base` holds the fixed scenario (road_length, dt, group_radius, ...);
        # `overrides` are the SA knobs this sample varies. SA knobs win on overlap.
        cfg = replace(PelotonConfig(seed=s), **base, **overrides)
        last = run_generations(generations, max_steps, cfg)[-1]
        out[s] = [last[m] for m in METRICS]
    return out.mean(axis=0)


def _fmt_dur(seconds: float) -> str:
    """Human-readable h/m/s for a duration (used by the live progress line)."""
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _print_progress(k: int, total: int, start: int, t0: float) -> None:
    """Print '[k/total] pct  elapsed  eta  rate' from the throughput so far.

    ETA extrapolates from samples completed in *this* process (k-start) over the
    wall time since it began, so a checkpoint-resume doesn't skew the rate.
    """
    elapsed = time.time() - t0
    completed = k - start
    if completed <= 0:
        return
    rate = completed / elapsed  # samples per second
    eta = (total - k) / rate if rate > 0 else float("inf")
    print(
        f"  [{k}/{total}] {100 * k / total:5.1f}%  "
        f"elapsed {_fmt_dur(elapsed)}  eta {_fmt_dur(eta)}  "
        f"({rate * 3600:.0f} samples/hr)",
        flush=True,
    )


def _simulate(
    X, names, generations, max_steps, replicates, processes, base, y_path=None
) -> np.ndarray:
    """Evaluate every sample row in parallel -> Y of shape (n_samples, n_metrics).

    With ``y_path`` (single-node ``full`` mode), each completed sample is
    checkpointed to it (append + flush) so a killed/timed-out job resumes from
    disk instead of recomputing; the column count is validated against METRICS on
    resume. Without ``y_path`` (chunked array ``evaluate`` mode) the rows are
    mapped in memory and returned -- the array task writes the Y chunk itself.
    """
    done = []
    if y_path is not None and y_path.exists():
        arr = np.loadtxt(y_path, delimiter=",", ndmin=2)
        # Guard against resuming onto a checkpoint from an incompatible METRICS set
        # (e.g. one written before time/stamina-spent targets were added): the column
        # count must match or the appended rows would be ragged and Y[:, j] would lie.
        if arr.shape[0] and arr.shape[1] != len(METRICS):
            raise ValueError(
                f"checkpoint {y_path} has {arr.shape[1]} metric columns but METRICS now "
                f"defines {len(METRICS)} ({', '.join(METRICS)}); it was written by an "
                f"incompatible run. Delete it and the matching .gsa_*_X.npy to start fresh."
            )
        done = list(arr)
    start = len(done)
    if start:
        print(f"  resuming from checkpoint: {start}/{len(X)} samples done", flush=True)

    tasks = [
        (row, names, generations, max_steps, replicates, base) for row in X[start:]
    ]
    total = len(X)
    print(
        f"  evaluating {len(tasks)} samples x {replicates} reps x {generations} gens "
        f"on {processes} procs (progress + eta printed per sample) ...",
        flush=True,
    )
    t0 = time.time()
    # imap keeps row order, so checkpoint row k is always written after k-1. The
    # checkpoint file (full mode) is opened for append; chunked evaluate mode has
    # no y_path and just collects rows in memory. Both report live progress.
    fh = open(y_path, "a") if y_path is not None else None
    try:
        with Pool(processes) as pool:
            for k, row in enumerate(pool.imap(_evaluate, tasks), start=start + 1):
                if fh is not None:
                    fh.write(",".join(repr(float(v)) for v in row) + "\n")
                    fh.flush()
                done.append(row)
                _print_progress(k, total, start, t0)
    finally:
        if fh is not None:
            fh.close()
    return np.array(done)


def _sample_design(method, n) -> np.ndarray:
    """Draw the SALib design matrix X for one method (Morris trajectories / Sobol base)."""
    problem = PROBLEMS[method]
    if method == "morris":
        return morris_sample(problem, n, num_levels=_MORRIS_LEVELS)
    if method == "sobol":
        return sobol_sample.sample(problem, n, calc_second_order=SOBOL_SECOND_ORDER)
    raise ValueError(f"unknown method {method!r}")


def _analyze(method, problem, X, Y):
    """Estimate sensitivity indices from design X and outputs Y.

    Returns ``(rows, s2_rows)``: ``rows`` is the long-form first/total-order
    (Sobol) or mu*/sigma (Morris) table; ``s2_rows`` holds the Sobol pairwise
    second-order indices (one row per parameter pair) when SOBOL_SECOND_ORDER is
    on, else an empty list.
    """
    names = problem["names"]
    rows, s2_rows = [], []
    for j, metric in enumerate(METRICS):
        if method == "morris":
            Si = morris_analyze(problem, X, Y[:, j], num_levels=_MORRIS_LEVELS)
            cols = {
                "mu_star": Si["mu_star"],
                "mu_star_conf": Si["mu_star_conf"],
                "sigma": Si["sigma"],
            }
        else:
            Si = sobol_analyze.analyze(
                problem, Y[:, j], calc_second_order=SOBOL_SECOND_ORDER
            )
            cols = {
                "S1": Si["S1"],
                "S1_conf": Si["S1_conf"],
                "ST": Si["ST"],
                "ST_conf": Si["ST_conf"],
            }
            if SOBOL_SECOND_ORDER:
                # S2/S2_conf are DxD with the pairwise indices in the upper triangle.
                S2, S2c = Si["S2"], Si["S2_conf"]
                for a in range(len(names)):
                    for b in range(a + 1, len(names)):
                        s2_rows.append(
                            {
                                "metric": metric,
                                "param_1": names[a],
                                "param_2": names[b],
                                "S2": S2[a, b],
                                "S2_conf": S2c[a, b],
                            }
                        )
        for i, param in enumerate(names):
            rows.append(
                {"metric": metric, "param": param, **{k: v[i] for k, v in cols.items()}}
            )
    return rows, s2_rows


def run_method(
    method, n, generations, max_steps, replicates, processes, out_dir, base=None
):
    """Sample, simulate, and estimate indices for one method.

    Returns ``(df, df_s2)``: the first/total-order (or Morris) indices and, for
    Sobol with SOBOL_SECOND_ORDER on, the pairwise second-order table (else None).
    """
    problem = PROBLEMS[method]  # Morris and Sobol vary different knob sets.
    # Pin the design to disk: Morris sampling isn't reproducible across runs, so a
    # resume must reuse the exact X that the checkpointed Y rows were computed for.
    x_path = Path(out_dir) / f".gsa_{method}_X.npy"
    if x_path.exists():
        X = np.load(x_path)
    else:
        X = _sample_design(method, n)
        np.save(x_path, X)

    y_path = Path(out_dir) / f".gsa_{method}_Y.csv"
    Y = _simulate(
        X,
        problem["names"],
        generations,
        max_steps,
        replicates,
        processes,
        base or {},
        y_path=y_path,
    )
    rows, s2_rows = _analyze(method, problem, X, Y)
    return pd.DataFrame(rows), (pd.DataFrame(s2_rows) if s2_rows else None)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=["full", "sample", "evaluate", "merge"],
        default="full",
        help="full=sample+evaluate+analyze in one process (default); the rest "
        "split a run across a Slurm array: sample=write X.npy only; "
        "evaluate=simulate a row chunk of X -> Y chunk; merge=combine "
        "Y chunks and write the indices CSV",
    )
    p.add_argument("--method", choices=["morris", "sobol", "both"], default="both")
    p.add_argument(
        "--samples",
        type=int,
        default=512,
        help="SALib N: Morris trajectories (~N*(D+1) runs) / Sobol base "
        "(~N*(D+2) runs); use a power of 2 for Sobol",
    )
    p.add_argument(
        "--replicates",
        type=int,
        default=5,
        help="seed replicates averaged per sample (tames ABM noise)",
    )
    p.add_argument(
        "--generations",
        type=int,
        default=30,
        help="races per evolution run (lambda only bites across generations)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="steps per race; must be high enough for riders to finish "
        "(~600+) or utility is degenerate and lambda can't bite",
    )
    p.add_argument("--processes", type=int, default=os.cpu_count())
    p.add_argument("--out-dir", default="data")
    # Fixed scenario knobs (not SA-varied): override PelotonConfig for every sample.
    # Defaults None = keep config defaults. For the long-road/coarse-dt experiment
    # pass e.g. --road-length 200000 --dt 60 --group-radius 300 (and bump --max-steps).
    p.add_argument(
        "--road-length", type=float, default=None, help="finish line in m (e.g. 200000)"
    )
    p.add_argument(
        "--dt", type=float, default=None, help="seconds of race per step (e.g. 60)"
    )
    p.add_argument(
        "--group-radius",
        type=float,
        default=None,
        help="longitudinal pack radius in m; scale with v*dt for coarse dt",
    )
    # --- split-job args (used by sample / evaluate / merge modes) ---
    p.add_argument(
        "--x-file",
        default=None,
        help="path to X.npy (written by sample; read by evaluate + merge)",
    )
    p.add_argument(
        "--y-out", default=None, help="path to save the Y chunk .npy (evaluate)"
    )
    p.add_argument(
        "--row-start", type=int, default=0, help="first X row to evaluate (evaluate)"
    )
    p.add_argument(
        "--row-end", type=int, default=None, help="exclusive end X row (evaluate)"
    )
    p.add_argument(
        "--merge-dir",
        default=None,
        help="directory holding the Y_<start>_<end>.npy chunks (merge)",
    )
    args = p.parse_args()

    base = {
        k: v
        for k, v in (
            ("road_length", args.road_length),
            ("dt", args.dt),
            ("group_radius", args.group_radius),
        )
        if v is not None
    }

    # ── sample mode: draw X for one method and persist it (login node, seconds) ──
    if args.mode == "sample":
        if args.method == "both":
            raise SystemExit("--mode sample needs a single --method (morris|sobol)")
        X = _sample_design(args.method, args.samples)
        xp = Path(args.x_file)
        xp.parent.mkdir(parents=True, exist_ok=True)
        np.save(xp, X)
        pd.DataFrame(X, columns=PROBLEMS[args.method]["names"]).to_csv(
            xp.with_name(xp.stem + "_index.csv"), index_label="sample_idx"
        )
        print(f"[gsa] {args.method}: saved X {X.shape} -> {xp}")
        print(f"[gsa] n_rows={len(X)}")  # the array wrapper reads this to size chunks
        return

    # ── evaluate mode: simulate a contiguous chunk of X rows -> Y chunk ──────────
    if args.mode == "evaluate":
        if args.method == "both":
            raise SystemExit("--mode evaluate needs a single --method (morris|sobol)")
        X = np.load(args.x_file)
        r_end = args.row_end if args.row_end is not None else len(X)
        X_chunk = X[args.row_start : r_end]
        Y_chunk = _simulate(
            X_chunk,
            PROBLEMS[args.method]["names"],
            args.generations,
            args.max_steps,
            args.replicates,
            args.processes,
            base,
        )
        yp = Path(args.y_out)
        yp.parent.mkdir(parents=True, exist_ok=True)
        np.save(yp, Y_chunk)
        print(
            f"[gsa] {args.method}: saved Y chunk {Y_chunk.shape} "
            f"(rows {args.row_start}..{r_end}) -> {yp}"
        )
        return

    # ── merge mode: concat the Y chunks (row order) and write the indices CSV ────
    if args.mode == "merge":
        if args.method == "both":
            raise SystemExit("--mode merge needs a single --method (morris|sobol)")
        problem = PROBLEMS[args.method]
        X = np.load(args.x_file)
        chunk_files = sorted(
            Path(args.merge_dir).glob("Y_*.npy"),
            key=lambda f: int(f.stem.split("_")[1]),
        )
        if not chunk_files:
            raise FileNotFoundError(f"no Y_*.npy chunks found in {args.merge_dir}")
        Y = np.concatenate([np.load(f) for f in chunk_files], axis=0)
        if Y.shape != (len(X), len(METRICS)):
            raise ValueError(
                f"merged Y {Y.shape} != expected ({len(X)}, {len(METRICS)}); a chunk is "
                f"missing or overlapping in {args.merge_dir}."
            )
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows, s2_rows = _analyze(args.method, problem, X, Y)
        out = out_dir / f"gsa_{args.method}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(
            f"[gsa] {args.method}: merged {len(chunk_files)} chunks "
            f"({Y.shape[0]} rows) -> {out}"
        )
        if s2_rows:
            out2 = out_dir / f"gsa_{args.method}_S2.csv"
            pd.DataFrame(s2_rows).to_csv(out2, index=False)
            print(f"[gsa] {args.method}: second-order indices -> {out2}")
        return

    # ── full mode (default): sample + evaluate + analyze in one process ──────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ["morris", "sobol"] if args.method == "both" else [args.method]
    for method in methods:
        print(f"[gsa] {method} (N={args.samples}) base={base or 'defaults'}")
        df, df_s2 = run_method(
            method,
            args.samples,
            args.generations,
            args.max_steps,
            args.replicates,
            args.processes,
            out_dir,
            base,
        )
        out = out_dir / f"gsa_{method}.csv"
        df.to_csv(out, index=False)
        print(f"[gsa] wrote {out}")
        if df_s2 is not None:
            out2 = out_dir / f"gsa_{method}_S2.csv"
            df_s2.to_csv(out2, index=False)
            print(f"[gsa] wrote {out2}  (second-order indices)")
        # Indices written: drop the checkpoint so a fresh re-run resamples.
        (out_dir / f".gsa_{method}_X.npy").unlink(missing_ok=True)
        (out_dir / f".gsa_{method}_Y.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
