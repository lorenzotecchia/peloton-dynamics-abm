"""Dump full per-agent state across a GSA parameter sample (no indices).

Same Morris/Sobol *sampling* as ``peloton.gsa`` (it reuses that module's PROBLEMS
and INT_PARAMS), but instead of reducing each sample to sensitivity indices it
runs the evolution loop and writes the complete ``main.py dump`` bundle for
*every generation* of *every sample*. The raw agent data lands on disk so the
sensitivity analysis can be (re)computed from it later — the "reuse data for SA"
workflow — rather than baked into indices here.

    python -m peloton.gsa_dump --method morris --samples 2 --generations 10 \
        --max-steps 2000 --processes $SLURM_CPUS_PER_TASK --out-dir DIR

No replication: every race uses seed 0 (the GSA's seed replicates are for taming
ABM noise in the *indices*, which we don't compute here). Keep --samples small —
this writes per-step per-agent rows for samples x generations whole races.

Output layout under <out-dir>:
    <method>/
        X.npy                    the SALib design matrix (one row per sample)
        sample_<i>/
            params.json          this sample's varied-knob values
            gen_<g>/             one full dump bundle per generation
                agent_timeseries.csv  model_timeseries.csv  agent_meta.csv
                finish_order.csv      config.json           manifest.json

With ``--last-gen-only`` only the final generation's ``gen_<g>/`` bundle is written
(the full learning loop still runs; intermediate generations evolve the coeffs but
are not dumped) — far less disk for the common "just the converged race" case.

Run count: Morris draws ~samples*(D+1) design rows (D=18), Sobol ~samples*(D+2)
(D=5); each design row then runs `generations` races. Size --samples/--generations
so the whole thing fits your walltime and disk.
"""

import argparse
import copy
import json
import os
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.sample import sobol as sobol_sample
from SALib.sample.morris import sample as morris_sample

from peloton.config import PelotonConfig
from peloton.evolution import evolve
from peloton.gsa import INT_PARAMS, PROBLEMS, _MORRIS_LEVELS
from peloton.model import PelotonModel
from peloton.recorder import record_run, write_bundle

# Seed for every race. No replication: the GSA replicates exist to average ABM
# noise out of the *indices*; here we keep the raw runs, one deterministic seed.
SEED = 0


def _dump_sample(task: tuple) -> tuple:
    """Run one sample's evolution, dumping the full bundle for every generation.

    Coefficients persist across generations via ``population`` exactly as in
    ``evolution.run_generations``; the only addition is that each generation's
    race is recorded (``record_run``) and written (``write_bundle``) before the
    field evolves into the next one.
    """
    idx, row, names, generations, max_steps, base, sample_dir, parquet, \
        last_gen_only = task
    sample_dir = Path(sample_dir)
    overrides = {n: (int(round(v)) if n in INT_PARAMS else float(v))
                 for n, v in zip(names, row)}

    population: list[dict] | None = None
    for gen in range(generations):
        # `base` is the fixed scenario (road_length/dt/group_radius); `overrides`
        # are this sample's SA knobs (they win on overlap). `population` carries
        # the learned coeffs forward from the previous generation.
        cfg = replace(PelotonConfig(seed=SEED), **base, **overrides)
        model = PelotonModel(config=cfg, population=population)
        data = record_run(cfg, max_steps, model=model)
        # Always run the full learning trajectory, but with `last_gen_only` only
        # the final generation's race is written to disk (the others still evolve
        # the coeffs forward, they're just not dumped).
        if not last_gen_only or gen == generations - 1:
            write_bundle(data, str(sample_dir / f"gen_{gen:04d}"), parquet)
        evolve(model.riders, model)
        population = [copy.deepcopy(r.coeffs) for r in model.riders]

    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "params.json").write_text(json.dumps(overrides, indent=2))
    return idx, str(sample_dir)


def _sample_design(method, n) -> np.ndarray:
    """Draw the dump's SALib design matrix (Sobol uses calc_second_order=False)."""
    problem = PROBLEMS[method]  # Morris and Sobol vary different knob sets.
    if method == "morris":
        return morris_sample(problem, n, num_levels=_MORRIS_LEVELS)
    if method == "sobol":
        return sobol_sample.sample(problem, n, calc_second_order=False)
    raise ValueError(f"unknown method {method!r}")


def run_dump(method, n, generations, max_steps, processes, out_dir, parquet, base,
             X=None, row_start=0, row_end=None, save_x=True,
             last_gen_only=False) -> None:
    """Dump every generation of a (chunk of a) sample design to disk.

    ``X`` lets a caller pass a pre-sampled design so a Slurm array of jobs all dump
    the *same* matrix; when None we sample it here. ``row_start``/``row_end`` limit
    this call to a contiguous row chunk -- the sample dir keeps its *global* index
    (``sample_<i>``) so chunks dumped by sibling jobs never collide and can share
    one out dir. ``save_x`` writes X.npy for provenance (the array wrapper does that
    once on the login node, so its tasks pass False). ``last_gen_only`` dumps only
    the final generation's race per sample (the learning loop still runs in full)."""
    problem = PROBLEMS[method]
    if X is None:
        X = _sample_design(method, n)

    method_dir = Path(out_dir) / method
    method_dir.mkdir(parents=True, exist_ok=True)
    if save_x:
        np.save(method_dir / "X.npy", X)

    r_end = len(X) if row_end is None else min(row_end, len(X))
    tasks = [
        (i, X[i], problem["names"], generations, max_steps, base,
         str(method_dir / f"sample_{i:04d}"), parquet, last_gen_only)
        for i in range(row_start, r_end)
    ]
    dumped = "last gen only" if last_gen_only else "all gens"
    print(f"[gsa-dump] {method}: {len(tasks)} samples (rows {row_start}..{r_end} "
          f"of {len(X)}) x {generations} gens (seed {SEED}, no replication, "
          f"dumping {dumped}) base={base or 'defaults'}", flush=True)
    with Pool(processes) as pool:
        for done, (idx, sd) in enumerate(pool.imap_unordered(_dump_sample, tasks), 1):
            print(f"  [{method}] sample {idx} dumped ({done}/{len(tasks)}) -> {sd}",
                  flush=True)
    print(f"[gsa-dump] wrote {method} dumps under {method_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["full", "sample", "dump"], default="full",
                   help="full=sample+dump all rows in one process (default); the other "
                        "two split a run across a Slurm array: sample=write X.npy only "
                        "(login node); dump=dump a --row-start/--row-end row chunk of a "
                        "pre-sampled --x-file (array task)")
    p.add_argument("--method", choices=["morris", "sobol", "both"], default="morris")
    p.add_argument("--samples", type=int, default=2,
                   help="SALib N: Morris trajectories (~N*(D+1) rows) / Sobol base "
                        "(~N*(D+2) rows); keep small — every row dumps full races")
    p.add_argument("--generations", type=int, default=10,
                   help="races per sample; each is dumped in full (lambda bites "
                        "across generations)")
    p.add_argument("--max-steps", type=int, default=2000,
                   help="steps per race; high enough for riders to finish (~600+)")
    p.add_argument("--processes", type=int, default=os.cpu_count())
    p.add_argument("--out-dir", default="data")
    p.add_argument("--parquet", action="store_true",
                   help="write Parquet instead of CSV (needs pyarrow)")
    p.add_argument("--last-gen-only", action="store_true",
                   help="dump only the final generation's race per sample (the "
                        "full learning loop still runs; intermediate generations "
                        "evolve the coeffs but are not written to disk)")
    # Fixed scenario knobs (not SA-varied): override PelotonConfig for every sample.
    p.add_argument("--road-length", type=float, default=None, help="finish line in m")
    p.add_argument("--dt", type=float, default=None, help="seconds of race per step")
    p.add_argument("--group-radius", type=float, default=None,
                   help="longitudinal pack radius in m")
    # --- split-job args (used by sample / dump modes for the Slurm array) ---
    p.add_argument("--x-file", default=None,
                   help="path to X.npy: written by --mode sample, read by --mode dump")
    p.add_argument("--row-start", type=int, default=0,
                   help="first design row to dump (--mode dump)")
    p.add_argument("--row-end", type=int, default=None,
                   help="exclusive last design row to dump (--mode dump)")
    args = p.parse_args()

    base = {k: v for k, v in (("road_length", args.road_length), ("dt", args.dt),
                              ("group_radius", args.group_radius)) if v is not None}

    # ── sample mode: draw X for one method and persist it (login node, seconds) ──
    if args.mode == "sample":
        if args.method == "both":
            raise SystemExit("--mode sample needs a single --method (morris|sobol)")
        X = _sample_design(args.method, args.samples)
        xp = Path(args.x_file)
        xp.parent.mkdir(parents=True, exist_ok=True)
        np.save(xp, X)
        pd.DataFrame(X, columns=PROBLEMS[args.method]["names"]).to_csv(
            xp.with_name(xp.stem + "_index.csv"), index_label="sample_idx")
        print(f"[gsa-dump] {args.method}: saved X {X.shape} -> {xp}")
        print(f"[gsa-dump] n_rows={len(X)}")  # the array wrapper reads this to size chunks
        return

    # ── dump mode: dump a contiguous row chunk of a pre-sampled X (array task) ───
    if args.mode == "dump":
        if args.method == "both":
            raise SystemExit("--mode dump needs a single --method (morris|sobol)")
        X = np.load(args.x_file)
        run_dump(args.method, args.samples, args.generations, args.max_steps,
                 args.processes, args.out_dir, args.parquet, base,
                 X=X, row_start=args.row_start, row_end=args.row_end, save_x=False,
                 last_gen_only=args.last_gen_only)
        return

    # ── full mode (default): sample + dump every row in one process ──────────────
    methods = ["morris", "sobol"] if args.method == "both" else [args.method]
    for method in methods:
        run_dump(method, args.samples, args.generations, args.max_steps,
                 args.processes, args.out_dir, args.parquet, base,
                 last_gen_only=args.last_gen_only)


if __name__ == "__main__":
    main()
