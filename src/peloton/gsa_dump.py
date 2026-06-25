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
    idx, row, names, generations, max_steps, base, sample_dir, parquet = task
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
        write_bundle(data, str(sample_dir / f"gen_{gen:04d}"), parquet)
        evolve(model.riders, model)
        population = [copy.deepcopy(r.coeffs) for r in model.riders]

    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "params.json").write_text(json.dumps(overrides, indent=2))
    return idx, str(sample_dir)


def run_dump(method, n, generations, max_steps, processes, out_dir, parquet, base) -> None:
    """Sample `method`, then dump every generation of every sample to disk."""
    problem = PROBLEMS[method]  # Morris and Sobol vary different knob sets.
    if method == "morris":
        X = morris_sample(problem, n, num_levels=_MORRIS_LEVELS)
    elif method == "sobol":
        X = sobol_sample.sample(problem, n, calc_second_order=False)
    else:
        raise ValueError(f"unknown method {method!r}")

    method_dir = Path(out_dir) / method
    method_dir.mkdir(parents=True, exist_ok=True)
    np.save(method_dir / "X.npy", X)

    tasks = [
        (i, row, problem["names"], generations, max_steps, base,
         str(method_dir / f"sample_{i:04d}"), parquet)
        for i, row in enumerate(X)
    ]
    print(f"[gsa-dump] {method}: {len(tasks)} samples x {generations} gens "
          f"(seed {SEED}, no replication) base={base or 'defaults'}", flush=True)
    with Pool(processes) as pool:
        for done, (idx, sd) in enumerate(pool.imap_unordered(_dump_sample, tasks), 1):
            print(f"  [{method}] sample {idx} dumped ({done}/{len(tasks)}) -> {sd}",
                  flush=True)
    print(f"[gsa-dump] wrote {method} dumps under {method_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
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
    # Fixed scenario knobs (not SA-varied): override PelotonConfig for every sample.
    p.add_argument("--road-length", type=float, default=None, help="finish line in m")
    p.add_argument("--dt", type=float, default=None, help="seconds of race per step")
    p.add_argument("--group-radius", type=float, default=None,
                   help="longitudinal pack radius in m")
    args = p.parse_args()

    base = {k: v for k, v in (("road_length", args.road_length), ("dt", args.dt),
                              ("group_radius", args.group_radius)) if v is not None}

    methods = ["morris", "sobol"] if args.method == "both" else [args.method]
    for method in methods:
        run_dump(method, args.samples, args.generations, args.max_steps,
                 args.processes, args.out_dir, args.parquet, base)


if __name__ == "__main__":
    main()
