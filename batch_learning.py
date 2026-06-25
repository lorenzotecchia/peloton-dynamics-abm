"""Run batch learning and produce summary plots.

Outputs:
1. Distribution of final mean strategy parameters across replications.
2. Learning-vs-skill plot: per-agent parameter change (gen0 -> genN) vs w_max10.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peloton.config import PelotonConfig
from peloton.evolution import _coeff_stats, _utility_stats, evolve
from peloton.model import PelotonModel
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="mesa")


def _flatten_coeffs(coeffs: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, params in coeffs.items():
        for param, value in params.items():
            flat[f"{key}.{param}"] = float(value)
    return flat


def _coeff_change(initial: dict, final: dict) -> dict[str, float]:
    init_flat = _flatten_coeffs(initial)
    final_flat = _flatten_coeffs(final)
    keys = [k for k in init_flat if k in final_flat]
    if not keys:
        return {"change_l2": 0.0, "change_mean_abs": 0.0, "change_rel_mean_abs": 0.0}
    abs_diffs = [abs(final_flat[k] - init_flat[k]) for k in keys]
    rel_diffs = [
        abs(final_flat[k] - init_flat[k]) / (abs(init_flat[k]) + 1e-9) for k in keys
    ]
    return {
        "change_l2": float(math.sqrt(sum(d * d for d in abs_diffs))),
        "change_mean_abs": float(statistics.mean(abs_diffs)),
        "change_rel_mean_abs": float(statistics.mean(rel_diffs)),
    }


def run_replication(
    seed: int, generations: int, max_steps: int, output_dir: Path
) -> dict | None:
    """Run one replication in-process and save generation + agent learning data."""
    cfg = PelotonConfig(seed=seed)

    import random

    rng = random.Random(seed)
    physiology = [
        max(50.0, rng.gauss(cfg.w_max10_mean, cfg.w_max10_std))
        for _ in range(cfg.n_agents)
    ]

    history: list[dict] = []
    population: list[dict] | None = None
    initial_agents: list[dict] = []
    final_coeffs: list[dict] = []

    for gen in range(generations):
        model = PelotonModel(config=cfg, population=population, physiology=None)

        if gen == 0:
            initial_agents = [
                {
                    "slot": i,
                    "w_max10": float(r.w_max10),
                    "coeffs": copy.deepcopy(r.coeffs),
                }
                for i, r in enumerate(model.riders)
            ]

        for _ in range(max_steps):
            if not model.running:
                break
            model.step()

        entry = {"generation": gen, "n_finished": model.n_finished}
        entry.update(model.datacollector.get_model_vars_dataframe().mean().to_dict())
        entry.update(_coeff_stats(model.riders))

        evolve(model.riders, model)
        entry.update(_utility_stats(model.riders))
        history.append(entry)

        population = [copy.deepcopy(r.coeffs) for r in model.riders]
        if gen == generations - 1:
            final_coeffs = [copy.deepcopy(r.coeffs) for r in model.riders]

    if not history or not initial_agents or not final_coeffs:
        return None

    rep_csv = output_dir / f"replication_{seed:03d}.csv"
    pd.DataFrame(history).to_csv(rep_csv, index=False)

    agent_rows: list[dict] = []
    for row in initial_agents:
        slot = row["slot"]
        if slot >= len(final_coeffs):
            continue
        delta = _coeff_change(row["coeffs"], final_coeffs[slot])
        agent_rows.append(
            {
                "seed": seed,
                "agent_slot": slot,
                "w_max10": row["w_max10"],
                **delta,
            }
        )
    agent_csv = output_dir / f"replication_{seed:03d}_agent_learning.csv"
    pd.DataFrame(agent_rows).to_csv(agent_csv, index=False)

    return {
        "seed": seed,
        "file": str(rep_csv),
        "agent_file": str(agent_csv),
        "generations": len(history),
        "final_row": history[-1],
    }


def run_all_replications(
    num_replications: int = 100,
    generations: int = 100,
    max_steps: int = 400,
    output_dir: str = "data/batch_learning",
) -> list[dict]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    replications: list[dict] = []
    for seed in range(num_replications):
        print(f"Running {seed + 1}/{num_replications}...", end="\r")
        rep = run_replication(seed, generations, max_steps, out)
        if rep:
            replications.append(rep)
    print(f"Completed {len(replications)}/{num_replications} replications.")
    return replications


def load_existing_replications(output_dir: Path, num_replications: int) -> list[dict]:
    replications: list[dict] = []
    for seed in range(num_replications):
        rep_csv = output_dir / f"replication_{seed:03d}.csv"
        agent_csv = output_dir / f"replication_{seed:03d}_agent_learning.csv"
        if not rep_csv.exists():
            continue
        df = pd.read_csv(rep_csv)
        replications.append(
            {
                "seed": seed,
                "file": str(rep_csv),
                "agent_file": str(agent_csv) if agent_csv.exists() else None,
                "generations": len(df),
                "final_row": df.iloc[-1].to_dict() if len(df) > 0 else {},
            }
        )
    return replications


def aggregate_final_coefficients(replications: list[dict]) -> dict:
    if not replications or not replications[0].get("final_row"):
        return {}
    sample = replications[0]["final_row"]
    coeff_keys = [k for k in sample if "." in k and k.endswith("_mean")]
    agg: dict[str, dict] = {}
    for key in coeff_keys:
        values = [
            float(rep["final_row"][key])
            for rep in replications
            if key in rep.get("final_row", {}) and not pd.isna(rep["final_row"][key])
        ]
        if not values:
            continue
        agg[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "values": values,
        }
    return agg


def _collect_agent_learning(replications: list[dict]) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for rep in replications:
        agent_file = rep.get("agent_file")
        if not agent_file:
            continue
        p = Path(agent_file)
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def plot_learning_vs_skill(agent_df: pd.DataFrame, plot_dir: Path) -> Path | None:
    if agent_df.empty:
        return None

    x = agent_df["w_max10"].to_numpy()
    y = agent_df["change_mean_abs"].to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1]) if len(agent_df) > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(x, y, s=12, alpha=0.25)
    if len(agent_df) > 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        xx = np.linspace(x.min(), x.max(), 200)
        axes[0].plot(xx, slope * xx + intercept, color="tab:red", lw=2)
    axes[0].set_title(f"Agent learning vs skill (r={corr:.3f})")
    axes[0].set_xlabel("Initial w_max10")
    axes[0].set_ylabel("Mean |Δ parameter| (gen0 -> genN)")
    axes[0].grid(alpha=0.3)

    bins = min(10, max(3, int(np.sqrt(len(agent_df)))))
    binned = agent_df.copy()
    binned["w_bin"] = pd.qcut(binned["w_max10"], q=bins, duplicates="drop")
    grouped = (
        binned.groupby("w_bin", observed=False)["change_mean_abs"]
        .agg(["mean", "std"])
        .reset_index()
    )
    mids = np.array([interval.mid for interval in grouped["w_bin"]])
    yerr = grouped["std"].fillna(0.0).to_numpy()
    axes[1].errorbar(mids, grouped["mean"], yerr=yerr, fmt="o-", capsize=3)
    axes[1].set_title("Average learning by skill bin")
    axes[1].set_xlabel("Initial w_max10 (bin midpoint)")
    axes[1].set_ylabel("Mean |Δ parameter|")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = plot_dir / "learning_vs_skill.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_final_mean_distributions(
    coeff_aggregates: dict, plot_dir: Path
) -> Path | None:
    if not coeff_aggregates:
        return None
    keys = sorted(coeff_aggregates.keys())
    n = len(keys)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False
    )
    for i, key in enumerate(keys):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        values = coeff_aggregates[key]["values"]
        ax.hist(values, bins=18, alpha=0.8, color="tab:blue")
        ax.set_title(key)
        ax.grid(alpha=0.25)
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    out = plot_dir / "final_mean_parameter_distributions.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def save_report(
    replications: list[dict],
    coeff_aggregates: dict,
    agent_df: pd.DataFrame,
    output_dir: Path,
    plots: list[Path],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"analysis_report_{timestamp}.json"

    learning_summary = {}
    if not agent_df.empty:
        corr = float(
            np.corrcoef(agent_df["w_max10"], agent_df["change_mean_abs"])[0, 1]
        )
        learning_summary = {
            "n_agents_total": int(len(agent_df)),
            "mean_change_mean_abs": float(agent_df["change_mean_abs"].mean()),
            "mean_change_l2": float(agent_df["change_l2"].mean()),
            "corr_w_max10_vs_change": corr,
        }

    payload = {
        "timestamp": timestamp,
        "num_replications": len(replications),
        "coefficient_distributions": {
            k: {kk: vv for kk, vv in v.items() if kk != "values"}
            for k, v in coeff_aggregates.items()
        },
        "learning_vs_skill_summary": learning_summary,
        "plots": [str(p) for p in plots if p is not None],
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return report_file


def print_results(coeff_aggregates: dict, agent_df: pd.DataFrame) -> None:
    print("\nFinal mean parameter distributions (across replications):")
    for key, stats in sorted(coeff_aggregates.items()):
        print(
            f"  {key:28s} "
            f"μ={stats['mean']:8.4f} σ={stats['std']:8.4f} "
            f"[{stats['min']:8.4f}, {stats['max']:8.4f}]"
        )
    if not agent_df.empty:
        corr = float(
            np.corrcoef(agent_df["w_max10"], agent_df["change_mean_abs"])[0, 1]
        )
        print("\nLearning vs skill:")
        print(f"  mean |Δ parameter| = {agent_df['change_mean_abs'].mean():.4f}")
        print(f"  corr(w_max10, |Δ parameter|) = {corr:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run replications and produce learning-vs-skill + distribution plots."
    )
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output-dir", default="data/batch_learning")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_run:
        replications = load_existing_replications(output_dir, args.replications)
    else:
        replications = run_all_replications(
            num_replications=args.replications,
            generations=args.generations,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
        )

    coeff_aggregates = aggregate_final_coefficients(replications)
    agent_df = _collect_agent_learning(replications)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    p1 = plot_learning_vs_skill(agent_df, plot_dir)
    p2 = plot_final_mean_distributions(coeff_aggregates, plot_dir)

    report_file = save_report(
        replications, coeff_aggregates, agent_df, output_dir, [p1, p2]
    )
    print_results(coeff_aggregates, agent_df)
    print(f"\nSaved report: {report_file}")
    if p1:
        print(f"Saved plot:   {p1}")
    if p2:
        print(f"Saved plot:   {p2}")


if __name__ == "__main__":
    main()
