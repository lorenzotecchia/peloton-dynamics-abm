"""Standalone plotting script for generation-0 vs generation-N comparison.

Reads the `comparison_summary.csv` produced by collect_generation_comparison.py
and produces comparison plots, without needing to re-run any simulation.

Usage:
    uv run python plot_generation_comparison.py --input-dir data/generation_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_LABELS = {
    "race_duration_steps": "Race duration (steps)",
    "avg_stamina": "Mean stamina (W'/W_full)",
    "pct_time_group": "% time in group",
    "pct_time_breakaway": "% time breakaway/solo",
    "pct_time_isolated": "% time isolated",
    "avg_num_groups": "Mean # groups (pack fragmentation)",
    "avg_num_groups_pct": "Mean # groups (% of n_agents)",
    "pct_finished": "% riders finished",
    "mean_finish_step": "Mean finish step (finishers only)",
}

SHRINKED_METRIC_LABELS = {
    "pct_time_group": "% time in group",
    "avg_num_groups": "Mean # groups (pack fragmentation)",
    "mean_finish_step": "Mean finish step (finishers only)",
}


def plot_generation_comparison(df: pd.DataFrame, plot_dir: Path) -> list[Path]:
    """Paired boxplot + spaghetti-lines per metric: one panel per metric,
    gen0 vs genlast, with one thin line per replication connecting its two
    values so the per-replication shift is visible alongside the population
    distributions.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        m
        for m in METRIC_LABELS
        if f"gen0_{m}" in df.columns and f"genlast_{m}" in df.columns
    ]
    if not metrics:
        raise SystemExit(
            "No matching gen0_/genlast_ metric columns found in the summary CSV."
        )

    cols = 3
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.0 * cols, 4.2 * rows), squeeze=False
    )

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        gen0_vals = df[f"gen0_{metric}"].to_numpy(dtype=float)
        last_vals = df[f"genlast_{metric}"].to_numpy(dtype=float)

        for g0, gl in zip(gen0_vals, last_vals):
            if np.isnan(g0) or np.isnan(gl):
                continue
            ax.plot([0, 1], [g0, gl], color="tab:gray", alpha=0.15, lw=0.8, zorder=1)

        bp = ax.boxplot(
            [gen0_vals[~np.isnan(gen0_vals)], last_vals[~np.isnan(last_vals)]],
            positions=[0, 1],
            widths=0.3,
            patch_artist=True,
            showfliers=False,
            zorder=2,
        )
        for patch, color in zip(bp["boxes"], ["tab:blue", "tab:orange"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Gen 0", "Gen last"])
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(alpha=0.25)

    for j in range(len(metrics), rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    out = plot_dir / "generation_comparison.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return [out]


def plot_metric_histograms(df: pd.DataFrame, plot_dir: Path) -> Path:
    """Overlaid histograms of gen0 vs genlast per metric (distribution shape,
    complementary to the boxplot view above)."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        m
        for m in METRIC_LABELS
        if f"gen0_{m}" in df.columns and f"genlast_{m}" in df.columns
    ]

    cols = 3
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.0 * cols, 3.6 * rows), squeeze=False
    )

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        gen0_vals = df[f"gen0_{metric}"].dropna().to_numpy()
        last_vals = df[f"genlast_{metric}"].dropna().to_numpy()

        bins = min(20, max(5, int(np.sqrt(max(len(gen0_vals), len(last_vals))))))
        ax.hist(gen0_vals, bins=bins, alpha=0.5, label="Gen 0", color="tab:blue")
        ax.hist(last_vals, bins=bins, alpha=0.5, label="Gen last", color="tab:orange")
        ax.set_title(METRIC_LABELS[metric])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    for j in range(len(metrics), rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    out = plot_dir / "generation_histograms.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def print_summary(df: pd.DataFrame) -> None:
    metrics = [
        m
        for m in METRIC_LABELS
        if f"gen0_{m}" in df.columns and f"genlast_{m}" in df.columns
    ]
    print("\nGeneration 0 vs Generation last (mean across replications):")
    for metric in metrics:
        g0 = df[f"gen0_{metric}"].mean()
        gl = df[f"genlast_{metric}"].mean()
        print(
            f"  {METRIC_LABELS[metric]:38s} gen0={g0:9.3f}  genlast={gl:9.3f}  delta={gl - g0:+9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot generation-0 vs generation-last race-dynamics comparisons "
        "from an existing comparison_summary.csv."
    )
    parser.add_argument(
        "--input-dir",
        default="data/generation_comparison",
        help="Directory containing comparison_summary.csv (output of collect_generation_comparison.py)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save plots (default: <input-dir>/plots)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    summary_path = input_dir / "comparison_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Could not find {summary_path}")

    df = pd.read_csv(summary_path)
    plot_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"

    box_plots = plot_generation_comparison(df, plot_dir)
    hist_plot = plot_metric_histograms(df, plot_dir)
    print_summary(df)
    for p in box_plots:
        print(f"Saved plot: {p}")
    print(f"Saved plot: {hist_plot}")


if __name__ == "__main__":
    main()
