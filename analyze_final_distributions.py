"""Deeper analysis of final strategy-parameter distributions across replications.

Reads the per-replication CSVs already produced by run_batch_learning.py
(`replication_XXX.csv`, one row per generation) and produces:

  1. summary_statistics.csv    mean/std/min/max/median per coefficient, across
                                replications (using each replication's final
                                generation row)
  2. boxplots.png               one boxplot per coefficient (alpha/beta/gamma/
                                delta for coop/leave/follow), final generation
                                values across all replications
  3. convergence_trajectories.png
                                mean +/- std band per coefficient across
                                generations, averaged over all replications
                                (shows *how* convergence happens, not just the
                                end state)
  4. correlation_heatmap.png    Pearson correlation between final coefficients,
                                across replications (do some parameters move
                                together?)

Does not re-run any simulation: pure post-hoc analysis of existing CSVs.

Usage:
    uv run python analyze_final_distributions.py --input-dir data/batch_learning --replications 120
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*'labels' parameter of boxplot.*")


def load_replication_histories(
    input_dir: Path, num_replications: int
) -> list[pd.DataFrame]:
    """Load each replication_XXX.csv (full per-generation history)."""
    histories = []
    for seed in range(num_replications):
        path = input_dir / f"replication_{seed:03d}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        histories.append(df)
    return histories


def _coeff_mean_columns(df: pd.DataFrame) -> list[str]:
    """Column names like 'coop.alpha_mean', 'leave.beta_mean', etc."""
    return [
        c
        for c in df.columns
        if "." in c and c.endswith("_mean") and not c.endswith("_std_mean")
    ]


def build_final_values_df(histories: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per replication: the final generation's coefficient means."""
    if not histories:
        return pd.DataFrame()
    coeff_cols = _coeff_mean_columns(histories[0])
    rows = []
    for df in histories:
        last = df.iloc[-1]
        row = {"seed": int(last["seed"])}
        for col in coeff_cols:
            row[col] = float(last[col])
        rows.append(row)
    return pd.DataFrame(rows)


def compute_summary_statistics(final_df: pd.DataFrame) -> pd.DataFrame:
    coeff_cols = [c for c in final_df.columns if c != "seed"]
    rows = []
    for col in coeff_cols:
        values = final_df[col].dropna().to_numpy()
        if len(values) == 0:
            continue
        rows.append(
            {
                "parameter": col,
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows).sort_values("parameter").reset_index(drop=True)


def plot_boxplots(final_df: pd.DataFrame, plot_dir: Path) -> Path:
    coeff_cols = sorted(c for c in final_df.columns if c != "seed")
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(coeff_cols)), 6))
    data = [final_df[c].dropna().to_numpy() for c in coeff_cols]
    bp = ax.boxplot(data, labels=coeff_cols, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("tab:blue")
        patch.set_alpha(0.5)
    ax.set_xticklabels(coeff_cols, rotation=45, ha="right")
    ax.set_ylabel("Final coefficient value")
    ax.set_title(f"Final parameter distributions across {len(final_df)} replications")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = plot_dir / "boxplots.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_convergence_trajectories(
    histories: list[pd.DataFrame], plot_dir: Path
) -> Path:
    coeff_cols = _coeff_mean_columns(histories[0])
    # Align on generation index; replications might have different lengths if
    # any run was cut short, so truncate to the shortest common length.
    min_len = min(len(df) for df in histories)
    stacked = {
        col: np.stack([df[col].to_numpy()[:min_len] for df in histories])
        for col in coeff_cols
    }

    cols = 4
    rows = (len(coeff_cols) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 3.4 * rows), squeeze=False
    )
    generations = np.arange(min_len)

    for idx, col in enumerate(coeff_cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        values = stacked[col]  # shape (n_replications, n_generations)
        mean_traj = values.mean(axis=0)
        std_traj = values.std(axis=0)
        ax.plot(generations, mean_traj, color="tab:blue", lw=1.5)
        ax.fill_between(
            generations,
            mean_traj - std_traj,
            mean_traj + std_traj,
            color="tab:blue",
            alpha=0.25,
        )
        ax.set_title(col, fontsize=9)
        ax.grid(alpha=0.25)

    for j in range(len(coeff_cols), rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.suptitle(
        f"Convergence trajectories (mean ± std across {len(histories)} replications)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = plot_dir / "convergence_trajectories.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_correlation_heatmap(final_df: pd.DataFrame, plot_dir: Path) -> Path:
    coeff_cols = sorted(c for c in final_df.columns if c != "seed")
    corr = final_df[coeff_cols].corr().to_numpy()

    fig, ax = plt.subplots(
        figsize=(max(8, 0.6 * len(coeff_cols)), max(7, 0.6 * len(coeff_cols)))
    )
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(coeff_cols)))
    ax.set_yticks(range(len(coeff_cols)))
    ax.set_xticklabels(coeff_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(coeff_cols, fontsize=8)
    for i in range(len(coeff_cols)):
        for j in range(len(coeff_cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation between final coefficients (across replications)")
    fig.tight_layout()
    out = plot_dir / "correlation_heatmap.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep analysis of final-parameter distributions and convergence trajectories."
    )
    parser.add_argument("--input-dir", default="data/batch_learning")
    parser.add_argument("--replications", type=int, default=120)
    parser.add_argument(
        "--output-dir", default=None, help="default: <input-dir>/analysis"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = load_replication_histories(input_dir, args.replications)
    if not histories:
        raise SystemExit(f"No replication_*.csv files found in {input_dir}")
    print(f"Loaded {len(histories)} replication histories.")

    final_df = build_final_values_df(histories)
    final_df.to_csv(output_dir / "final_values_per_replication.csv", index=False)

    summary = compute_summary_statistics(final_df)
    summary.to_csv(output_dir / "summary_statistics.csv", index=False)
    print("\nSummary statistics (final generation, across replications):")
    print(summary.to_string(index=False))

    p1 = plot_boxplots(final_df, output_dir)
    p2 = plot_convergence_trajectories(histories, output_dir)
    p3 = plot_correlation_heatmap(final_df, output_dir)

    print(f"\nSaved: {output_dir / 'final_values_per_replication.csv'}")
    print(f"Saved: {output_dir / 'summary_statistics.csv'}")
    print(f"Saved plot: {p1}")
    print(f"Saved plot: {p2}")
    print(f"Saved plot: {p3}")


if __name__ == "__main__":
    main()
