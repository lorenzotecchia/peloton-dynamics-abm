"""Plot Morris and Sobol GSA results from data/gsa_morris.csv and data/gsa_sobol.csv.

Morris figure  (plots/gsa_morris.png):
  2x2 grid, one subplot per metric.  Each subplot: mu* vs sigma scatter with
  labelled parameter points and horizontal error bars on mu*.

Sobol figure  (plots/gsa_sobol.png):
  2x2 grid, one subplot per metric.  Each subplot: grouped bar chart of S1 and
  ST with confidence-interval error bars.

Usage:
    python scripts/plot_gsa.py
    python scripts/plot_gsa.py --morris data/gsa_morris.csv \
                                --sobol  data/gsa_sobol.csv \
                                --out-dir plots
"""
import argparse
import os

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as exc:
    raise RuntimeError(
        "pandas, matplotlib and numpy are required. "
        "Install with 'pip install pandas matplotlib numpy'."
    ) from exc

PARAM_LABELS = {
    "recovery_rate": "recovery\nrate",
    "breakaway_speed_frac": "breakaway\nspeed frac",
    "utility_decay": "utility\ndecay",
    "k_s": "k_s",
}

METRIC_LABELS = {
    "MeanStamina": "Mean Stamina",
    "NumGroups": "Num Groups",
    "Breakaways": "Breakaways",
    "MeanExposure": "Mean Exposure",
}


def _axes_grid(metrics: list[str], figsize_per: tuple[float, float] = (4.5, 3.8)):
    n = len(metrics)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False)
    return fig, axes, nrows, ncols


def plot_morris(df: pd.DataFrame, out_path: str) -> None:
    metrics = df["metric"].unique().tolist()
    fig, axes, nrows, ncols = _axes_grid(metrics)

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = df[df["metric"] == metric].copy()

        ax.errorbar(
            sub["mu_star"], sub["sigma"],
            xerr=sub["mu_star_conf"],
            fmt="o", ms=7, capsize=4, color="steelblue",
            ecolor="steelblue", elinewidth=1.2,
        )

        for _, row in sub.iterrows():
            label = PARAM_LABELS.get(row["param"], row["param"])
            ax.annotate(
                label,
                xy=(row["mu_star"], row["sigma"]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8,
            )

        # reference line sigma = mu* (linear model boundary)
        xlim = ax.get_xlim()
        xmax = max(sub["mu_star"].max() * 1.3, 1e-9)
        xs = np.linspace(0, xmax, 100)
        ax.plot(xs, xs, "k--", lw=0.8, alpha=0.4, label="σ = μ*")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_xlabel("μ* (mean absolute elementary effect)", fontsize=9)
        ax.set_ylabel("σ (std of elementary effects)", fontsize=9)
        ax.grid(alpha=0.3)

    # hide unused axes
    for j in range(len(metrics), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle("Morris Sensitivity Analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote Morris plot to {out_path}")


def plot_sobol(df: pd.DataFrame, out_path: str) -> None:
    metrics = df["metric"].unique().tolist()
    fig, axes, nrows, ncols = _axes_grid(metrics)

    params = df["param"].unique().tolist()
    x = np.arange(len(params))
    w = 0.35

    colors = {"S1": "#4878cf", "ST": "#d65f5f"}

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = df[df["metric"] == metric].set_index("param").reindex(params)

        ax.bar(x - w / 2, sub["S1"],  width=w, label="S1 (first-order)",
               color=colors["S1"], alpha=0.85,
               yerr=sub["S1_conf"], capsize=4, error_kw={"elinewidth": 1.2})
        ax.bar(x + w / 2, sub["ST"],  width=w, label="ST (total-order)",
               color=colors["ST"], alpha=0.85,
               yerr=sub["ST_conf"], capsize=4, error_kw={"elinewidth": 1.2})

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [PARAM_LABELS.get(p, p) for p in params], fontsize=8
        )
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_ylabel("Sensitivity index", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(metrics), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle("Sobol Sensitivity Analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote Sobol plot to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Morris and Sobol GSA results")
    p.add_argument("--morris",   default="data/gsa_morris.csv")
    p.add_argument("--sobol",    default="data/gsa_sobol.csv")
    p.add_argument("--out-dir",  default="plots")
    args = p.parse_args()

    morris_df = pd.read_csv(args.morris)
    sobol_df  = pd.read_csv(args.sobol)

    plot_morris(morris_df, os.path.join(args.out_dir, "gsa_morris.png"))
    plot_sobol(sobol_df,   os.path.join(args.out_dir, "gsa_sobol.png"))


if __name__ == "__main__":
    main()
