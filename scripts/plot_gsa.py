"""Plot Morris and Sobol GSA results.

Discovers the most recent morris and sobol data directories under ``data/``
(or accepts explicit paths), then generates:
  - Morris: mu_star vs sigma scatter + ranked mu_star bar chart
  - Sobol:  S1/ST bar chart + S2 heatmap

Run:
    uv run python scripts/plot_gsa.py
    uv run python scripts/plot_gsa.py --morris-dir data/... --sobol-dir data/...
    uv run python scripts/plot_gsa.py --out-dir plots/my_gsa
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_LABELS = {
    "sum_finish_time": "Total finish time",
    "n_finished": "Riders finished",
    "sum_stamina_spent": "Total stamina spent",
}

# Outputs where lower values ≡ better (only used for axis labelling)
OUTPUTS = list(OUTPUT_LABELS.keys())

# Morris: reference lines for sigma/mu_star ratio
MORRIS_RATIO_LINES = [0.5, 1.0, 2.0]

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _find_dir(data_root: Path, tag: str) -> Path | None:
    """Return most recent directory whose name ends with ``-<tag>``."""
    candidates = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and d.name.endswith(f"-{tag}")],
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None


def _load_csv(path: Path, required_cols: list[str]) -> pd.DataFrame | None:
    """Load CSV if it exists and has required columns; return None otherwise."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not all(c in df.columns for c in required_cols):
        return None
    return df


# ---------------------------------------------------------------------------
# Morris plots
# ---------------------------------------------------------------------------

def _morris_scatter(df: pd.DataFrame, output_key: str, ax: plt.Axes) -> None:
    """mu_star vs sigma scatter (elementary effects sensitivity plot)."""
    # Drop parameters that are entirely zero (no variation)
    active = df[(df["mu_star"] > 0) | (df["sigma"] > 0)].copy()
    if active.empty:
        ax.text(0.5, 0.5, "no sensitivity", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    ax.errorbar(
        active["mu_star"], active["sigma"],
        xerr=active["mu_star_conf"],
        fmt="o", ms=5, lw=1, alpha=0.8, color="steelblue",
        ecolor="lightsteelblue",
    )

    for _, row in active.iterrows():
        ax.annotate(
            row["parameter"],
            (row["mu_star"], row["sigma"]),
            xytext=(4, 2), textcoords="offset points",
            fontsize=7, alpha=0.85,
        )

    # sigma / mu_star ratio reference lines
    xlim_max = active["mu_star"].max() * 1.15 or 1.0
    x_ref = np.linspace(0, xlim_max, 200)
    for r in MORRIS_RATIO_LINES:
        ax.plot(x_ref, r * x_ref, "--", lw=0.8, color="gray", alpha=0.5,
                label=f"σ/μ* = {r}")

    ax.set_xlabel("μ* (mean absolute effect)")
    ax.set_ylabel("σ (std of effects)")
    ax.set_title(OUTPUT_LABELS.get(output_key, output_key))
    ax.legend(fontsize=7, framealpha=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def _morris_bar(df: pd.DataFrame, output_key: str, ax: plt.Axes) -> None:
    """Ranked bar chart of mu_star with confidence interval."""
    active = df[df["mu_star"] > 0].sort_values("mu_star", ascending=True).copy()
    if active.empty:
        ax.text(0.5, 0.5, "no sensitivity", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    y = range(len(active))
    ax.barh(y, active["mu_star"], xerr=active["mu_star_conf"],
            align="center", height=0.6, color="steelblue", alpha=0.8,
            error_kw={"elinewidth": 1, "capsize": 3})
    ax.set_yticks(list(y))
    ax.set_yticklabels(active["parameter"], fontsize=7)
    ax.set_xlabel("μ*")
    ax.set_title(OUTPUT_LABELS.get(output_key, output_key))
    ax.set_xlim(left=0)


def plot_morris(morris_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for key in OUTPUTS:
        path = morris_dir / f"morris_{key}.csv"
        df = _load_csv(path, ["parameter", "mu_star", "sigma", "mu_star_conf"])
        if df is not None:
            loaded[key] = df

    if not loaded:
        print(f"[morris] no data found in {morris_dir}")
        return

    n = len(loaded)

    # --- Scatter: mu_star vs sigma ---
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (key, df) in zip(axes, loaded.items()):
        _morris_scatter(df, key, ax)
    fig.suptitle("Morris GSA — elementary effects (μ* vs σ)", fontsize=12, fontweight="bold")
    path_scatter = out_dir / "morris_scatter.png"
    fig.savefig(path_scatter, dpi=180)
    plt.close(fig)
    print(f"[morris] scatter saved → {path_scatter}")

    # --- Bar: ranked mu_star ---
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 7), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (key, df) in zip(axes, loaded.items()):
        _morris_bar(df, key, ax)
    fig.suptitle("Morris GSA — parameter importance (μ*)", fontsize=12, fontweight="bold")
    path_bar = out_dir / "morris_bar.png"
    fig.savefig(path_bar, dpi=180)
    plt.close(fig)
    print(f"[morris] bar saved    → {path_bar}")


# ---------------------------------------------------------------------------
# Sobol plots
# ---------------------------------------------------------------------------

def _sobol_bar(df: pd.DataFrame, output_key: str, ax: plt.Axes) -> None:
    """Side-by-side S1 / ST bar chart with confidence intervals."""
    df = df.dropna(subset=["S1", "ST"])
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    df = df.sort_values("ST", ascending=False)
    params = df["parameter"].tolist()
    x = np.arange(len(params))
    w = 0.35

    ax.bar(x - w / 2, df["S1"].clip(lower=0), w,
           yerr=df["S1_conf"], capsize=3, label="S1 (first-order)",
           color="steelblue", alpha=0.85, error_kw={"elinewidth": 1})
    ax.bar(x + w / 2, df["ST"].clip(lower=0), w,
           yerr=df["ST_conf"], capsize=3, label="ST (total-order)",
           color="coral", alpha=0.85, error_kw={"elinewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Sensitivity index")
    ax.set_title(OUTPUT_LABELS.get(output_key, output_key))
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())


def _sobol_s2_heatmap(df: pd.DataFrame, output_key: str, ax: plt.Axes) -> None:
    """S2 matrix heatmap (symmetric)."""
    params = sorted(set(df["param_i"].tolist() + df["param_j"].tolist()))
    n = len(params)
    idx = {p: i for i, p in enumerate(params)}
    mat = np.full((n, n), np.nan)
    for _, row in df.iterrows():
        i, j = idx[row["param_i"]], idx[row["param_j"]]
        mat[i, j] = row["S2"]
        mat[j, i] = row["S2"]

    vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(params, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(params, fontsize=8)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="S2")
    ax.set_title(OUTPUT_LABELS.get(output_key, output_key))


def plot_sobol(sobol_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    s1st_loaded, s2_loaded = {}, {}
    for key in OUTPUTS:
        df_s1st = _load_csv(sobol_dir / f"sobol_S1_ST_{key}.csv",
                             ["parameter", "S1", "ST"])
        if df_s1st is not None:
            s1st_loaded[key] = df_s1st

        df_s2 = _load_csv(sobol_dir / f"sobol_S2_{key}.csv",
                           ["param_i", "param_j", "S2"])
        if df_s2 is not None:
            s2_loaded[key] = df_s2

    if not s1st_loaded and not s2_loaded:
        print(f"[sobol] no data found in {sobol_dir}")
        return

    # --- S1/ST bar charts ---
    if s1st_loaded:
        n = len(s1st_loaded)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), constrained_layout=True)
        if n == 1:
            axes = [axes]
        for ax, (key, df) in zip(axes, s1st_loaded.items()):
            _sobol_bar(df, key, ax)
        fig.suptitle("Sobol GSA — first-order (S1) and total-order (ST) indices",
                     fontsize=12, fontweight="bold")
        path_bar = out_dir / "sobol_s1_st_bar.png"
        fig.savefig(path_bar, dpi=180)
        plt.close(fig)
        print(f"[sobol] S1/ST bar saved   → {path_bar}")

    # --- S2 heatmaps ---
    if s2_loaded:
        n = len(s2_loaded)
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), constrained_layout=True)
        if n == 1:
            axes = [axes]
        for ax, (key, df) in zip(axes, s2_loaded.items()):
            _sobol_s2_heatmap(df, key, ax)
        fig.suptitle("Sobol GSA — second-order interaction indices (S2)",
                     fontsize=12, fontweight="bold")
        path_s2 = out_dir / "sobol_s2_heatmap.png"
        fig.savefig(path_s2, dpi=180)
        plt.close(fig)
        print(f"[sobol] S2 heatmap saved  → {path_s2}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Morris and Sobol GSA results.")
    parser.add_argument("--morris-dir", type=Path, default=None,
                        help="Path to morris data directory (auto-discovered if omitted).")
    parser.add_argument("--sobol-dir", type=Path, default=None,
                        help="Path to sobol data directory (auto-discovered if omitted).")
    parser.add_argument("--out-dir", type=Path, default=Path("plots/gsa"),
                        help="Output directory for plots (default: plots/gsa).")
    args = parser.parse_args()

    data_root = Path("data")

    morris_dir = args.morris_dir or _find_dir(data_root, "morris")
    sobol_dir = args.sobol_dir or _find_dir(data_root, "sobol")

    if morris_dir is None and sobol_dir is None:
        parser.error("No morris or sobol data directories found under data/.")

    if morris_dir:
        print(f"[morris] using {morris_dir}")
        plot_morris(morris_dir, args.out_dir)
    else:
        print("[morris] no directory found, skipping.")

    if sobol_dir:
        print(f"[sobol]  using {sobol_dir}")
        plot_sobol(sobol_dir, args.out_dir)
    else:
        print("[sobol]  no directory found, skipping.")

    print(f"\nAll plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
