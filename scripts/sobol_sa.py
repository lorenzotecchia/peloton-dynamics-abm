"""Sobol global sensitivity analysis for the peloton ABM.

Computes first-order (S1), second-order (S2), and total-order (ST) Sobol
sensitivity indices for 8 physics/structural parameters against 5 race-level
output quantities.  Uses Saltelli's extension of the Sobol sequence so that
all three index families can be estimated from a single sample matrix.

    uv run python scripts/sobol_sa.py              # default: N=256, ~4 608 runs
    uv run python scripts/sobol_sa.py --n 1024     # Snellius-scale: ~18 432 runs
    uv run python scripts/sobol_sa.py --no-plot    # CSV only (headless HPC node)

Total model evaluations = N * (2*D + 2) where D = 8 parameters.
"""

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on HPC
import matplotlib.pyplot as plt

try:
    from SALib.sample import sobol as sobol_sample  # SALib >= 1.5 preferred API
    from SALib.analyze import sobol as sobol_analyze
except ImportError:
    sys.exit("SALib >= 1.5 is required.  Install with: uv add salib")

from peloton.model import PelotonModel

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------
PARAM_NAMES = [
    "k_s",
    "draft_coefficient",
    "breakaway_speed_frac",
    "k_aero",
    "c_roll",
    "recovery_rate",
    "cp_fraction",
    "group_radius",
]

PARAM_BOUNDS = [
    [0.70, 1.00],   # k_s: pack-speed coefficient (Martins 2013: [0.7, 1])
    [0.50, 0.80],   # draft_coefficient: full-shelter drag factor (default 0.62)
    [0.85, 1.05],   # breakaway_speed_frac: solo speed / s_m  (default 0.95)
    [0.54, 1.26],   # k_aero: aerodynamic coefficient  (default 0.9, ±40 %)
    [1.50, 4.50],   # c_roll: rolling resistance  (default 3.0, ±50 %)
    [0.02, 0.30],   # recovery_rate: W' recovery multiplier below CP (default 0.1)
    [0.55, 0.85],   # cp_fraction: critical power / W_max10 (default 0.70)
    [1.00, 6.00],   # group_radius: same-group distance in metres (default 3.0)
]

PROBLEM = {
    "num_vars": len(PARAM_NAMES),
    "names": PARAM_NAMES,
    "bounds": PARAM_BOUNDS,
}

# Short labels for axis tick marks
PARAM_TICK = [
    "k_s",
    "draft_coeff",
    "brkwy_frac",
    "k_aero",
    "c_roll",
    "recovery_r",
    "cp_frac",
    "grp_radius",
]

# ---------------------------------------------------------------------------
# Output quantities of interest (QoIs)
# ---------------------------------------------------------------------------
QOI_NAMES = [
    "race_steps",
    "n_finished",
    "mean_stamina",
    "num_groups",
    "mean_exposure",
]

QOI_LABELS = {
    "race_steps":    "Race finish time (steps)",
    "n_finished":    "Riders finished (count)",
    "mean_stamina":  "Mean final stamina (W'/W_full)",
    "num_groups":    "Group fragmentation (# groups)",
    "mean_exposure": "Mean wind exposure",
}

# ---------------------------------------------------------------------------
# Module-level runner — must be at module scope for multiprocessing pickling
# ---------------------------------------------------------------------------

def _run_single(args: tuple) -> dict | None:
    """Run one single race and return the 5 QoI scalars."""
    sample, seed, max_steps = args
    overrides = dict(zip(PARAM_NAMES, sample.tolist()))
    overrides["seed"] = seed
    try:
        model = PelotonModel(**overrides)
        for _ in range(max_steps):
            if not model.running:
                break
            model.step()
        dc = model.datacollector.get_model_vars_dataframe()
        final = dc.iloc[-1]
        return {
            "race_steps":    model.steps,
            "n_finished":    model.n_finished,
            "mean_stamina":  float(final["MeanStamina"]),
            "num_groups":    float(final["NumGroups"]),
            "mean_exposure": float(final["MeanExposure"]),
        }
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return None

# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_samples(
    param_values: np.ndarray,
    max_steps: int,
    processes: int,
) -> pd.DataFrame:
    n = len(param_values)
    args = [(param_values[i], i, max_steps) for i in range(n)]
    results: list[dict | None] = [None] * n

    with ProcessPoolExecutor(max_workers=processes) as pool:
        futures = {pool.submit(_run_single, a): i for i, a in enumerate(args)}
        done = 0
        report_every = max(1, n // 20)
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            done += 1
            if done % report_every == 0 or done == n:
                print(f"  {done}/{n} ({100 * done // n}%)", flush=True)

    failed = sum(1 for r in results if r is None)
    if failed:
        print(f"  WARNING: {failed}/{n} runs failed — NaN substituted")

    rows = []
    for r in results:
        if r is None:
            rows.append({q: float("nan") for q in QOI_NAMES})
        else:
            rows.append(r)
    return pd.DataFrame(rows, columns=QOI_NAMES)

# ---------------------------------------------------------------------------
# Sobol analysis
# ---------------------------------------------------------------------------

def analyze(
    param_values: np.ndarray,
    Y_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute S1, S2, ST for every QoI.  Returns (s1_st_df, s2_df)."""
    s1_st_rows = []
    s2_rows = []

    for qoi in QOI_NAMES:
        Y = Y_df[qoi].values.copy()
        n_nan = np.sum(~np.isfinite(Y))
        if n_nan > len(Y) * 0.1:
            print(f"  SKIP {qoi}: {n_nan}/{len(Y)} non-finite values (>10%)")
            continue
        if n_nan:
            Y = np.where(np.isfinite(Y), Y, np.nanmean(Y))   # impute rare failures

        Si = sobol_analyze.analyze(
            PROBLEM, Y,
            calc_second_order=True,
            conf_level=0.95,
            print_to_console=False,
        )

        for i, pname in enumerate(PARAM_NAMES):
            s1_st_rows.append({
                "qoi":     qoi,
                "param":   pname,
                "S1":      Si["S1"][i],
                "S1_conf": Si["S1_conf"][i],
                "ST":      Si["ST"][i],
                "ST_conf": Si["ST_conf"][i],
            })

        for i in range(len(PARAM_NAMES)):
            for j in range(i + 1, len(PARAM_NAMES)):
                s2_rows.append({
                    "qoi":     qoi,
                    "param_i": PARAM_NAMES[i],
                    "param_j": PARAM_NAMES[j],
                    "S2":      Si["S2"][i, j],
                    "S2_conf": Si["S2_conf"][i, j],
                })

    return pd.DataFrame(s1_st_rows), pd.DataFrame(s2_rows)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_s1_st(s1_st_df: pd.DataFrame, out_path: str) -> None:
    """Grouped bar chart (S1 and ST side by side) for each QoI."""
    qois = [q for q in QOI_NAMES if q in s1_st_df["qoi"].values]
    n_qoi = len(qois)
    if n_qoi == 0:
        return

    fig, axes = plt.subplots(n_qoi, 1, figsize=(13, 3.5 * n_qoi), squeeze=False)

    x = np.arange(len(PARAM_NAMES))
    w = 0.35

    for row, qoi in enumerate(qois):
        ax = axes[row][0]
        sub = (
            s1_st_df[s1_st_df["qoi"] == qoi]
            .set_index("param")
            .reindex(PARAM_NAMES)
        )

        ax.bar(
            x - w / 2, sub["S1"], w,
            label="S1 (first-order)",
            color="steelblue", alpha=0.85,
            yerr=sub["S1_conf"], capsize=3,
            error_kw={"elinewidth": 0.8, "ecolor": "navy"},
        )
        ax.bar(
            x + w / 2, sub["ST"], w,
            label="ST (total-order)",
            color="coral", alpha=0.85,
            yerr=sub["ST_conf"], capsize=3,
            error_kw={"elinewidth": 0.8, "ecolor": "darkred"},
        )

        # ST - S1 gap (shaded) highlights higher-order interaction share
        s1 = sub["S1"].values
        st = sub["ST"].values
        for xi, (s1_v, st_v) in enumerate(zip(s1, st)):
            if np.isfinite(s1_v) and np.isfinite(st_v) and st_v > s1_v:
                ax.annotate(
                    "", xy=(xi + w / 2, st_v), xytext=(xi + w / 2, s1_v),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6,
                                   linestyle="dashed"),
                )

        ax.set_xticks(x)
        ax.set_xticklabels(PARAM_TICK, fontsize=8)
        ax.set_ylabel("Sobol index")
        ax.set_title(QOI_LABELS.get(qoi, qoi), fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5)
        vals = sub[["S1", "ST"]].values
        finite_vals = vals[np.isfinite(vals)]
        ymin = float(np.min(finite_vals)) if finite_vals.size else 0.0
        ymax = float(np.max(finite_vals)) if finite_vals.size else 0.5
        ax.set_ylim(bottom=min(0, ymin - 0.02), top=max(0.05, ymax + 0.05))
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    fig.suptitle("Sobol Sensitivity: S1 (first-order) and ST (total-order)", fontsize=12, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  S1/ST plot  → {out_path}")


def plot_s2(s2_df: pd.DataFrame, out_path: str) -> None:
    """Heatmap grid of S2 second-order indices, one cell per QoI."""
    qois = [q for q in QOI_NAMES if q in s2_df["qoi"].values]
    n_qoi = len(qois)
    if n_qoi == 0:
        return

    D = len(PARAM_NAMES)
    ncols = min(3, n_qoi)
    nrows = (n_qoi + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.8 * nrows),
        squeeze=False,
    )

    # Consistent colour scale across all QoIs
    s2_vals = s2_df["S2"].values
    vmax = max(float(np.nanpercentile(s2_vals, 95)), 0.01)

    for idx, qoi in enumerate(qois):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = s2_df[s2_df["qoi"] == qoi]

        mat = np.full((D, D), np.nan)
        for _, row in sub.iterrows():
            i = PARAM_NAMES.index(row["param_i"])
            j = PARAM_NAMES.index(row["param_j"])
            mat[i, j] = row["S2"]
            mat[j, i] = row["S2"]   # mirror for a readable square heatmap

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(D))
        ax.set_yticks(range(D))
        ax.set_xticklabels(PARAM_TICK, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(PARAM_TICK, fontsize=7)
        ax.set_title(QOI_LABELS.get(qoi, qoi), fontsize=9)

        for i in range(D):
            for j in range(D):
                if not np.isnan(mat[i, j]):
                    text_color = "white" if mat[i, j] > vmax * 0.6 else "black"
                    ax.text(
                        j, i, f"{mat[i, j]:.3f}",
                        ha="center", va="center",
                        fontsize=6, color=text_color,
                    )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="S2")

    for idx in range(n_qoi, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle("Sobol Sensitivity: S2 second-order indices", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  S2 heatmap  → {out_path}")


def print_summary(s1_st_df: pd.DataFrame, s2_df: pd.DataFrame) -> None:
    """Print ranked parameter importance to stdout."""
    if s1_st_df.empty:
        return
    print("\n── Mean Sobol indices across all QoIs (ranked by ST) ─────────────")
    agg = (
        s1_st_df.groupby("param")[["S1", "ST"]]
        .mean()
        .sort_values("ST", ascending=False)
    )
    agg["ST-S1 (interactions)"] = agg["ST"] - agg["S1"]
    print(agg.round(4).to_string())

    if not s2_df.empty:
        print("\n── Top 10 parameter pairs by mean S2 ─────────────────────────────")
        top = (
            s2_df.groupby(["param_i", "param_j"])["S2"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        print(top.round(4).to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n", type=int, default=256,
        help="Saltelli base sample size N. "
             "Total runs = N*(2*D+2) = N*18 for D=8. "
             "Default 256 → 4608 runs.",
    )
    parser.add_argument("--max-steps", type=int, default=400,
                        help="Max simulation steps per run (default 400).")
    parser.add_argument("--processes", type=int, default=os.cpu_count(),
                        help="Parallel worker processes (default: all CPUs).")
    parser.add_argument("--out-dir", default="plots/sobol",
                        help="Directory for output CSVs and plots.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the Saltelli sampler (default 0).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation (CSV outputs only).")
    args = parser.parse_args()

    D = PROBLEM["num_vars"]
    n_runs = args.n * (2 * D + 2)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Sobol SA — N={args.n}, D={D}, total_runs={n_runs}")
    print(f"Parameters : {PARAM_NAMES}")
    print(f"QoIs       : {QOI_NAMES}")
    print(f"Processes  : {args.processes}")
    print(f"Output dir : {args.out_dir}")

    # ------------------------------------------------------------------
    print(f"\n[1/4] Saltelli sampling (seed={args.seed}) …")
    param_values = sobol_sample.sample(
        PROBLEM, args.n, calc_second_order=True, seed=args.seed,
    )
    print(f"  Sample matrix: {param_values.shape}  ({param_values.shape[0]} runs)")

    # ------------------------------------------------------------------
    print("\n[2/4] Running simulations …")
    Y_df = run_samples(param_values, args.max_steps, args.processes)

    raw_path = os.path.join(args.out_dir, "sobol_raw.csv")
    pd.concat(
        [pd.DataFrame(param_values, columns=PARAM_NAMES), Y_df], axis=1
    ).to_csv(raw_path, index=False)
    print(f"  Raw results → {raw_path}")

    # ------------------------------------------------------------------
    print("\n[3/4] Computing Sobol indices …")
    s1_st_df, s2_df = analyze(param_values, Y_df)

    s1st_path = os.path.join(args.out_dir, "sobol_s1_st.csv")
    s2_path   = os.path.join(args.out_dir, "sobol_s2.csv")
    s1_st_df.to_csv(s1st_path, index=False)
    s2_df.to_csv(s2_path, index=False)
    print(f"  S1/ST table → {s1st_path}")
    print(f"  S2 table    → {s2_path}")

    # ------------------------------------------------------------------
    print("\n[4/4] Generating plots …")
    if args.no_plot:
        print("  --no-plot set; skipping.")
    else:
        plot_s1_st(s1_st_df, os.path.join(args.out_dir, "sobol_s1_st.png"))
        plot_s2(s2_df, os.path.join(args.out_dir, "sobol_s2.png"))

    print_summary(s1_st_df, s2_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
