"""Analyse learning output from ``main.py learn``.

    uv run python analyze.py learning.csv

Reads the per-generation trajectory CSV (coefficient mean/std, optionally with a
``seed`` column) and, if present, the sibling ``<stem>_riders.csv`` (final
per-rider records). Prints three diagnostics and saves plots next to the input:

  1. SELECTION      - is each coefficient driven (learning) or drifting (noise)?
                      Credible only when the direction agrees across seeds.
  2. DIFFERENTIATION- does the population spread into roles (std rises)?
  3. ABILITY        - do strong engines learn different strategies than weak ones?
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COEFFS = [
    f"{k}.{p}"
    for k in ("coop", "leave", "follow")
    for p in ("alpha", "beta", "gamma", "delta")
]


def _directionality(series: pd.Series):
    """Net displacement and net/path ratio: ~0 = random walk, |.|->1 = directed."""
    net = series.iloc[-1] - series.iloc[0]
    path = series.diff().abs().sum()
    return net, (net / path if path else 0.0)


def selection_table(df: pd.DataFrame) -> None:
    seeds = sorted(df["seed"].unique()) if "seed" in df else [None]
    print(f"\n== SELECTION & DIFFERENTIATION (over {len(seeds)} seed(s)) ==")
    print(
        f"{'coefficient':16}{'net_move':>10}{'direction':>11}{'seeds_agree':>12}{'std_end':>9}"
    )
    rows = []
    for c in COEFFS:
        nets, dirs, stds = [], [], []
        for s in seeds:
            sub = (df[df["seed"] == s] if s is not None else df).sort_values(
                "generation"
            )
            net, d = _directionality(sub[f"{c}_mean"])
            nets.append(net)
            dirs.append(d)
            stds.append(sub[f"{c}_std"].iloc[-1])
        nets = np.array(nets)
        agree = max(
            (nets > 0).mean(), (nets < 0).mean()
        )  # sign consistency across seeds
        rows.append((c, nets.mean(), np.mean(dirs), agree, np.mean(stds)))
    for c, net, d, agree, std in sorted(rows, key=lambda r: -abs(r[2])):
        flag = "  <- selection" if abs(d) > 0.4 and agree >= 0.8 else ""
        print(f"{c:16}{net:>+10.3f}{d:>+11.2f}{agree:>11.0%}{std:>9.3f}{flag}")
    print("  credible selection = |direction|>0.4 AND seeds_agree>=80%; else drift.")
    print("  std_end rising above the ~0.1 noise floor => roles differentiating.")


def ability_table(riders: pd.DataFrame):
    print("\n== ABILITY vs STRATEGY (does engine quality predict learned strategy?) ==")
    print(
        f"{'coefficient':16}{'corr(w_max10)':>15}{'weak25%':>10}{'strong25%':>11}{'strong-weak':>13}"
    )
    lo, hi = riders.w_max10.quantile(0.25), riders.w_max10.quantile(0.75)
    weak, strong = riders[riders.w_max10 <= lo], riders[riders.w_max10 >= hi]
    rows = []
    for c in COEFFS:
        if c not in riders:
            continue
        r = riders.w_max10.corr(riders[c])
        rows.append(
            (c, r, weak[c].mean(), strong[c].mean(), strong[c].mean() - weak[c].mean())
        )
    rows.sort(key=lambda x: -abs(0 if np.isnan(x[1]) else x[1]))
    for c, r, w, s, d in rows:
        print(f"{c:16}{r:>+15.2f}{w:>10.2f}{s:>11.2f}{d:>+13.2f}")
    print(
        "  |corr|>~0.2 => ability predicts that strategy dimension (a role by skill)."
    )
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: uv run python analyze.py <trajectory.csv>")
        sys.exit(1)
    traj = Path(sys.argv[1])
    df = pd.read_csv(traj)
    selection_table(df)

    # Seed-averaged coefficient trajectories.
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in (
        "coop.alpha_mean",
        "coop.delta_mean",
        "leave.alpha_mean",
        "follow.alpha_mean",
    ):
        m = (
            df.groupby("generation")[c].mean()
            if "seed" in df
            else df.set_index("generation")[c]
        )
        ax.plot(m.index, m.values, label=c)
    ax.set(
        xlabel="generation", ylabel="coefficient mean", title="coefficient trajectories"
    )
    ax.legend(fontsize=8)
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    p_traj = plots_dir / (traj.stem + "_trajectories.png")
    fig.savefig(p_traj, dpi=110, bbox_inches="tight")
    saved = [p_traj.name]

    riders_path = traj.with_name(traj.stem + "_riders.csv")
    if riders_path.exists():
        riders = pd.read_csv(riders_path)
        rows = ability_table(riders)
        c, corr = max(rows, key=lambda x: abs(0 if np.isnan(x[1]) else x[1]))[:2]
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(riders.w_max10, riders[c], s=10, alpha=0.4)
        ax2.set(
            xlabel="w_max10 (engine)",
            ylabel=c,
            title=f"ability vs {c} (corr {corr:+.2f})",
        )
        p_ab = plots_dir / (traj.stem + "_ability.png")
        fig2.savefig(p_ab, dpi=110, bbox_inches="tight")
        saved.append(p_ab.name)
    else:
        print(f"\n(no {riders_path.name} found — rerun `learn` for per-rider records)")

    print("\nplots saved:", ", ".join(saved))


if __name__ == "__main__":
    main()
