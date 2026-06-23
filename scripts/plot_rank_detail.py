"""Stamina fraction, pulling share, field rank, and state over time for a single rider.

Plots four stacked panels (shared x-axis):
  Top        stamina fraction (W'/W_full) over time
  Upper-mid  pulling share / wind exposure (0 = sheltered, 1 = full wind)
  Lower-mid  field rank over time (1 = physically furthest ahead)
  Bottom     state strip (grouped / break_group / solo / isolated)

Background shading encodes the rider's state at each moment.

Usage:
    uv run python scripts/plot_rank_detail.py --dir analysis_output/20260623-134214/
    uv run python scripts/plot_rank_detail.py --dir analysis_output/20260623-134214/ --rank 3
    uv run python scripts/plot_rank_detail.py --dir analysis_output/20260623-134214/ --uid 12
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


STATE_COLORS = {
    "grouped":     "#d0e8ff",
    "break_group": "#ffe0b0",
    "solo":        "#ffd0d0",
    "isolated":    "#f0f0f0",
}
STATE_ORDER = ["grouped", "break_group", "solo", "isolated"]
STATE_DOT_COLORS = {
    "grouped": "tab:blue", "break_group": "tab:orange",
    "solo": "tab:red",     "isolated":    "tab:gray",
}


def shade_states(ax, agent_df):
    prev_t = agent_df["time"].iloc[0]
    prev_s = agent_df["state"].iloc[0]
    for _, row in agent_df.iterrows():
        if row["state"] != prev_s:
            ax.axvspan(prev_t, row["time"],
                       color=STATE_COLORS.get(prev_s, "white"), alpha=0.4, lw=0)
            prev_t, prev_s = row["time"], row["state"]
    ax.axvspan(prev_t, agent_df["time"].iloc[-1],
               color=STATE_COLORS.get(prev_s, "white"), alpha=0.4, lw=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="analysis_output/worm")
    p.add_argument("--rank", type=int, default=1,
                   help="Finish rank of the rider to plot (default: 1 = winner)")
    p.add_argument("--uid", type=int, default=None,
                   help="Plot a specific unique_id instead of a finish rank")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ts   = pd.read_csv(os.path.join(args.dir, "agent_timeseries.csv"))
    meta = pd.read_csv(os.path.join(args.dir, "agent_meta.csv"))

    if args.uid is not None:
        uid = args.uid
        finish_rank = meta.loc[meta["unique_id"] == uid, "finish_rank"]
        label = f"uid={uid}" + (f"  rank={int(finish_rank.iloc[0])}" if not finish_rank.empty else "")
    else:
        row = meta[meta["finish_rank"] == args.rank]
        if row.empty:
            raise SystemExit(f"No rider with finish_rank={args.rank} in {args.dir}")
        uid = int(row["unique_id"].iloc[0])
        label = f"rank={args.rank}  uid={uid}"

    ts["field_rank"] = ts.groupby("step")["x"].rank(ascending=False, method="first")
    a = ts[ts["unique_id"] == uid].copy().reset_index(drop=True)
    if a.empty:
        raise SystemExit(f"No timeseries rows for uid={uid}")
    n_agents = ts["unique_id"].nunique()

    out = args.out or os.path.join(args.dir, f"rank{args.rank}_stamina_vs_pull.png")

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2, 1]})
    ax_stam, ax_pull, ax_rank, ax_state = axes

    for ax in (ax_stam, ax_pull, ax_rank, ax_state):
        shade_states(ax, a)

    # stamina
    ax_stam.plot(a["time"], a["stamina_frac"], color="steelblue", lw=1.5)
    ax_stam.axhline(0, color="red", lw=0.8, ls="--", alpha=0.6, label="W′ = 0 (exhausted)")
    ax_stam.set_ylabel("Stamina  W′/W_full")
    ax_stam.set_ylim(-0.05, 1.1)
    ax_stam.legend(fontsize=8, loc="upper right")
    ax_stam.grid(True, alpha=0.3)
    ax_stam.set_title(f"Agent {label} — stamina fraction & pulling share over time")

    # pulling share
    ax_pull.plot(a["time"], a["exposure"], color="darkorange", lw=1.5)
    ax_pull.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5, label="full wind (solo)")
    ax_pull.set_ylabel("Pulling share / exposure\n(0 = sheltered, 1 = full wind)")
    ax_pull.set_ylim(-0.05, 1.1)
    ax_pull.legend(fontsize=8, loc="upper right")
    ax_pull.grid(True, alpha=0.3)

    # field rank
    ax_rank.plot(a["time"], a["field_rank"], color="mediumpurple", lw=1.5)
    ax_rank.set_ylabel("Field rank\n(1 = front)")
    ax_rank.set_ylim(n_agents + 0.5, 0.5)   # inverted: 1 at top
    ax_rank.axhline(1, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax_rank.grid(True, alpha=0.3)

    # state strip
    state_y = {s: i for i, s in enumerate(STATE_ORDER)}
    ax_state.scatter(
        a["time"], a["state"].map(state_y),
        c=[STATE_DOT_COLORS.get(s, "k") for s in a["state"]],
        s=6, zorder=3,
    )
    ax_state.set_yticks(range(len(STATE_ORDER)))
    ax_state.set_yticklabels(STATE_ORDER, fontsize=7)
    ax_state.set_xlabel("Time (s)")
    ax_state.set_ylabel("State")
    ax_state.grid(True, alpha=0.3)

    legend_patches = [
        mpatches.Patch(color=STATE_COLORS[s], alpha=0.7, label=s)
        for s in STATE_ORDER
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, title="Background = rider state", title_fontsize=8,
               bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
