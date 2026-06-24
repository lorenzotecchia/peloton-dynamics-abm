"""Track the agent count of the leading pack and its gap to the front rider.

The leading pack is the geometrically contiguous group (state == 'grouped',
group_size > 1) with the highest mean x position at each step.

The front rider is the agent in 'solo' or 'isolated' state with the highest x
that is ahead of the leading pack's front. Gap = front_rider_x - pack_front_x.
Steps where no solo/isolated rider is ahead of the pack show no gap.

Usage:
    uv run python scripts/plot_leading_pack.py --dir analysis_output/v-and-v/
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="analysis_output/v-and-v")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ts  = pd.read_csv(os.path.join(args.dir, "agent_timeseries.csv"))
    out = args.out or os.path.join(args.dir, "leading_pack_size.png")

    ts = ts[ts["step"] > 0]

    rows = []
    for (step, t), grp in ts.groupby(["step", "time"]):
        packed = grp[grp["state"] == "grouped"]
        if packed.empty:
            continue
        lead_gid  = packed.groupby("group_id")["x"].mean().idxmax()
        members   = packed[packed["group_id"] == lead_gid]
        pack_front = members["x"].max()

        # Front rider: solo or isolated agent ahead of the pack front
        ahead = grp[
            grp["state"].isin(["solo", "isolated"]) &
            (grp["x"] > pack_front)
        ]
        if not ahead.empty:
            front_rider    = ahead.loc[ahead["x"].idxmax()]
            gap            = front_rider["x"] - pack_front
            front_rider_id = int(front_rider["unique_id"])
            front_state    = front_rider["state"]
        else:
            gap = float("nan")
            front_rider_id = None
            front_state    = None

        team_counts = members["team_id"].value_counts().to_dict()

        rows.append({
            "time":           t,
            "size":           len(members),
            "front_x":        pack_front,
            "mean_x":         members["x"].mean(),
            "gap":            gap,
            "front_rider_id": front_rider_id,
            "front_state":    front_state,
            "team_counts":    team_counts,
        })

    df = pd.DataFrame(rows)

    teams       = sorted(ts["team_id"].unique())
    team_cmap   = matplotlib.colormaps.get_cmap("tab10")
    team_colors = {tid: team_cmap(i) for i, tid in enumerate(teams)}

    # Build a (time × team) count matrix
    team_df = pd.DataFrame(index=df["time"], data=0,
                           columns=pd.Index(teams, name="team_id"), dtype=float)
    for _, row in df.iterrows():
        for tid, cnt in row["team_counts"].items():
            team_df.loc[row["time"], tid] = cnt

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    ax_pos, ax_gap, ax_stack, ax_frac = axes

    ax_pos.set_title("Leading pack — position, gap to front rider, and team membership")
    ax_pos.plot(df["time"], df["front_x"], color="darkorange", lw=1.2, label="front of pack")
    ax_pos.plot(df["time"], df["mean_x"],  color="darkorange", lw=0.8, ls="--", alpha=0.6, label="mean x")
    ax_pos.set_ylabel("Position (m)")
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    solo_mask     = df["front_state"] == "solo"
    isolated_mask = df["front_state"] == "isolated"
    ax_gap.plot(df["time"], df["gap"], color="0.8", lw=0.8, zorder=1)
    ax_gap.scatter(df.loc[solo_mask,     "time"], df.loc[solo_mask,     "gap"],
                   color="tab:red",  s=4, zorder=2, label="front rider: solo")
    ax_gap.scatter(df.loc[isolated_mask, "time"], df.loc[isolated_mask, "gap"],
                   color="tab:gray", s=4, zorder=2, label="front rider: isolated")
    ax_gap.set_ylabel("Gap to front rider (m)")
    ax_gap.legend(fontsize=8, markerscale=2)
    ax_gap.grid(True, alpha=0.3)

    times  = team_df.index.values
    bottom = np.zeros(len(times))
    for tid in teams:
        vals = team_df[tid].values
        ax_stack.fill_between(times, bottom, bottom + vals,
                              color=team_colors[tid], alpha=0.85, label=f"Team {tid}")
        bottom += vals
    ax_stack.set_ylabel("Agents in pack\n(by team)")
    ax_stack.legend(loc="upper right", fontsize=7, ncol=len(teams) // 2 + 1)
    ax_stack.grid(True, alpha=0.3)

    total  = team_df.sum(axis=1).replace(0, float("nan"))
    bottom = np.zeros(len(times))
    for tid in teams:
        frac = (team_df[tid] / total).fillna(0).values
        ax_frac.fill_between(times, bottom, bottom + frac,
                             color=team_colors[tid], alpha=0.85)
        bottom += frac
    ax_frac.set_ylim(0, 1)
    ax_frac.set_ylabel("Team fraction\nin pack")
    ax_frac.set_xlabel("Time (s)")
    ax_frac.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
