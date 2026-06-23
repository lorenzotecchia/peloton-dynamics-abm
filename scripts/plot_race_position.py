"""Race position plots for a dump produced by `main.py dump`.

Three views (--mode):
  gap   gap behind the leader, m  (0 = on the front) — detrended relative position.
  rank  position in the field over time — a bump chart; crossings = overtakes.
  abs   absolute x(t) space-time / trajectory diagram.

    uv run python scripts/plot_race_position.py --dir analysis_output/v-and-v --mode gap
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="analysis_output/v-and-v")
    p.add_argument("--mode", choices=["gap", "rank", "abs"], default="gap")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ts = pd.read_csv(os.path.join(args.dir, "agent_timeseries.csv"))
    meta = pd.read_csv(os.path.join(args.dir, "agent_meta.csv"))
    cfg = json.load(open(os.path.join(args.dir, "config.json")))
    out = args.out or os.path.join(args.dir, f"race_position_{args.mode}.png")

    # Detrend: at each time, the leader's x and each rider's rank in the field.
    ts = ts.copy()
    ts["leader_x"] = ts.groupby("step")["x"].transform("max")
    ts["gap"] = ts["leader_x"] - ts["x"]                       # m behind the front
    ts["field_rank"] = ts.groupby("step")["x"].rank(ascending=False, method="first")

    rank = meta.set_index("unique_id")["finish_rank"]
    n = int(rank.max())
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(11, 7))
    for uid, g in ts.groupby("unique_id"):
        r = rank.get(uid)
        if pd.isna(r):
            color, lw, z = "0.7", 0.6, 1
        else:
            color, lw, z = cmap((r - 1) / max(1, n - 1)), 1.0, 3 if r <= 5 else 2
        ycol = {"gap": "gap", "rank": "field_rank", "abs": "x"}[args.mode]
        ax.plot(g["time"], g[ycol], color=color, linewidth=lw, alpha=0.85, zorder=z)

    if args.mode == "gap":
        ax.set_ylabel("Gap behind leader (m)")
        ax.invert_yaxis()                  # front of race at the top
        ax.axhline(0, color="k", ls="--", lw=1)
        title = "Gap behind leader — detrended relative position"
    elif args.mode == "rank":
        ax.set_ylabel("Position in field (1 = front)")
        ax.invert_yaxis()
        title = "Bump chart — field position over time (crossings = overtakes)"
    else:
        ax.set_ylabel("Distance along course (m)")
        ax.axhline(cfg["road_length"], color="k", ls="--", lw=1)
        title = "Trajectory / space-time diagram — absolute x(t)"

    ax.set_xlabel("Time (s)")
    ax.set_title(f"{title}\ncolour = finish rank (dark/purple = winner)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, n))
    fig.colorbar(sm, ax=ax, label="finish rank")
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
