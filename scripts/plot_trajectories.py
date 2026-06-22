"""Run a simulation and plot rider trajectories.

Time on x-axis (seconds), distance traveled on y-axis (meters). Lines are colored
by team_id. Status markers: grouped (green), break_group (orange), solo (red),
finished (gray).

Run: python scripts\plot_trajectories.py
"""
from collections import defaultdict
import math
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
except Exception as e:
    raise RuntimeError("Matplotlib and numpy are required to run this script. Install them with 'pip install matplotlib numpy'.")

from peloton.model import PelotonModel
from peloton import group

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

def run_and_record(cfg_overrides=None, max_steps=20000):
    model = PelotonModel(config=None, **(cfg_overrides or {}))
    cfg = model.config

    # Initialize histories: per-agent lists of times, x positions, statuses
    histories = {r.unique_id: {"times": [], "x": [], "status": [], "team": r.team_id}
                 for r in model.riders}

    step = 0
    while getattr(model, "running", True) and step < max_steps:
        # Determine groups of currently active agents (on-road)
        active = list(model.agents)
        packs = group.detect_groups(active, cfg.group_radius)
        pack_of = {}
        for i, p in enumerate(packs):
            for a in p:
                pack_of[a] = (i, len(p))

        # Determine breakaway groups among solo riders
        solo_actives = [a for a in active if getattr(a, "solo", False)]
        solo_packs = group.detect_groups(solo_actives, cfg.group_radius)
        solo_pack_of = {}
        for i, p in enumerate(solo_packs):
            for a in p:
                solo_pack_of[a] = (i, len(p))

        t = step * cfg.dt
        # Record state for all riders (including finished) using model.riders stable list
        for r in model.riders:
            # Get x position if available (some removed agents still keep pos)
            x = None
            try:
                pos = getattr(r, "pos", None)
                if pos is not None:
                    x = float(pos[0])
            except Exception:
                x = None

            # Determine status label
            if r in active:
                if getattr(r, "solo", False):
                    # Solo rider: if part of a solo pack (>1) it's a break_group
                    sp = solo_pack_of.get(r)
                    if sp is not None and sp[1] > 1:
                        status = "break_group"
                    else:
                        status = "solo"
                else:
                    p = pack_of.get(r)
                    if p is not None and p[1] > 1:
                        status = "grouped"
                    else:
                        status = "isolated"
            else:
                status = "finished"

            histories[r.unique_id]["times"].append(t)
            histories[r.unique_id]["x"].append(np.nan if x is None else x)
            histories[r.unique_id]["status"].append(status)

        model.step()
        step += 1

    return histories, cfg, step


def plot_histories(histories, cfg, out_path):
    # Build color map for teams
    team_ids = sorted({v["team"] for v in histories.values()})
    n_teams = max(1, max(team_ids) + 1) if team_ids else 1
    cmap = plt.get_cmap('tab10')
    team_colors = {tid: cmap(tid % 10) for tid in team_ids}

    fig, ax = plt.subplots(figsize=(12, 6))

    status_marker_color = {
        "grouped": (0.2, 0.8, 0.2, 0.6),    # green
        "break_group": (1.0, 0.5, 0.0, 0.9), # orange
        "solo": (1.0, 0.0, 0.0, 0.9),       # red
        "finished": (0.5, 0.5, 0.5, 0.8),   # gray
        "isolated": (0.2, 0.6, 0.9, 0.6),   # blue-ish for isolated
    }

    for uid, data in histories.items():
        times = np.array(data["times"])
        xs = np.array(data["x"], dtype=float)
        status = data["status"]
        team = data["team"]
        color = team_colors.get(team, (0.0, 0.0, 0.0))

        # Plot continuous line (NaNs break the line where data missing)
        ax.plot(times, xs, color=color, linewidth=1.2, alpha=0.8)

        # Overlay status markers as small dots (sparse sampling to avoid overplotting)
        # We'll plot every step but use small sizes.
        for s, col in status_marker_color.items():
            mask = np.array([st == s for st in status])
            if not mask.any():
                continue
            ax.scatter(times[mask], xs[mask], c=[col], s=6, marker='o')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance along road (m)')
    ax.set_title('Cyclist trajectories — colored by team; markers show status')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Create a legend for teams and status
    team_patches = [plt.Line2D([0], [0], color=team_colors[tid], lw=3) for tid in team_ids]
    team_labels = [f'Team {tid}' for tid in team_ids]

    status_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=6)
                      for col in status_marker_color.values()]
    status_labels = list(status_marker_color.keys())

    leg1 = ax.legend(team_patches, team_labels, title='Teams', loc='upper left')
    leg2 = ax.legend(status_patches, status_labels, title='Status', loc='upper right')
    ax.add_artist(leg1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Plot saved to {out_path}')


if __name__ == '__main__':
    histories, cfg, steps = run_and_record()
    out = os.path.join(OUT_DIR, f'trajectories_steps_{steps}.png')
    plot_histories(histories, cfg, out)
