"""Track a single cyclist's stamina, wind-friction energy, and riding state over a race.

Stamina is W' (anaerobic work capacity) as a fraction of the rider's full W'.
Wind friction power is the aerodynamic component of power output: k_aero * cf_eff * v^3 (W).
Cumulative wind friction energy is the integral of that power over time (kJ).

State strip (third panel) shows the rider's moment-to-moment situation:
  grouped     — sheltered in a pack with at least one other rider
  isolated    — alone but not flagged solo (dropped from pack, not yet chasing)
  solo        — flagged solo: off the front or chasing alone
  break_group — flagged solo but riding with other solo riders (collective breakaway)

Run:
    uv run python scripts/plot_agent_tracking.py [--rider N] [--out plots/agent_tracking.png]

--rider N   index into model.riders (0-based); defaults to 0 (first spawned rider)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")   # headless: no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from peloton import group as group_mod
from peloton.model import PelotonModel

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# State colour palette (also used as background tints on the metric panels)
STATE_COLORS = {
    "grouped":     "#4caf50",   # green
    "isolated":    "#2196f3",   # blue
    "solo":        "#f44336",   # red
    "break_group": "#ff9800",   # orange
}
STATE_ALPHA_BG = 0.08   # transparency for background tints on metric panels
STATE_ALPHA_STRIP = 0.85


def _rider_state(tracked, model, cfg) -> str:
    """Determine current state of *tracked* from the live model geometry."""
    active = list(model.agents)
    if tracked not in active:
        return "finished"

    if getattr(tracked, "solo", False):
        solo_riders = [a for a in active if getattr(a, "solo", False)]
        solo_packs = group_mod.detect_groups(solo_riders, cfg.group_radius)
        for sp in solo_packs:
            if tracked in sp and len(sp) > 1:
                return "break_group"
        return "solo"
    else:
        packs = group_mod.detect_groups(active, cfg.group_radius)
        for p in packs:
            if tracked in p and len(p) > 1:
                return "grouped"
        return "isolated"


def run_and_track(rider_index: int = 0, max_steps: int = 20000):
    model = PelotonModel()
    cfg = model.config

    if rider_index >= len(model.riders):
        raise ValueError(
            f"rider_index={rider_index} out of range; model has {len(model.riders)} riders."
        )

    tracked = model.riders[rider_index]
    uid = tracked.unique_id

    times = []
    stamina_frac = []   # w_prime / w_full
    wind_power_w = []   # aerodynamic power in W
    states = []         # string state label per step

    step = 0
    while getattr(model, "running", True) and step < max_steps:
        t = step * cfg.dt
        active_ids = {a.unique_id for a in model.agents}

        if uid in active_ids:
            times.append(t)
            stamina_frac.append(tracked.w_prime / tracked.w_full if tracked.w_full else 0.0)
            wind_power_w.append(tracked.wind_power)
            states.append(_rider_state(tracked, model, cfg))

        model.step()
        step += 1

    times = np.array(times)
    stamina_frac = np.array(stamina_frac)
    wind_power_w = np.array(wind_power_w)
    cumul_wind_kj = np.cumsum(wind_power_w * cfg.dt) / 1000.0

    finish_step = next((s for uid2, s in model.finish_order if uid2 == uid), None)
    finish_time = finish_step * cfg.dt if finish_step is not None else None

    return times, stamina_frac, wind_power_w, cumul_wind_kj, states, finish_time, cfg, tracked


def _add_state_background(ax, times, states):
    """Draw semi-transparent vertical spans on *ax* coloured by rider state."""
    if len(times) == 0:
        return
    # Walk through contiguous blocks of the same state
    prev_state = states[0]
    block_start = times[0]
    for i in range(1, len(states)):
        if states[i] != prev_state:
            ax.axvspan(block_start, times[i], color=STATE_COLORS.get(prev_state, "#aaaaaa"),
                       alpha=STATE_ALPHA_BG, linewidth=0)
            prev_state = states[i]
            block_start = times[i]
    # final block
    ax.axvspan(block_start, times[-1], color=STATE_COLORS.get(prev_state, "#aaaaaa"),
               alpha=STATE_ALPHA_BG, linewidth=0)


def _draw_state_strip(ax, times, states):
    """Fill a thin axis with solid colour blocks per state (the state-strip panel)."""
    if len(times) == 0:
        return
    prev_state = states[0]
    block_start = times[0]
    for i in range(1, len(states)):
        if states[i] != prev_state:
            ax.axvspan(block_start, times[i], color=STATE_COLORS.get(prev_state, "#aaaaaa"),
                       alpha=STATE_ALPHA_STRIP, linewidth=0)
            prev_state = states[i]
            block_start = times[i]
    ax.axvspan(block_start, times[-1], color=STATE_COLORS.get(prev_state, "#aaaaaa"),
               alpha=STATE_ALPHA_STRIP, linewidth=0)
    ax.set_yticks([])
    ax.set_ylabel("State", fontsize=8, rotation=0, labelpad=30, va="center")


def plot(times, stamina_frac, wind_power_w, cumul_wind_kj, states,
         finish_time, cfg, tracked, out_path):

    fig = plt.figure(figsize=(11, 9))
    # Three panels: stamina (large), wind power (large), state strip (thin)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 0.6], hspace=0.08)
    ax1   = fig.add_subplot(gs[0])
    ax2   = fig.add_subplot(gs[1], sharex=ax1)
    ax_st = fig.add_subplot(gs[2], sharex=ax1)

    # --- state background tints on metric panels ---
    _add_state_background(ax1, times, states)
    _add_state_background(ax2, times, states)

    # --- top panel: stamina ---
    ax1.plot(times, stamina_frac * 100.0, color="steelblue", linewidth=1.5, zorder=3)
    ax1.set_ylabel("Stamina remaining (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.set_title(
        f"Rider {tracked.unique_id} — W_max10={tracked.w_max10:.0f} W, "
        f"CP={tracked.cp:.0f} W, W_full={tracked.w_full:.0f} J",
        fontsize=10,
    )
    plt.setp(ax1.get_xticklabels(), visible=False)

    # --- middle panel: instantaneous wind power + cumulative energy ---
    color_inst  = "firebrick"
    color_cumul = "darkorange"

    ax2.plot(times, wind_power_w, color=color_inst, linewidth=1.2,
             label="Wind friction power (W)", zorder=3)
    ax2.set_ylabel("Wind friction power (W)", color=color_inst)
    ax2.tick_params(axis="y", labelcolor=color_inst)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3, zorder=0)

    ax2b = ax2.twinx()
    ax2b.plot(times, cumul_wind_kj, color=color_cumul, linewidth=1.5,
              linestyle="--", label="Cumulative wind energy (kJ)", zorder=3)
    ax2b.set_ylabel("Cumulative wind energy (kJ)", color=color_cumul)
    ax2b.tick_params(axis="y", labelcolor=color_cumul)
    plt.setp(ax2.get_xticklabels(), visible=False)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # --- bottom: state strip ---
    _draw_state_strip(ax_st, times, states)
    ax_st.set_xlabel("Time (s)")

    # State legend (patches)
    patches = [
        mpatches.Patch(color=col, alpha=STATE_ALPHA_STRIP, label=name)
        for name, col in STATE_COLORS.items()
    ]
    ax_st.legend(handles=patches, loc="lower right", fontsize=7,
                 ncol=len(patches), framealpha=0.8)

    # --- finish line on all three panels ---
    if finish_time is not None:
        for ax in (ax1, ax2, ax_st):
            ax.axvline(finish_time, color="gray", linestyle=":", linewidth=1.2, zorder=4)
        ax1.text(finish_time, ax1.get_ylim()[1] * 0.97, " finish",
                 va="top", ha="left", fontsize=8, color="gray")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot stamina, wind friction, and state for a single rider")
    p.add_argument("--rider", type=int, default=0,
                   help="Index into model.riders (0-based, default 0)")
    p.add_argument("--out", default=os.path.join(OUT_DIR, "agent_tracking.png"))
    args = p.parse_args()

    times, stamina_frac, wind_power_w, cumul_wind_kj, states, finish_time, cfg, tracked = \
        run_and_track(args.rider)

    plot(times, stamina_frac, wind_power_w, cumul_wind_kj, states,
         finish_time, cfg, tracked, args.out)


if __name__ == "__main__":
    main()
