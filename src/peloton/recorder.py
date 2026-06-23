"""Run one race and dump *everything* to disk for post-simulation analysis.

This is a read-only observer over a normal ``PelotonModel`` run: it snapshots the
full per-agent state at every step (before the model advances, so each snapshot is
the situation the riders are about to act under) and writes a tidy bundle of files.
Nothing in the simulation core is modified — every column is either a live agent
attribute or derived from one (speed from position deltas, ``cf_eff`` back-solved
from ``wind_power``, group membership from ``group.detect_groups``).

Output bundle (one directory):
    agent_timeseries.csv   long format, one row per (step, agent) — the bulk of it
    model_timeseries.csv   the model-level reporters (Mesa DataCollector)
    agent_meta.csv         one row per rider: static physiology, flat coeffs, finish
    finish_order.csv       rank, rider, finish step/time
    config.json            the full PelotonConfig used
    manifest.json          run metadata + a menu of analyses the data supports

Run:
    uv run python main.py dump [--seed N] [--max-steps N] [--out-dir DIR] [--parquet]
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime

from peloton import group as group_mod
from peloton.config import PelotonConfig
from peloton.model import PelotonModel


def _rider_state(agent, active, solo_packs_cache, cfg) -> str:
    """Label a rider's situation (mirrors scripts/plot_agent_tracking.py)."""
    if getattr(agent, "solo", False):
        for sp in solo_packs_cache:
            if agent in sp and len(sp) > 1:
                return "break_group"
        return "solo"
    # not solo: grouped if sharing a geometric pack with anyone, else isolated
    return "grouped" if agent._group_size > 1 else "isolated"


def record_run(config: PelotonConfig, max_steps: int) -> dict:
    """Run a single race, capturing the full per-step per-agent state.

    Returns a dict of plain Python rows / tables ready to be written out, so this
    function has no I/O or pandas dependency itself (easy to unit-test or reuse).
    """
    model = PelotonModel(config=config)
    cfg = model.config
    k_aero = cfg.k_aero
    dt = cfg.dt

    # uid -> last x, to derive ground speed (dx/dt) one step later.
    last_x: dict[int, float] = {}
    rows: list[dict] = []

    step = 0
    while getattr(model, "running", True) and step < max_steps:
        active = list(model.agents)
        t = step * dt

        # Geometric packs over all active riders (drafting view) and over solo
        # riders only (breakaway view) — both computed once per step.
        packs = group_mod.detect_groups(active, cfg.group_radius)
        group_of: dict[int, tuple[int, int]] = {}
        for gid, members in enumerate(packs):
            for m in members:
                group_of[m.unique_id] = (gid, len(members))
        solo_packs = group_mod.detect_groups(
            [a for a in active if getattr(a, "solo", False)], cfg.group_radius
        )

        for a in active:
            gid, gsize = group_of.get(a.unique_id, (-1, 1))
            a._group_size = gsize  # used by _rider_state below

            x = a.pos[0]
            prev = last_x.get(a.unique_id)
            speed = (x - prev) / dt if prev is not None else float("nan")
            last_x[a.unique_id] = x

            stamina_frac = a.w_prime / a.w_full if a.w_full else 0.0
            # wind_power = k_aero * cf_eff * v^3  ->  back out the drag factor the
            # rider actually experienced (only meaningful while moving).
            cf_eff = (
                a.wind_power / (k_aero * speed**3)
                if speed and speed > 0.1
                else float("nan")
            )

            rows.append({
                "step": step,
                "time": t,
                "unique_id": a.unique_id,
                "team_id": a.team_id,
                "x": x,
                "y": a.pos[1],
                "dist_to_finish": cfg.road_length - x,
                "speed": speed,
                "w_prime": a.w_prime,
                "w_full": a.w_full,
                "stamina_frac": stamina_frac,
                "wind_power": a.wind_power,
                "cf_eff": cf_eff,
                "exposure": a.exposure,
                "solo": int(a.solo),
                "break_cooldown": a.break_cooldown,
                "group_id": gid,
                "group_size": gsize,
                "state": _rider_state(a, active, solo_packs, cfg),
            })

        model.step()
        step += 1

    # --- model-level reporter time series (Mesa already collected this) ---
    model_df = model.datacollector.get_model_vars_dataframe()
    model_rows = [
        {"step": i, "time": i * dt, **{k: row[k] for k in model_df.columns}}
        for i, (_, row) in enumerate(model_df.iterrows())
    ]

    # --- static per-rider metadata, flattened coeffs, finish outcome ---
    rank_of = {uid: r for r, (uid, _s) in enumerate(model.finish_order, start=1)}
    step_of = {uid: s for uid, s in model.finish_order}
    meta_rows = []
    for r in model.riders:
        flat_coeffs = {
            f"coeff.{grp}.{p}": v
            for grp, params in r.coeffs.items()
            for p, v in params.items()
        }
        meta_rows.append({
            "unique_id": r.unique_id,
            "team_id": r.team_id,
            "w_max10": r.w_max10,
            "cp": r.cp,
            "s_m": r.s_m,
            "s_cp": r.s_cp,
            "w_full": r.w_full,
            "utility": r.utility,
            "finish_rank": rank_of.get(r.unique_id),
            "finish_step": step_of.get(r.unique_id),
            "finished": r.unique_id in rank_of,
            **flat_coeffs,
        })

    finish_rows = [
        {"rank": r, "unique_id": uid, "finish_step": s, "finish_time": s * dt}
        for r, (uid, s) in enumerate(model.finish_order, start=1)
    ]

    return {
        "agent_timeseries": rows,
        "model_timeseries": model_rows,
        "agent_meta": meta_rows,
        "finish_order": finish_rows,
        "config": asdict(cfg),
        "n_steps": step,
        "n_finished": model.n_finished,
        "n_agents": cfg.n_agents,
    }


ANALYSIS_MENU = [
    "Finish-order vs. physiology: regress finish_rank on w_max10/cp/w_full (does the strongest engine win, or does drafting flatten it?).",
    "Energy economy: per-rider cumulative wind energy (cumsum of wind_power*dt) — who spent the least to finish where? Plot vs. finish_rank.",
    "Drafting benefit: distribution of cf_eff / exposure per rider over the race; time spent leading (cf_eff~1) vs. sheltered.",
    "Pack dynamics: NumGroups & group_size over time — when does the field shatter? Track the largest-group size as a fragmentation index.",
    "Breakaway analysis: count/duration of 'solo' and 'break_group' states per rider; success rate (did breakers finish higher?).",
    "Stamina trajectories: stamina_frac vs. time per rider; identify who blows up (hits 0) and when, relative to finish.",
    "Speed profiles: speed vs. dist_to_finish; detect sprint-finish surges and the pace of breakaways vs. the bunch.",
    "Strategy <-> outcome: join agent_meta coeffs (coop/leave/follow alpha/beta/gamma/delta) to finish_rank to see which strategies pay off.",
    "Spatial/temporal: x(t) worm plot of all riders (catch/escape events visible as converging/diverging lines).",
    "Across-race learning: run `learn` to get coefficient trajectories; correlate with the per-race dumps here.",
]


def dump_run(config: PelotonConfig, max_steps: int, out_dir: str, parquet: bool) -> str:
    """Run a race and write the full analysis bundle to ``out_dir``. Returns the dir."""
    import pandas as pd

    data = record_run(config, max_steps)
    os.makedirs(out_dir, exist_ok=True)

    tables = {
        "agent_timeseries": data["agent_timeseries"],
        "model_timeseries": data["model_timeseries"],
        "agent_meta": data["agent_meta"],
        "finish_order": data["finish_order"],
    }
    written = []
    for name, rows in tables.items():
        df = pd.DataFrame(rows)
        if parquet:
            path = os.path.join(out_dir, f"{name}.parquet")
            df.to_parquet(path, index=False)
        else:
            path = os.path.join(out_dir, f"{name}.csv")
            df.to_csv(path, index=False)
        written.append(path)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(data["config"], f, indent=2)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_agents": data["n_agents"],
        "n_steps": data["n_steps"],
        "n_finished": data["n_finished"],
        "n_timeseries_rows": len(data["agent_timeseries"]),
        "files": {
            "agent_timeseries": "one row per (step, agent): position, speed, stamina, "
                                "wind power, cf_eff, exposure, solo/break state, group id/size",
            "model_timeseries": "MeanStamina, NumGroups, Breakaways, MeanExposure per step",
            "agent_meta": "static physiology + flattened strategy coeffs + finish outcome",
            "finish_order": "rank, rider, finish step/time",
            "config.json": "the full PelotonConfig used for this run",
        },
        "analysis_menu": ANALYSIS_MENU,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dumped {data['n_steps']} steps x {data['n_agents']} riders "
          f"({len(data['agent_timeseries'])} rows) to {out_dir}/")
    print(f"  files: {', '.join(os.path.basename(p) for p in written)}, config.json, manifest.json")
    print(f"  race: {data['n_finished']}/{data['n_agents']} finished in {data['n_steps']} steps")
    print("\nWhat you can analyze (see manifest.json -> analysis_menu):")
    for i, idea in enumerate(ANALYSIS_MENU, start=1):
        print(f"  {i:>2}. {idea}")
    return out_dir
