"""Collect generation-0 vs generation-N race dynamics for comparison.

For each replication, runs `generations` races in sequence (coefficients evolve
between races, exactly like run_batch_learning.py), but does *detailed*
step-by-step tracking only for generation 0 and the final generation. The
intermediate generations are run "plain" (no per-step bookkeeping) just to
advance the evolutionary state efficiently.

Per detailed generation, for every agent we record (across the whole race):
  - finish_step (None if it didn't finish -> DNF)
  - average stamina (mean of w_prime / w_full while active)
  - % of active time spent in a group (pack size >= 2, not solo)
  - % of active time spent in a breakaway/solo/chase (agent.solo == True)
  - % of active time "isolated" (alone, detected group size == 1, not solo)
Per race (race-level), we also record, for every step:
  - num_groups (pack fragmentation)
  - num_active riders
  - mean stamina across active riders

Definitions / assumptions (see also chat discussion):
  - "breakaway" and "chaser" share the same `agent.solo` flag in the model and
    cannot be told apart without changing model.py, so they are lumped into a
    single `pct_time_breakaway` category.
  - "isolated" = detected as alone (group size 1) while NOT flagged solo. This
    is an edge case that should rarely happen but is tracked separately so it
    doesn't get silently absorbed into "group".
  - Race duration is the number of steps until `model.running` becomes False
    (everyone finished) or `max_steps` is hit (whichever first).

Note on w_max10: this script does NOT fix/persist physiology across
generations — each generation resamples `w_max10` per slot exactly like the
unmodified model does (see CyclistAgent.__init__). This means "w_max10 at
gen0" for a given slot is not the same physical rider as "w_max10 at genlast"
for that slot; keep that in mind when interpreting any per-slot comparison
(the race-level and population-level aggregates in comparison_summary.csv are
unaffected by this, since they average across all agents within a generation).

Outputs (per replication, under <output_dir>/replication_<seed>/):
  gen_000_agents.csv         per-agent stats, generation 0
  gen_000_race_steps.csv     per-step race-level stats, generation 0
  gen_last_agents.csv        per-agent stats, last generation
  gen_last_race_steps.csv    per-step race-level stats, last generation

Plus a top-level combined file:
  <output_dir>/comparison_summary.csv   one row per replication, gen0 vs genlast

Use plot_generation_comparison.py separately to generate plots from this CSV.
"""

from __future__ import annotations

import argparse
import copy
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from peloton import group
from peloton.config import PelotonConfig
from peloton.evolution import evolve
from peloton.model import PelotonModel


# --------------------------------------------------------------------------- #
# Race simulation with per-step / per-agent tracking
# --------------------------------------------------------------------------- #


def _run_race_with_tracking(model: PelotonModel, max_steps: int) -> tuple[list[dict], list[dict]]:
    """Run one race, recording per-step race stats and per-agent time-in-state stats.

    Returns (race_step_records, agent_stats).
    """
    cfg = model.config

    agent_records: dict[int, dict] = {
        a.unique_id: {
            "w_max10": a.w_max10,
            "steps_active": 0,
            "steps_group": 0,
            "steps_breakaway": 0,
            "steps_isolated": 0,
            "stamina_sum": 0.0,
            "finish_step": None,
        }
        for a in model.riders
    }

    race_step_records: list[dict] = []
    step = 0
    while model.running and step < max_steps:
        active = list(model.agents)
        if not active:
            break

        groups = group.detect_groups(active, cfg.group_radius)
        group_size: dict[int, int] = {}
        for grp in groups:
            for a in grp:
                group_size[a.unique_id] = len(grp)

        stam_values = []
        for a in active:
            rec = agent_records[a.unique_id]
            rec["steps_active"] += 1
            stamina = (a.w_prime / a.w_full) if a.w_full else 0.0
            rec["stamina_sum"] += stamina
            stam_values.append(stamina)

            if a.solo:
                rec["steps_breakaway"] += 1
            elif group_size.get(a.unique_id, 1) >= 2:
                rec["steps_group"] += 1
            else:
                rec["steps_isolated"] += 1

        race_step_records.append(
            {
                "step": step,
                "num_active": len(active),
                "num_groups": len(groups),
                "mean_stamina": statistics.mean(stam_values) if stam_values else 0.0,
            }
        )

        model.step()
        step += 1

    # Assign finish steps from the model's finish order (set during model.step()).
    for unique_id, finish_step in model.finish_order:
        if unique_id in agent_records:
            agent_records[unique_id]["finish_step"] = finish_step

    agent_stats: list[dict] = []
    for a in model.riders:
        rec = agent_records[a.unique_id]
        active_steps = rec["steps_active"]
        if active_steps > 0:
            avg_stamina = rec["stamina_sum"] / active_steps
            pct_group = 100.0 * rec["steps_group"] / active_steps
            pct_breakaway = 100.0 * rec["steps_breakaway"] / active_steps
            pct_isolated = 100.0 * rec["steps_isolated"] / active_steps
        else:
            avg_stamina = pct_group = pct_breakaway = pct_isolated = float("nan")

        agent_stats.append(
            {
                "agent_id": a.unique_id,
                "w_max10": rec["w_max10"],
                "finish_step": rec["finish_step"],
                "finished": rec["finish_step"] is not None,
                "avg_stamina": avg_stamina,
                "pct_time_group": pct_group,
                "pct_time_breakaway": pct_breakaway,
                "pct_time_isolated": pct_isolated,
            }
        )

    return race_step_records, agent_stats


def _run_race_plain(model: PelotonModel, max_steps: int) -> None:
    """Run one race with no extra bookkeeping (used for non-tracked generations)."""
    for _ in range(max_steps):
        if not model.running:
            break
        model.step()


# --------------------------------------------------------------------------- #
# Per-replication driver
# --------------------------------------------------------------------------- #


def run_replication_comparison(
    seed: int, generations: int, max_steps: int, output_dir: Path
) -> dict | None:
    cfg = PelotonConfig(seed=seed)

    population: list[dict] | None = None
    gen0_race_steps: list[dict] | None = None
    gen0_agents: list[dict] | None = None
    last_race_steps: list[dict] | None = None
    last_agents: list[dict] | None = None

    for gen in range(generations):
        model = PelotonModel(config=cfg, population=population)
        detailed = gen == 0 or gen == generations - 1

        if detailed:
            race_steps, agent_stats = _run_race_with_tracking(model, max_steps)
        else:
            _run_race_plain(model, max_steps)

        evolve(model.riders, model)
        population = [copy.deepcopy(r.coeffs) for r in model.riders]

        if gen == 0 and detailed:
            gen0_race_steps, gen0_agents = race_steps, agent_stats
        if gen == generations - 1 and detailed:
            last_race_steps, last_agents = race_steps, agent_stats

    if gen0_race_steps is None or last_race_steps is None:
        return None

    rep_dir = output_dir / f"replication_{seed:03d}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(gen0_agents).to_csv(rep_dir / "gen_000_agents.csv", index=False)
    pd.DataFrame(gen0_race_steps).to_csv(rep_dir / "gen_000_race_steps.csv", index=False)
    pd.DataFrame(last_agents).to_csv(rep_dir / "gen_last_agents.csv", index=False)
    pd.DataFrame(last_race_steps).to_csv(rep_dir / "gen_last_race_steps.csv", index=False)

    return _summarize_replication(
        seed, gen0_race_steps, gen0_agents, last_race_steps, last_agents, cfg.n_agents
    )


def _summarize_replication(
    seed: int,
    gen0_race_steps: list[dict],
    gen0_agents: list[dict],
    last_race_steps: list[dict],
    last_agents: list[dict],
    n_agents: int,
) -> dict:
    def _safe_mean(values):
        vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(statistics.mean(vals)) if vals else float("nan")

    def _agent_means(agents: list[dict]) -> dict:
        return {
            "avg_stamina": _safe_mean([a["avg_stamina"] for a in agents]),
            "pct_time_group": _safe_mean([a["pct_time_group"] for a in agents]),
            "pct_time_breakaway": _safe_mean([a["pct_time_breakaway"] for a in agents]),
            "pct_time_isolated": _safe_mean([a["pct_time_isolated"] for a in agents]),
            "pct_finished": 100.0 * sum(a["finished"] for a in agents) / len(agents),
            "mean_finish_step": _safe_mean(
                [a["finish_step"] for a in agents if a["finished"]]
            ),
        }

    def _race_means(race_steps: list[dict]) -> dict:
        num_groups = [r["num_groups"] for r in race_steps]
        return {
            "race_duration_steps": len(race_steps),
            "avg_num_groups": _safe_mean(num_groups),
            "avg_num_groups_pct": _safe_mean(num_groups) / n_agents * 100.0,
        }

    gen0_agent_means = _agent_means(gen0_agents)
    last_agent_means = _agent_means(last_agents)
    gen0_race_means = _race_means(gen0_race_steps)
    last_race_means = _race_means(last_race_steps)

    row = {"seed": seed}
    for k, v in gen0_race_means.items():
        row[f"gen0_{k}"] = v
    for k, v in last_race_means.items():
        row[f"genlast_{k}"] = v
    for k, v in gen0_agent_means.items():
        row[f"gen0_{k}"] = v
    for k, v in last_agent_means.items():
        row[f"genlast_{k}"] = v
    return row


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #


def run_all_replications(
    num_replications: int, generations: int, max_steps: int, output_dir: Path
) -> list[dict]:
    rows: list[dict] = []
    for seed in range(num_replications):
        print(f"Running {seed + 1}/{num_replications}...", end="\r")
        row = run_replication_comparison(seed, generations, max_steps, output_dir)
        if row:
            rows.append(row)
    print(f"Completed {len(rows)}/{num_replications} replications.")
    return rows


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare race dynamics (duration, stamina, group/breakaway/isolated time, "
        "fragmentation) between generation 0 and the final generation, across replications."
    )
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--output-dir", default="data/generation_comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_all_replications(args.replications, args.generations, args.max_steps, output_dir)
    df = pd.DataFrame(rows)
    summary_path = output_dir / "comparison_summary.csv"
    df.to_csv(summary_path, index=False)

    print(f"\nSaved summary: {summary_path}")
    print(
        "Run plot_generation_comparison.py --input-dir "
        f"{output_dir} to generate the comparison plots."
    )


if __name__ == "__main__":
    main()
