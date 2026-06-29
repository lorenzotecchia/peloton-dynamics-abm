# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent-Based Model of competitive cycling (peloton dynamics) using [Mesa](https://mesa.readthedocs.io/) 3.x. Cyclists are agents that learn cooperation/breakaway strategies *across races* (one race = one model run; learning happens between races). Models drafting (energy savings behind other riders), pack fragmentation, breakaway formation, and follow/chase decisions.

The simulation runs on [Snellius](https://www.surf.nl/en/services/snellius-the-national-supercomputer) (SURF HPC) via Slurm for parameter sweeps and sensitivity analysis. The Solara viz auto-deploys to Hugging Face Spaces.

## Commands

```bash
uv sync                                   # install deps from uv.lock (Python 3.12+)

# main.py is the CLI (subcommands), not a placeholder:
uv run python main.py run                 # one race, headless, prints finish order
uv run python main.py dump                # one race, dump per-step per-agent data (CSV/--parquet)
uv run python main.py learn --generations 200   # races in sequence, learning between -> data/learning.csv + data/population.json
uv run python main.py solara              # interactive Solara viz (blocks; Ctrl-C to stop)
uv run python main.py solara --population data/population.json   # replay a learned field
uv run python main.py test                # run the suite (pytest)

uv run solara run run_app.py              # Solara viz directly (what HF Spaces runs)

# Sensitivity analysis / parallel replicates (designed for one fat CPU node):
uv run python -m peloton.sweep --runs 256 --out data/results.csv
uv run python -m peloton.gsa --method both --samples 512 --replicates 5 --generations 30

# Tests
uv run pytest                             # all
uv run pytest tests/test_evolution.py     # one file
uv run pytest tests/test_model.py::test_name   # one test
```

## Snellius / HPC Workflow

Run **once** on the login node (never a compute node): `bash scripts/snellius-py-runtime-setup.sh`.

Do **not** run training/SA on the login node — submit Slurm jobs from `scripts/`:
- `job-gpu-h100-salloc.sh` / `-08hr.sh` (interactive), `job-gpu-h100-sbatch.sh` (batch)
- `job-cpu-rome-*` equivalents (the SA work is CPU-bound: `job-cpu-rome-gsa-{morris,sobol}*.sh`)
- `_gsa-*.sh` are shared job bodies / array helpers sourced by the above; `plot_*.py` and `plot-SA.sh` render results from `data-SA/`.

Job logs land in `logs/` (`batch_<jobid>.log` / `.err`).

## Architecture

`src/peloton/` is the package (`pyproject.toml` builds `src/peloton`). Dynamics are orchestrated **pack-wise by the model**, not by individual agents — a rider can't set a pack's speed alone.

- **`config.py`** (`PelotonConfig`): frozen dataclass of every tunable knob (physiology, drafting, breakaway, evolution). This is the single source of truth — `model._resolve_config` reads field names/types straight off it, so adding a field here automatically makes it a SolaraViz slider override and an SA target with no other edit.
- **`agent.py`** (`CyclistAgent`): **state only** (physiology + `coeffs` + `solo`/`break_cooldown`/`utility`). No behaviour.
- **`model.py`** (`PelotonModel`): 1-D `ContinuousSpace` (riders are points at x = distance travelled; y is viz-only). `step()` re-detects packs by geometry each step, resolves breakaways/follows, then advances each pack. `self.riders` keeps every rider in spawn order (never pruned; `model.agents` drops finishers) so evolution can read coefficients across races.
- **`group.py`**: pure pack functions — `detect_groups` (single-linkage by `group_radius` along x), `group_speed` (contribution-weighted), `draft_factors` (per-rider `cf_eff` from leadership fraction). Group-size→draft-factor table is from Blocken et al. 2018.
- **`energy.py`**: pure physiology — Olds power equation `P = k_aero·cf_eff·v³ + c_roll·v`, Newton inversion `solo_speed`, critical power, and W' (anaerobic capacity) drain/recover dynamics. No Mesa.
- **`strategy.py`**: three logistic decision heads — `contribution` (coop), `breakaway_prob` (leave), `follow_prob`. Each is `sigmoid(α + β·d_finish/L + γ·T + δ·E)`; the **learned coefficients are α/β/γ/δ per head** (12 numbers/rider), not SA parameters. Note `contribution` uses `(1 − W'/W_full)` (spent riders pull more) while breakaway/follow use `W'/W_full` (fresh riders attack).
- **`evolution.py`**: across-race learning. `run_generations` is the outer loop carrying a coefficient population between races; `evolve` is the update rule — the worst `evo_bottom_frac` of riders imitate (blend by `imitation_mu` toward) a roulette-picked donor from the top `evo_top_frac`, plus Gaussian noise. Utility = `exp(-utility_decay·finish_position)`, DNF = 0.
- **`recorder.py`** (`dump_run`): full per-step per-agent trace for offline analysis.
- **`sweep.py`** / **`gsa.py`** / **`gsa_dump.py`**: `batch_run` replicates; SALib Morris (screen all live knobs) + Sobol (focused subset). See each module docstring for which knobs are live vs. dead/fixed.
- **`viz.py`** + `run_app.py`: SolaraViz components. The model self-loads `PELOTON_POPULATION` env var on every (re)instantiation so a learned field survives SolaraViz's slider-reset.

### Gotchas worth knowing
- `learning_rate` and `elite_fraction` in `PelotonConfig` are **dead** (the imitation rule replaced the old similarity-weighted pull); `road_width`/`rider_*` are viz-only.
- SolaraViz reset injects `scenario=`/`rng=` kwargs; the model consumes/reroutes them in `__init__` — keep that handling if you touch the constructor.
- `ContinuousSpace` treats `x_max` as exclusive, hence the `+1e-6` pad so `road_length` itself is a legal finish position.
