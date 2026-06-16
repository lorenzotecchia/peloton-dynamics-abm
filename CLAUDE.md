# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent-Based Model of competitive cycling (peloton dynamics) using [Mesa](https://mesa.readthedocs.io/). Cyclists are agents that evolve cooperation/defection strategies through evolutionary game theory. Key behaviours to model: drafting (energy savings behind other riders), breakaway formation, lead-out trains, and sprint finish positioning.

The simulation is intended to run on [Snellius](https://www.surf.nl/en/services/snellius-the-national-supercomputer) (SURF national HPC cluster) via Slurm.

## Commands

```bash
# Install dependencies (creates .venv from uv.lock)
uv sync

# Run entry point
uv run python main.py

# Run a specific script
uv run python <script>.py
```

Python version: 3.12+ (see `.python-version`).

## Snellius / HPC Workflow

Setup (run **once** on login node, never on compute node):
```bash
bash scripts/snellius-py-runtime-setup.sh
```

Submit Slurm jobs (do **not** run training/simulation on the login node):
```bash
# GPU H100 — interactive allocation
bash scripts/job-gpu-h100-salloc.sh      # 1 hr default
bash scripts/job-gpu-h100-salloc-08hr.sh # 8 hr

# GPU H100 — batch
bash scripts/job-gpu-h100-sbatch.sh

# CPU Rome — interactive / batch equivalents exist too
```

Job logs land in `jobs/logs/`.

## Architecture

The project is in early MVP stage. `main.py` is a placeholder. The intended Mesa structure is:

- **Agents** (`CyclistAgent`): carry state (energy, velocity, strategy probabilities α/β/γ), decide each step whether to cooperate (share pace-setting work), defect (sit in), or break away.
- **Model** (`PelotonModel`): 1-D or 2-D continuous space, hosts cyclists, advances time steps, collects data.
- **Strategy update**: evolutionary game theory — payoff matrix drives probability updates each generation.

Key physics constants to keep consistent:
- Drafting reduces air resistance via `CF_draft = 0.62 - 0.0104·d_w + 0.0452·d_w²` (wheel-to-wheel distance in metres).
- Riders are considered in the same group when within 3 m of each other.
- Cooperation probability for rider *i* is parameterised as `σᵢ(αᵢ + βᵢ·d_finish + γᵢ·E_left)`; α/β/γ are the learned outputs, not sensitivity-analysis parameters.
