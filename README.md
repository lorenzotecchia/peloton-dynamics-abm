---
title: Cycling Peloton MVP
emoji: 🚴
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Cycling Peloton ABM

Agent-based model of competitive cycling (peloton dynamics) built on
[Mesa](https://mesa.readthedocs.io/) 3.x, with a live
[Solara](https://solara.dev/) visualization. Cyclists drift into drafting
formations on a scrolling road and **learn** cooperation/breakaway strategies
across successive races (evolutionary imitation between runs).

## Setup

```bash
uv sync          # creates .venv from uv.lock (Python 3.12+)
```

## Usage

`main.py` is the CLI:

```bash
uv run python main.py solara                      # interactive viz (Ctrl-C to stop)
uv run python main.py run --max-steps 2000        # one headless race, prints finish order
uv run python main.py learn --generations 200     # race repeatedly, learning between races
uv run python main.py dump --max-steps 2000        # one race, full per-step per-agent trace
uv run python main.py test                        # run the test suite
```

`learn` writes a coefficient trajectory to `data/learning.csv` and the learned
field to `data/population.json`; replay it in the viz with
`uv run python main.py solara --population data/population.json`.

A 10 km race needs ~2000 steps to finish — the `run` default of 200 will not
cross the line.

## Sensitivity analysis (HPC)

Parameter sweeps and global sensitivity analysis (SALib Morris + Sobol) run on
[Snellius](https://www.surf.nl/en/services/snellius-the-national-supercomputer)
via Slurm — see `scripts/` and `CLAUDE.md` for the workflow. Locally:

```bash
uv run python -m peloton.sweep --runs 256 --out data/results.csv
uv run python -m peloton.gsa --method both --samples 512 --replicates 5 --generations 30
```

## Deploy

This repo auto-deploys to Hugging Face Spaces (Docker SDK) via GitHub Actions —
see `.github/workflows/deploy-hf.yml`. The frontmatter above is what HF reads to
build the Space.
