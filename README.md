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
[Mesa](https://mesa.readthedocs.io/), with a live [Solara](https://solara.dev/)
visualization. Cyclists drift into drafting formations on a scrolling road.

## Run locally

```bash
uv sync
uv run solara run run_app.py
```

## Deploy

This repo auto-deploys to Hugging Face Spaces (Docker SDK) via GitHub Actions —
see `.github/workflows/deploy-hf.yml`. The frontmatter above is what HF reads to
build the Space.
