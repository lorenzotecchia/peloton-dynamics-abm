#!/usr/bin/env bash
set -euo pipefail

DIR="analysis_output/v-and-v"

uv run python main.py dump --out-dir "$DIR"

uv run python scripts/plot_race_position.py --dir "$DIR" --mode abs
uv run python scripts/plot_rank_detail.py    --dir "$DIR" --rank 1
uv run python scripts/plot_rank_detail.py    --dir "$DIR" --rank 2
