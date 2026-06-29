#!/usr/bin/env bash
set -euo pipefail

# generate data (default: collect all last generation per evolution. you can dump agent state per step per generation per evolution as well but the storage is really large)
./scripts/job-cpu-rome-gsa-sobol-dump.sh
# or, speed up
#./scripts/job-cpu-rome-gsa-sobol-dump-array.sh

# note: plotting is performed on snellius due to large size data set
uv run python scripts/gsa_sobol_heatmaps.py \
    --sobol-dir /gpfs/work5/0/prjs2142/gsa-agent-dump-per-run/24244290-94c6e10-sobol-dump \
    --out-dir data-SA/gsa_heatmaps-sobol-518-all-bins-filled-seq \
    --bins 10 \
    --slice-nearest 100000

# 224G, 10 mins - more samples needed to fill-in the heatmap
DUMP_ALL_GENERATIONS=1 TASK_TIME=00:15:00 bash scripts/job-cpu-rome-gsa-sobol-dump-array.sh 64 100

uv run python scripts/gsa_sobol_heatmaps.py \
    --sobol-dir /gpfs/work5/0/prjs2142/gsa-agent-dump-per-run/24277275-9b1b60f-sobol-dump/sobol/ \
   --out-dir data-SA/gsa_heatmaps-sobol-sample64-gen100 \
   --bins 10 \
   --slice-nearest 100000

