#!/usr/bin/env bash
# Run the Morris (cheap screening) GSA on one Snellius Rome CPU node.
#
# Pure-Python/Mesa model (no GPU): the `rome` partition, with the GSA driver
# fanning samples across the node's 128 cores.
#
# Usage (from the project root, on a login node):
#   bash scripts/job-cpu-rome-gsa-morris.sh [SAMPLES] [REPLICATES] [GENERATIONS] \
#                                           [MAX_STEPS] [ROAD_LENGTH] [DT] [GROUP_RADIUS]
# Defaults: SAMPLES=512, REPLICATES=5, GENERATIONS=150, MAX_STEPS=500,
#           ROAD_LENGTH=100000 (100 km), DT=60 (1 min), GROUP_RADIUS=200.
# Morris does ~SAMPLES*(D+1) runs (D=4 params), each over GENERATIONS races x
# REPLICATES seeds. Indices land in data/gsa_morris.csv; logs in jobs/logs/.

set -euo pipefail

SAMPLES="${1:-512}"
REPLICATES="${2:-5}"
GENERATIONS="${3:-150}"
MAX_STEPS="${4:-500}"
ROAD_LENGTH="${5:-100000}"
DT="${6:-60}"
GROUP_RADIUS="${7:-200}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

sbatch --job-name=peloton-gsa-morris \
  --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=128 \
  --time=08:00:00 \
  --mail-user=lorenzo.tecchia@student.uva.nl \
  --mail-type=END,FAIL \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-gsa-morris-%j.out \
  --error=jobs/logs/peloton-gsa-morris-%j.err \
  --export=ALL,METHOD=morris,SAMPLES="$SAMPLES",REPLICATES="$REPLICATES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",ROAD_LENGTH="$ROAD_LENGTH",DT="$DT",GROUP_RADIUS="$GROUP_RADIUS" \
  "$SCRIPT_DIR/_gsa-job-body.sh"
