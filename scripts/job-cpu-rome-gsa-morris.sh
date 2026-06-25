#!/usr/bin/env bash
# Run the Morris (cheap screening) GSA on one Snellius Rome CPU node.
#
# Pure-Python/Mesa model (no GPU): the `rome` partition, with the GSA driver
# fanning samples across the node's 128 cores.
#
# Usage (from the project root, on a login node):
#   bash scripts/job-cpu-rome-gsa-morris.sh [SAMPLES] [REPLICATES] [GENERATIONS] \
#                                           [MAX_STEPS] [ROAD_LENGTH] [DT] [GROUP_RADIUS]
# Defaults: SAMPLES=20, REPLICATES=10, GENERATIONS=150, MAX_STEPS=2500,
#           ROAD_LENGTH=10000 (10 km), DT=2 (2 s), GROUP_RADIUS=3.
# The scenario defaults (ROAD_LENGTH/DT/GROUP_RADIUS) match PelotonConfig.
# Morris screens all 18 model knobs and SAMPLES is the trajectory count r: the
# screening literature uses r~10-50, so 20 is a solid default (not the Sobol-sized
# 512). Total ~SAMPLES*(D+1) runs (D=18 params), each over GENERATIONS races x
# REPLICATES seeds. Output goes to <GSA_OUT_BASE>/<SLURM JOB ID>-<GIT HASH>-morris/
# (GSA_OUT_BASE defaults to /gpfs/work5/0/prjs2142/gsa-agent-dump-per-run);
# gsa_morris.csv lands there. Logs in jobs/logs/.

set -euo pipefail

SAMPLES="${1:-20}"
REPLICATES="${2:-10}"
GENERATIONS="${3:-150}"
MAX_STEPS="${4:-2500}"
ROAD_LENGTH="${5:-10000}"
DT="${6:-2}"
GROUP_RADIUS="${7:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

sbatch --job-name=peloton-gsa-morris \
  --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=128 \
  --time=08:00:00 \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-gsa-morris-%j.out \
  --error=jobs/logs/peloton-gsa-morris-%j.err \
  --export=ALL,METHOD=morris,SAMPLES="$SAMPLES",REPLICATES="$REPLICATES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",ROAD_LENGTH="$ROAD_LENGTH",DT="$DT",GROUP_RADIUS="$GROUP_RADIUS" \
  "$SCRIPT_DIR/_gsa-job-body.sh"
