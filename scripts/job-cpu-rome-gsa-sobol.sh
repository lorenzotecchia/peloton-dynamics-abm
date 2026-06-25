#!/usr/bin/env bash
# Run the Sobol (variance-based first/total-order) GSA on one Snellius Rome CPU node.
#
# Pure-Python/Mesa model (no GPU): the `rome` partition, with the GSA driver
# fanning samples across the node's 128 cores.
#
# Usage (from the project root, on a login node):
#   bash scripts/job-cpu-rome-gsa-sobol.sh [SAMPLES] [REPLICATES] [GENERATIONS] \
#                                          [MAX_STEPS] [ROAD_LENGTH] [DT] [GROUP_RADIUS]
# Defaults: SAMPLES=512, REPLICATES=10, GENERATIONS=150, MAX_STEPS=2500,
#           ROAD_LENGTH=10000 (10 km), DT=2 (2 s), GROUP_RADIUS=3.
# The scenario defaults (ROAD_LENGTH/DT/GROUP_RADIUS) match PelotonConfig.
# Sobol does ~SAMPLES*(D+2) runs (D=5 params; use a power of 2 for SAMPLES),
# each over GENERATIONS races x REPLICATES seeds. Output goes to
# <GSA_OUT_BASE>/<SLURM JOB ID>-<GIT HASH>-sobol/ (GSA_OUT_BASE defaults to
# /gpfs/work5/0/prjs2142/gsa-agent-dump-per-run); gsa_sobol.csv lands there.
# Logs in jobs/logs/.

set -euo pipefail

SAMPLES="${1:-512}"
REPLICATES="${2:-10}"
GENERATIONS="${3:-150}"
MAX_STEPS="${4:-2500}"
ROAD_LENGTH="${5:-10000}"
DT="${6:-2}"
GROUP_RADIUS="${7:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

JOB_ID="$(sbatch --parsable --job-name=peloton-gsa-sobol \
  --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=128 \
  --time=12:00:00 \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-gsa-sobol-%j.out \
  --error=jobs/logs/peloton-gsa-sobol-%j.err \
  --export=ALL,METHOD=sobol,SAMPLES="$SAMPLES",REPLICATES="$REPLICATES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",ROAD_LENGTH="$ROAD_LENGTH",DT="$DT",GROUP_RADIUS="$GROUP_RADIUS" \
  "$SCRIPT_DIR/_gsa-job-body.sh")"
JOB_ID="${JOB_ID%%;*}"  # --parsable may append ";cluster"; keep just the id

OUT_FILE="$PROJECT_ROOT/jobs/logs/peloton-gsa-sobol-${JOB_ID}.out"
ERR_FILE="$PROJECT_ROOT/jobs/logs/peloton-gsa-sobol-${JOB_ID}.err"
echo "Submitted Sobol GSA as job $JOB_ID"
echo "  stdout: $OUT_FILE"
echo "  stderr: $ERR_FILE"
echo "Tailing stdout (Ctrl-C stops the tail, NOT the job; waits while queued)..."
exec tail -F "$OUT_FILE"
