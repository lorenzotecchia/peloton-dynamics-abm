#!/usr/bin/env bash
# DRY RUN of the Sobol GSA: same pipeline as job-cpu-rome-gsa-sobol.sh but with
# tiny SAMPLES/REPLICATES/GENERATIONS and a short wall time, to validate the
# end-to-end plumbing (sampling -> sim -> indices -> output dir) cheaply before
# committing to the full run. Output is routed under a dryrun/ subdir so it never
# mixes with production results.
#
# Usage (from the project root, on a login node):
#   bash scripts/job-cpu-rome-gsa-sobol-dryrun.sh [SAMPLES] [REPLICATES] [GENERATIONS] \
#                                                 [MAX_STEPS] [ROAD_LENGTH] [DT] [GROUP_RADIUS]
# Defaults: SAMPLES=4, REPLICATES=2, GENERATIONS=3, MAX_STEPS=2500,
#           ROAD_LENGTH=10000 (10 km), DT=2 (2 s), GROUP_RADIUS=3.
# Sobol does ~SAMPLES*(D+2) runs (D=5; keep SAMPLES a power of 2); at SAMPLES=4
# that's ~28 samples. Output: <GSA_OUT_BASE>/<SLURM JOB ID>-<GIT HASH>-sobol/,
# with GSA_OUT_BASE defaulting to
# /gpfs/work5/0/prjs2142/gsa-agent-dump-per-run/dryrun. Logs in jobs/logs/.

set -euo pipefail

SAMPLES="${1:-4}"
REPLICATES="${2:-2}"
GENERATIONS="${3:-3}"
MAX_STEPS="${4:-2500}"
ROAD_LENGTH="${5:-10000}"
DT="${6:-2}"
GROUP_RADIUS="${7:-3}"

# Keep dry-run output out of the production base.
GSA_OUT_BASE="${GSA_OUT_BASE:-/gpfs/work5/0/prjs2142/gsa-agent-dump-per-run/dryrun}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

JOB_ID="$(sbatch --parsable --job-name=peloton-gsa-sobol-dryrun \
  --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=128 \
  --time=00:30:00 \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-gsa-sobol-dryrun-%j.out \
  --error=jobs/logs/peloton-gsa-sobol-dryrun-%j.err \
  --export=ALL,METHOD=sobol,SAMPLES="$SAMPLES",REPLICATES="$REPLICATES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",ROAD_LENGTH="$ROAD_LENGTH",DT="$DT",GROUP_RADIUS="$GROUP_RADIUS",GSA_OUT_BASE="$GSA_OUT_BASE" \
  "$SCRIPT_DIR/_gsa-job-body.sh")"
JOB_ID="${JOB_ID%%;*}"  # --parsable may append ";cluster"; keep just the id

OUT_FILE="$PROJECT_ROOT/jobs/logs/peloton-gsa-sobol-dryrun-${JOB_ID}.out"
ERR_FILE="$PROJECT_ROOT/jobs/logs/peloton-gsa-sobol-dryrun-${JOB_ID}.err"
echo "Submitted Sobol GSA DRY RUN as job $JOB_ID"
echo "  stdout: $OUT_FILE"
echo "  stderr: $ERR_FILE"
echo "Tailing stdout (Ctrl-C stops the tail, NOT the job; waits while queued)..."
exec tail -F "$OUT_FILE"
