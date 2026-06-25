#!/usr/bin/env bash
# DRY RUN of the Morris GSA: same pipeline as job-cpu-rome-gsa-morris.sh but with
# tiny SAMPLES/REPLICATES/GENERATIONS and a short wall time, to validate the
# end-to-end plumbing (sampling -> sim -> indices -> output dir) cheaply before
# committing to the full run. Output is routed under a dryrun/ subdir so it never
# mixes with production results.
#
# Usage (from the project root, on a login node):
#   bash scripts/job-cpu-rome-gsa-morris-dryrun.sh [SAMPLES] [REPLICATES] [GENERATIONS] \
#                                                  [MAX_STEPS] [ROAD_LENGTH] [DT] [GROUP_RADIUS]
# Defaults: SAMPLES=2, REPLICATES=2, GENERATIONS=3, MAX_STEPS=2500,
#           ROAD_LENGTH=10000 (10 km), DT=2 (2 s), GROUP_RADIUS=3.
# Morris does ~SAMPLES*(D+1) runs (D=18); at SAMPLES=2 that's ~38 samples.
# Output: <GSA_OUT_BASE>/<SLURM JOB ID>-<GIT HASH>-morris/, with GSA_OUT_BASE
# defaulting to /gpfs/work5/0/prjs2142/gsa-agent-dump-per-run/dryrun. Logs in jobs/logs/.

set -euo pipefail

SAMPLES="${1:-2}"
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

JOB_ID="$(sbatch --parsable --job-name=peloton-gsa-morris-dryrun \
  --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=128 \
  --time=00:30:00 \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-gsa-morris-dryrun-%j.out \
  --error=jobs/logs/peloton-gsa-morris-dryrun-%j.err \
  --export=ALL,METHOD=morris,SAMPLES="$SAMPLES",REPLICATES="$REPLICATES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",ROAD_LENGTH="$ROAD_LENGTH",DT="$DT",GROUP_RADIUS="$GROUP_RADIUS",GSA_OUT_BASE="$GSA_OUT_BASE" \
  "$SCRIPT_DIR/_gsa-job-body.sh")"
JOB_ID="${JOB_ID%%;*}"  # --parsable may append ";cluster"; keep just the id

OUT_FILE="$PROJECT_ROOT/jobs/logs/peloton-gsa-morris-dryrun-${JOB_ID}.out"
ERR_FILE="$PROJECT_ROOT/jobs/logs/peloton-gsa-morris-dryrun-${JOB_ID}.err"
echo "Submitted Morris GSA DRY RUN as job $JOB_ID"
echo "  stdout: $OUT_FILE"
echo "  stderr: $ERR_FILE"
echo "Waiting for job $JOB_ID to start and create its log (appears once it leaves"
echo "the queue and begins running)..."
while [[ ! -e "$OUT_FILE" ]]; do
  # Bail out if the job left the queue without ever producing a log (failed to start).
  if [[ -z "$(squeue -h -j "$JOB_ID" -o '%T' 2>/dev/null)" ]]; then
    echo "Job $JOB_ID is no longer queued but $OUT_FILE never appeared." >&2
    echo "Check 'sacct -j $JOB_ID' and $ERR_FILE." >&2
    exit 1
  fi
  sleep 2
done
echo "Log is live. Streaming stdout until the job finishes"
echo "(Ctrl-C stops watching; the job keeps running on the cluster)..."
echo "----------------------------------------------------------------------"
tail -n +1 -F "$OUT_FILE" &
TAIL_PID=$!

# Poll Slurm until the job leaves the active queue (no PENDING/RUNNING/COMPLETING).
while squeue -h -j "$JOB_ID" -o '%T' 2>/dev/null | grep -q .; do
  sleep 10
done

sleep 3                                  # let tail flush the final lines
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true
echo "----------------------------------------------------------------------"

# Final disposition from the accounting DB (-X = the job allocation, not steps).
STATE="$(sacct -j "$JOB_ID" -n -X -o State 2>/dev/null | head -n1 | tr -d ' ')"
case "$STATE" in
  COMPLETED)     echo "Job $JOB_ID COMPLETED. stdout: $OUT_FILE" ;;
  ""|UNKNOWN)    echo "Job $JOB_ID ended; sacct gave no state. Check 'sacct -j $JOB_ID' and $ERR_FILE." ;;
  *)             echo "Job $JOB_ID ended in state $STATE -- inspect $ERR_FILE and 'sacct -j $JOB_ID --format=State,ExitCode,Elapsed'." >&2; exit 1 ;;
esac
