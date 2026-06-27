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
  --time=48:00:00 \
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
echo "Log is live. Streaming stdout until the job finishes."
echo "A per-sample progress line ([k/total] pct, elapsed, eta, samples/hr) is"
echo "printed as the design matrix is evaluated, so you can gauge the time left."
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
