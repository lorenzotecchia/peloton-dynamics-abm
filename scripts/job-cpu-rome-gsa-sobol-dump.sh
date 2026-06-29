#!/usr/bin/env bash
# Dump full per-agent state across a small Sobol sample on one Snellius Rome CPU
# node (1-hour walltime). Like job-cpu-rome-gsa-sobol.sh but it computes NO
# indices: it writes the complete main.py-dump bundle for every generation of
# every sample, so the SA can be recomputed from the raw data later.
#
# Usage (from the project root, on a login node):
#   bash scripts/job-cpu-rome-gsa-sobol-dump.sh [SAMPLES] [GENERATIONS] \
#                                [MAX_STEPS] [ROAD_LENGTH] [DT] [GROUP_RADIUS]
# Defaults: SAMPLES=512, GENERATIONS=150, MAX_STEPS=2000, ROAD_LENGTH=10000 (10 km),
#           DT=2 (2 s), GROUP_RADIUS=3. No replication (every race uses seed 0).
# Sobol draws ~SAMPLES*(D+2) design rows (D=5; use a power of 2 for SAMPLES), each
# run over GENERATIONS races and dumped in full: SAMPLES=512 -> ~3584 rows x 150
# gens ~= 538k races (runtime tracks rows*gens; calibrated from ~12 min at 7168
# rows x 10 gens). Fewer parameter points but a full 150-generation learning
# trajectory dumped per point. Completed sample dirs
# persist as they finish, so a wall-clock kill only loses the samples still in
# flight. Output goes to
# <GSA_OUT_BASE>/<SLURM JOB ID>-<GIT HASH>-sobol-dump/ (GSA_OUT_BASE defaults to
# /gpfs/work5/0/prjs2142/gsa-agent-dump-per-run). Writes Parquet by default
# (smaller); set PARQUET=0 for CSV. Logs in jobs/logs/.
#
# By default only the FINAL generation's race is dumped per sample (the full
# learning loop still runs, so the dumped race is the converged one). Set
# DUMP_ALL_GENERATIONS=1 to dump every generation's full per-step per-agent state
# (the original behaviour; far more disk).

set -euo pipefail

SAMPLES="${1:-512}"
GENERATIONS="${2:-150}"
MAX_STEPS="${3:-2000}"
ROAD_LENGTH="${4:-10000}"
DT="${5:-2}"
GROUP_RADIUS="${6:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

JOB_ID="$(sbatch --parsable --job-name=peloton-gsa-sobol-dump \
  --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=128 \
  --time=04:00:00 \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-gsa-sobol-dump-%j.out \
  --error=jobs/logs/peloton-gsa-sobol-dump-%j.err \
  --export=ALL,METHOD=sobol,SAMPLES="$SAMPLES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",ROAD_LENGTH="$ROAD_LENGTH",DT="$DT",GROUP_RADIUS="$GROUP_RADIUS",PARQUET="${PARQUET:-1}",DUMP_ALL_GENERATIONS="${DUMP_ALL_GENERATIONS:-0}" \
  "$SCRIPT_DIR/_gsa-dump-job-body.sh")"
JOB_ID="${JOB_ID%%;*}"  # --parsable may append ";cluster"; keep just the id

OUT_FILE="$PROJECT_ROOT/jobs/logs/peloton-gsa-sobol-dump-${JOB_ID}.out"
ERR_FILE="$PROJECT_ROOT/jobs/logs/peloton-gsa-sobol-dump-${JOB_ID}.err"
echo "Submitted Sobol agent-dump as job $JOB_ID"
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
