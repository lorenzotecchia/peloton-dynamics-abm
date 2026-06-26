#!/usr/bin/env bash
# Parallel (Slurm ARRAY) version of job-cpu-rome-gsa-sobol-dump.sh. Same full
# per-agent agent-dump, but the Sobol design matrix is split into K chunks, each
# dumped by its own array task (one full Rome node), so the wall-clock is roughly
# the single-node time / K. Uses the same row-chunk trick as
# job-cpu-rome-gsa-sobol-array.sh, but with NO merge step: every sample_<i>/ dir
# is self-contained, so the K tasks just write their chunks into one shared out
# dir (keyed by the array job id) and we're done when the array finishes.
#
# Flow:
#   login node  : `--mode sample` writes X.npy (+ X_index.csv)             (seconds)
#   array 0..K-1: `--mode dump`, each dumps a row chunk -> sample_<i>/ dirs
#
# Usage (from the project root, on a login node; run with bash, NOT sbatch):
#   bash scripts/job-cpu-rome-gsa-sobol-dump-array.sh [SAMPLES] [GENERATIONS] \
#                            [MAX_STEPS] [K] [ROAD_LENGTH] [DT] [GROUP_RADIUS]
# Defaults: SAMPLES=64, GENERATIONS=150, MAX_STEPS=2000, K=8, ROAD_LENGTH=10000
#           (10 km), DT=2 (2 s), GROUP_RADIUS=3. No replication (every race seed 0).
# Env overrides: GSA_OUT_BASE (default /gpfs/work5/0/prjs2142/gsa-agent-dump-per-run),
#                TASK_TIME (per-chunk wall time, default 02:00:00),
#                PARQUET (1=Parquet default, 0=CSV).
# Output: <GSA_OUT_BASE>/<ARRAY JOB ID>-<GIT HASH>-sobol-dump/sobol/sample_<i>/...
# Completed sample dirs persist as they finish, so a wall-clock kill only loses
# the samples still in flight.

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: launch with 'bash', not 'sbatch' (this wrapper submits the job itself):" >&2
    echo "  bash scripts/job-cpu-rome-gsa-sobol-dump-array.sh [SAMPLES] ..." >&2
    exit 1
fi

METHOD=sobol
SAMPLES="${1:-64}"
GENERATIONS="${2:-150}"
MAX_STEPS="${3:-2000}"
K="${4:-8}"                 # number of array chunks (= nodes used in parallel)
ROAD_LENGTH="${5:-10000}"
DT="${6:-2}"
GROUP_RADIUS="${7:-3}"
GSA_OUT_BASE="${GSA_OUT_BASE:-/gpfs/work5/0/prjs2142/gsa-agent-dump-per-run}"
TASK_TIME="${TASK_TIME:-02:00:00}"
PARQUET="${PARQUET:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

GIT_HASH="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"

# Shared work dir (must be on a shared FS visible to every node): holds X.npy that
# all array tasks load. Out dir is keyed by the array job id (below).
WORK_DIR="$PROJECT_ROOT/jobs/gsa-work/${METHOD}-dump-$(date +%Y%m%d_%H%M%S)-${GIT_HASH}"
mkdir -p "$WORK_DIR"

# ── Step 1: sample X on the login node (seconds) ─────────────────────────────
echo "[array] $METHOD dump: sampling design matrix X (N=$SAMPLES) ..."
uv run python -m peloton.gsa_dump --mode sample --method "$METHOD" \
  --samples "$SAMPLES" --x-file "$WORK_DIR/X.npy"

N_ROWS="$(uv run python -c "import numpy as np; print(len(np.load('$WORK_DIR/X.npy')))")"
CHUNK_SIZE=$(( (N_ROWS + K - 1) / K ))   # ceiling division -> chunks tile [0, N_ROWS)
echo "[array] N_ROWS=$N_ROWS  K=$K  chunk_size=$CHUNK_SIZE  work_dir=$WORK_DIR"

# ── Step 2: submit the dump array (K tasks, one full node each) ───────────────
# The task body derives the shared out dir from SLURM_ARRAY_JOB_ID, so all chunks
# land together without us knowing the id up front.
ARRAY_JOB_ID="$(sbatch --parsable \
  --job-name=peloton-gsa-${METHOD}-dump \
  --partition=rome --nodes=1 --ntasks=1 --gpus=0 --cpus-per-task=128 \
  --time="$TASK_TIME" \
  --array="0-$(( K - 1 ))" \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-gsa-${METHOD}-dump-%A_%a.out \
  --error=jobs/logs/peloton-gsa-${METHOD}-dump-%A_%a.err \
  --export=ALL,METHOD="$METHOD",WORK_DIR="$WORK_DIR",N_ROWS="$N_ROWS",CHUNK_SIZE="$CHUNK_SIZE",SAMPLES="$SAMPLES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",ROAD_LENGTH="$ROAD_LENGTH",DT="$DT",GROUP_RADIUS="$GROUP_RADIUS",PARQUET="$PARQUET" \
  "$SCRIPT_DIR/_gsa-dump-array-body.sh")"
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"  # --parsable may append ";cluster"

# Out dir matches the one the task body builds from SLURM_ARRAY_JOB_ID (== ARRAY_JOB_ID).
OUT_DIR="${GSA_OUT_BASE}/${ARRAY_JOB_ID}-${GIT_HASH}-${METHOD}-dump"

# Drop the design matrix beside the dumps for provenance (tasks don't write it).
mkdir -p "$OUT_DIR/$METHOD"
cp "$WORK_DIR/X.npy" "$OUT_DIR/$METHOD/X.npy" 2>/dev/null || true
cp "$WORK_DIR/X_index.csv" "$OUT_DIR/$METHOD/sample_index.csv" 2>/dev/null || true

echo ""
echo "[summary] $METHOD agent-dump split into $K chunks of ~$CHUNK_SIZE rows ($N_ROWS total)"
echo "  dump array : $ARRAY_JOB_ID  (tasks 0-$(( K - 1 )), 128 cores each)"
echo "  result     : $OUT_DIR/$METHOD/sample_<i>/"
echo "  work dir   : $WORK_DIR  (X.npy; safe to delete after the array finishes)"
echo "  task logs  : $PROJECT_ROOT/jobs/logs/peloton-gsa-${METHOD}-dump-${ARRAY_JOB_ID}_*.out"
echo ""

# ── Step 3: watch the array until every task leaves the queue ────────────────
echo "Watching dump array $ARRAY_JOB_ID until all tasks finish (Ctrl-C stops"
echo "watching; the jobs keep running on the cluster)..."
while [[ -n "$(squeue -h -j "$ARRAY_JOB_ID" -o '%T' 2>/dev/null)" ]]; do
    states="$(squeue -h -j "$ARRAY_JOB_ID" -o '%T' 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
    echo "  [$(date +%H:%M:%S)] array $ARRAY_JOB_ID: ${states:-<none active>}"
    sleep 30
done

# Final disposition from the accounting DB (-X = the job allocations, not steps).
ASTATES="$(sacct -j "$ARRAY_JOB_ID" -n -X -o State 2>/dev/null | tr -d ' ' | sort -u)"
echo "Dump array $ARRAY_JOB_ID finished; task states: $(echo "$ASTATES" | tr '\n' ' ')"
if [[ -z "$ASTATES" ]]; then
    echo "sacct returned no state. Check 'sacct -j $ARRAY_JOB_ID' and the task .err logs." >&2
    exit 1
elif [[ "$ASTATES" != "COMPLETED" ]]; then
    echo "Some tasks did not COMPLETE -- inspect the task logs in $PROJECT_ROOT/jobs/logs" >&2
    echo "and 'sacct -j $ARRAY_JOB_ID --format=JobID,State,ExitCode,Elapsed'." >&2
    exit 1
fi
echo "All chunks COMPLETED. Dumps under $OUT_DIR/$METHOD/"
