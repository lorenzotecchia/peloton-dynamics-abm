#!/usr/bin/env bash
# Run the Sobol GSA as a Slurm ARRAY: split the design matrix into K chunks, each
# evaluated by its own array task (one full Rome node), then a merge job (which
# depends on the array via afterok) computes the indices. Wall-clock ~= the
# single-node time / K, because the K chunks run on K nodes in parallel.
#
# Flow:
#   login node  : `--mode sample` writes X.npy (+ X_index.csv)            (seconds)
#   array 0..K-1: `--mode evaluate`, each does a row chunk -> Y_<s>_<e>.npy
#   merge job   : `--mode merge` concatenates the Y chunks -> gsa_sobol.csv
#
# Usage (from the project root, on a login node; run with bash, NOT sbatch):
#   bash scripts/job-cpu-rome-gsa-sobol-array.sh [SAMPLES] [REPLICATES] [GENERATIONS] \
#                                                [MAX_STEPS] [K] [ROAD_LENGTH] [DT] [GROUP_RADIUS]
# Defaults: SAMPLES=512, REPLICATES=10, GENERATIONS=150, MAX_STEPS=2500, K=8,
#           ROAD_LENGTH=10000 (10 km), DT=2 (2 s), GROUP_RADIUS=3 (match PelotonConfig).
# Env overrides: GSA_OUT_BASE (default /gpfs/work5/0/prjs2142/gsa-agent-dump-per-run),
#                TASK_TIME (per-chunk wall time, default 12:00:00).
# Output: <GSA_OUT_BASE>/<ARRAY JOB ID>-<GIT HASH>-sobol/gsa_sobol.csv (+ sample_index.csv).

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: launch with 'bash', not 'sbatch' (this wrapper submits the jobs itself):" >&2
    echo "  bash scripts/job-cpu-rome-gsa-sobol-array.sh [SAMPLES] ..." >&2
    exit 1
fi

METHOD=sobol
SAMPLES="${1:-512}"
REPLICATES="${2:-10}"
GENERATIONS="${3:-150}"
MAX_STEPS="${4:-2500}"
K="${5:-8}"                 # number of array chunks (= nodes used in parallel)
ROAD_LENGTH="${6:-10000}"
DT="${7:-2}"
GROUP_RADIUS="${8:-3}"
GSA_OUT_BASE="${GSA_OUT_BASE:-/gpfs/work5/0/prjs2142/gsa-agent-dump-per-run}"
TASK_TIME="${TASK_TIME:-12:00:00}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

GIT_HASH="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"

# Shared work dir (must be on a shared FS visible to every node): X.npy + Y chunks.
WORK_DIR="$PROJECT_ROOT/jobs/gsa-work/${METHOD}-$(date +%Y%m%d_%H%M%S)-${GIT_HASH}"
mkdir -p "$WORK_DIR"

# ── Step 1: sample X on the login node (seconds) ─────────────────────────────
echo "[array] $METHOD: sampling design matrix X (N=$SAMPLES) ..."
uv run python -m peloton.gsa --mode sample --method "$METHOD" \
  --samples "$SAMPLES" --x-file "$WORK_DIR/X.npy"

N_ROWS="$(uv run python -c "import numpy as np; print(len(np.load('$WORK_DIR/X.npy')))")"
CHUNK_SIZE=$(( (N_ROWS + K - 1) / K ))   # ceiling division -> chunks tile [0, N_ROWS)
echo "[array] N_ROWS=$N_ROWS  K=$K  chunk_size=$CHUNK_SIZE  work_dir=$WORK_DIR"

# ── Step 2: submit the evaluation array (K tasks, one full node each) ─────────
ARRAY_JOB_ID="$(sbatch --parsable \
  --job-name=peloton-gsa-${METHOD} \
  --partition=rome --nodes=1 --ntasks=1 --gpus=0 --cpus-per-task=128 \
  --time="$TASK_TIME" \
  --array="0-$(( K - 1 ))" \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-gsa-${METHOD}-%A_%a.out \
  --error=jobs/logs/peloton-gsa-${METHOD}-%A_%a.err \
  --export=ALL,METHOD="$METHOD",WORK_DIR="$WORK_DIR",N_ROWS="$N_ROWS",CHUNK_SIZE="$CHUNK_SIZE",REPLICATES="$REPLICATES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",ROAD_LENGTH="$ROAD_LENGTH",DT="$DT",GROUP_RADIUS="$GROUP_RADIUS" \
  "$SCRIPT_DIR/_gsa-array-eval.sh")"
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"  # --parsable may append ";cluster"

# ── Step 3: submit the merge, gated on every array task succeeding ────────────
OUT_DIR="${GSA_OUT_BASE}/${ARRAY_JOB_ID}-${GIT_HASH}-${METHOD}"
MERGE_JOB_ID="$(sbatch --parsable \
  --job-name=peloton-gsa-${METHOD}-merge \
  --partition=rome --nodes=1 --ntasks=1 --gpus=0 --cpus-per-task=1 \
  --time=00:30:00 \
  --dependency="afterok:${ARRAY_JOB_ID}" \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-gsa-${METHOD}-merge-${ARRAY_JOB_ID}.out \
  --error=jobs/logs/peloton-gsa-${METHOD}-merge-${ARRAY_JOB_ID}.err \
  --export=ALL,METHOD="$METHOD",WORK_DIR="$WORK_DIR",OUT_DIR="$OUT_DIR" \
  "$SCRIPT_DIR/_gsa-array-merge.sh")"
MERGE_JOB_ID="${MERGE_JOB_ID%%;*}"

MERGE_OUT_FILE="$PROJECT_ROOT/jobs/logs/peloton-gsa-${METHOD}-merge-${ARRAY_JOB_ID}.out"
MERGE_ERR_FILE="$PROJECT_ROOT/jobs/logs/peloton-gsa-${METHOD}-merge-${ARRAY_JOB_ID}.err"

echo ""
echo "[summary] $METHOD GSA split into $K chunks of ~$CHUNK_SIZE rows ($N_ROWS total)"
echo "  eval array : $ARRAY_JOB_ID  (tasks 0-$(( K - 1 )), 128 cores each)"
echo "  merge job  : $MERGE_JOB_ID  (runs afterok the whole array)"
echo "  result     : $OUT_DIR/gsa_${METHOD}.csv"
echo "  work dir   : $WORK_DIR  (X.npy + Y chunks; safe to delete after merge)"
echo "  task logs  : $PROJECT_ROOT/jobs/logs/peloton-gsa-${METHOD}-${ARRAY_JOB_ID}_*.out"
echo ""

# Watch the pipeline (eval array -> merge) and exit when it ends.
exec "$SCRIPT_DIR/_gsa-array-watch.sh" \
  "$ARRAY_JOB_ID" "$MERGE_JOB_ID" "$METHOD" "$MERGE_OUT_FILE" "$MERGE_ERR_FILE" "$OUT_DIR"
