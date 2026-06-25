#!/usr/bin/env bash
# DRY RUN of the Morris GSA ARRAY pipeline: same split sample->evaluate->merge flow
# as job-cpu-rome-gsa-morris-array.sh, but with the dry-run amount (SAMPLES=128,
# REPLICATES=5, GENERATIONS=3) and a short per-chunk wall time, to validate the
# whole array pipeline (chunking, the afterok merge, the watcher) cheaply. Output
# is routed under a dryrun/ subdir so it never mixes with production results.
#
# Usage (from the project root, on a login node; run with bash, NOT sbatch):
#   bash scripts/job-cpu-rome-gsa-morris-array-dryrun.sh [SAMPLES] [REPLICATES] [GENERATIONS] \
#                                                        [MAX_STEPS] [K] [ROAD_LENGTH] [DT] [GROUP_RADIUS]
# Defaults: SAMPLES=128, REPLICATES=5, GENERATIONS=3, MAX_STEPS=2500, K=2,
#           ROAD_LENGTH=10000 (10 km), DT=2 (2 s), GROUP_RADIUS=3 (match PelotonConfig).
# Morris screens all 18 knobs; SAMPLES is the trajectory count r (-> ~r*(D+1) rows).
# Env overrides: GSA_OUT_BASE (default .../gsa-agent-dump-per-run/dryrun),
#                TASK_TIME (per-chunk wall time, default 03:00:00).
# Output: <GSA_OUT_BASE>/<ARRAY JOB ID>-<GIT HASH>-morris/gsa_morris.csv (+ sample_index.csv).

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: launch with 'bash', not 'sbatch' (this wrapper submits the jobs itself):" >&2
    echo "  bash scripts/job-cpu-rome-gsa-morris-array-dryrun.sh [SAMPLES] ..." >&2
    exit 1
fi

METHOD=morris
NAME="peloton-gsa-${METHOD}-dryrun"
SAMPLES="${1:-128}"
REPLICATES="${2:-5}"
GENERATIONS="${3:-3}"
MAX_STEPS="${4:-2500}"
K="${5:-2}"                 # number of array chunks (= nodes used in parallel)
ROAD_LENGTH="${6:-10000}"
DT="${7:-2}"
GROUP_RADIUS="${8:-3}"
GSA_OUT_BASE="${GSA_OUT_BASE:-/gpfs/work5/0/prjs2142/gsa-agent-dump-per-run/dryrun}"
TASK_TIME="${TASK_TIME:-03:00:00}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

GIT_HASH="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"

# Shared work dir (must be on a shared FS visible to every node): X.npy + Y chunks.
WORK_DIR="$PROJECT_ROOT/jobs/gsa-work/${METHOD}-dryrun-$(date +%Y%m%d_%H%M%S)-${GIT_HASH}"
mkdir -p "$WORK_DIR"

# ── Step 1: sample X on the login node (seconds) ─────────────────────────────
echo "[array-dryrun] $METHOD: sampling design matrix X (N=$SAMPLES) ..."
uv run python -m peloton.gsa --mode sample --method "$METHOD" \
  --samples "$SAMPLES" --x-file "$WORK_DIR/X.npy"

N_ROWS="$(uv run python -c "import numpy as np; print(len(np.load('$WORK_DIR/X.npy')))")"
CHUNK_SIZE=$(( (N_ROWS + K - 1) / K ))   # ceiling division -> chunks tile [0, N_ROWS)
echo "[array-dryrun] N_ROWS=$N_ROWS  K=$K  chunk_size=$CHUNK_SIZE  work_dir=$WORK_DIR"

# ── Step 2: submit the evaluation array (K tasks, one full node each) ─────────
ARRAY_JOB_ID="$(sbatch --parsable \
  --job-name="$NAME" \
  --partition=rome --nodes=1 --ntasks=1 --gpus=0 --cpus-per-task=128 \
  --time="$TASK_TIME" \
  --array="0-$(( K - 1 ))" \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/${NAME}-%A_%a.out \
  --error=jobs/logs/${NAME}-%A_%a.err \
  --export=ALL,METHOD="$METHOD",WORK_DIR="$WORK_DIR",N_ROWS="$N_ROWS",CHUNK_SIZE="$CHUNK_SIZE",REPLICATES="$REPLICATES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",ROAD_LENGTH="$ROAD_LENGTH",DT="$DT",GROUP_RADIUS="$GROUP_RADIUS" \
  "$SCRIPT_DIR/_gsa-array-eval.sh")"
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"  # --parsable may append ";cluster"

# ── Step 3: submit the merge, gated on every array task succeeding ────────────
OUT_DIR="${GSA_OUT_BASE}/${ARRAY_JOB_ID}-${GIT_HASH}-${METHOD}"
MERGE_JOB_ID="$(sbatch --parsable \
  --job-name="${NAME}-merge" \
  --partition=rome --nodes=1 --ntasks=1 --gpus=0 --cpus-per-task=1 \
  --time=00:30:00 \
  --dependency="afterok:${ARRAY_JOB_ID}" \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/${NAME}-merge-${ARRAY_JOB_ID}.out \
  --error=jobs/logs/${NAME}-merge-${ARRAY_JOB_ID}.err \
  --export=ALL,METHOD="$METHOD",WORK_DIR="$WORK_DIR",OUT_DIR="$OUT_DIR" \
  "$SCRIPT_DIR/_gsa-array-merge.sh")"
MERGE_JOB_ID="${MERGE_JOB_ID%%;*}"

MERGE_OUT_FILE="$PROJECT_ROOT/jobs/logs/${NAME}-merge-${ARRAY_JOB_ID}.out"
MERGE_ERR_FILE="$PROJECT_ROOT/jobs/logs/${NAME}-merge-${ARRAY_JOB_ID}.err"

echo ""
echo "[summary] $METHOD GSA DRY RUN split into $K chunks of ~$CHUNK_SIZE rows ($N_ROWS total)"
echo "  eval array : $ARRAY_JOB_ID  (tasks 0-$(( K - 1 )), 128 cores each)"
echo "  merge job  : $MERGE_JOB_ID  (runs afterok the whole array)"
echo "  result     : $OUT_DIR/gsa_${METHOD}.csv"
echo "  work dir   : $WORK_DIR  (X.npy + Y chunks; safe to delete after merge)"
echo "  task logs  : $PROJECT_ROOT/jobs/logs/${NAME}-${ARRAY_JOB_ID}_*.out"
echo ""

# Watch the pipeline (eval array -> merge) and exit when it ends.
exec "$SCRIPT_DIR/_gsa-array-watch.sh" \
  "$ARRAY_JOB_ID" "$MERGE_JOB_ID" "$METHOD" "$MERGE_OUT_FILE" "$MERGE_ERR_FILE" "$OUT_DIR"
