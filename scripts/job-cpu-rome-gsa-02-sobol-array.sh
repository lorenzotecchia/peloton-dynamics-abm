#!/usr/bin/env bash
# Run the Sobol GSA as a Slurm array: K nodes evaluate X in parallel, then a
# merge job computes the indices.  Wall-clock time ~= single-node time / K.
#
# Flow:
#   login node  : generate X.npy  (fast, runs locally)
#   array job 0 : evaluate rows [0,   CHUNK) → Y_000000_XXXXXX.npy
#   array job 1 : evaluate rows [CHUNK,2*CHUNK) → Y_XXXXXX_YYYYYY.npy
#   ...
#   merge job   : concatenate Y chunks → gsa_sobol.csv  (dependency: afterok array)
#
# Dump layout under DUMP_BASE:
#   <ARRAY_JOB_ID>-<GIT_HASH>-sobol/
#     meta.json
#     sample_index.csv
#     s<NNNNN>_r<RR>/   (agent dumps written by each array task)
#
# Usage (from project root, on a login node):
#   bash scripts/job-cpu-rome-gsa-sobol-array.sh [SAMPLES] [REPLICATES] [GENERATIONS] [MAX_STEPS] [K] [DUMP_BASE]
# Defaults: SAMPLES=512, REPLICATES=5, GENERATIONS=30, MAX_STEPS=2000, K=6,
#           DUMP_BASE=/home/tho/prjs2142/gsa-agent-dump-per-run

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run with 'bash', not 'sbatch':" >&2
    echo "  bash scripts/job-cpu-rome-gsa-sobol-array.sh [SAMPLES] ..." >&2
    exit 1
fi

SAMPLES="${1:-512}"
REPLICATES="${2:-5}"
GENERATIONS="${3:-30}"
MAX_STEPS="${4:-2000}"
K="${5:-6}"                                          # number of array chunks (nodes)
DUMP_BASE="${6:-/home/tho/prjs2142/gsa-agent-dump-per-run}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

GIT_HASH=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)

# Sobol design matrix size: N * (D+2), D=4 (matches PROBLEM in gsa.py)
D=4
N_ROWS=$(( SAMPLES * (D + 2) ))
CHUNK_SIZE=$(( (N_ROWS + K - 1) / K ))   # ceiling division

# Shared work directory for X.npy and Y chunk files (must be on shared FS)
WORK_DIR="$PROJECT_ROOT/jobs/sobol-work/$(date +%Y%m%d_%H%M%S)-${GIT_HASH}"
mkdir -p "$WORK_DIR"

echo "[array] samples=$SAMPLES replicates=$REPLICATES generations=$GENERATIONS max_steps=$MAX_STEPS"
echo "[array] K=$K chunks, N_ROWS=$N_ROWS, chunk_size=$CHUNK_SIZE"
echo "[array] work_dir=$WORK_DIR"

# ── Step 1: generate X on the login node (seconds) ───────────────────────────
uv run python -m peloton.gsa \
    --mode    sample \
    --samples "$SAMPLES" \
    --x-file  "$WORK_DIR/X.npy"

# ── Step 2: submit the evaluation array job ───────────────────────────────────
ARRAY_OUT=$(sbatch \
    --job-name=peloton-gsa-sobol \
    --partition=rome \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=128 \
    --time=04:00:00 \
    --array="0-$(( K - 1 ))" \
    --output="$PROJECT_ROOT/jobs/logs/peloton-gsa-sobol-%A_%a.out" \
    --error="$PROJECT_ROOT/jobs/logs/peloton-gsa-sobol-%A_%a.err" \
    --export=ALL,SAMPLES="$SAMPLES",REPLICATES="$REPLICATES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",N_ROWS="$N_ROWS",CHUNK_SIZE="$CHUNK_SIZE",WORK_DIR="$WORK_DIR",DUMP_BASE="$DUMP_BASE",GIT_HASH="$GIT_HASH" \
    <<'ARRAY_EOF'
#!/usr/bin/env bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi
module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
export PATH="$HOME/.local/bin:$PATH"

ROW_START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
ROW_END=$(( ROW_START + CHUNK_SIZE ))
[[ $ROW_END -gt $N_ROWS ]] && ROW_END=$N_ROWS

DUMP_DIR="${DUMP_BASE}/${SLURM_ARRAY_JOB_ID}-${GIT_HASH}-sobol"
Y_OUT="${WORK_DIR}/Y_$(printf '%06d' $ROW_START)_$(printf '%06d' $ROW_END).npy"

echo "[task $SLURM_ARRAY_TASK_ID] rows $ROW_START..$ROW_END → $Y_OUT"

uv run python -m peloton.gsa \
    --mode       evaluate \
    --x-file     "$WORK_DIR/X.npy" \
    --row-start  "$ROW_START" \
    --row-end    "$ROW_END" \
    --y-out      "$Y_OUT" \
    --replicates "$REPLICATES" \
    --generations "$GENERATIONS" \
    --max-steps  "$MAX_STEPS" \
    --processes  "$SLURM_CPUS_PER_TASK" \
    --dump-dir   "$DUMP_DIR" \
    --parquet
ARRAY_EOF
)

ARRAY_JOB_ID=$(echo "$ARRAY_OUT" | awk '{print $NF}')
echo "$ARRAY_OUT"

# ── Step 3: submit merge job, runs after all array tasks succeed ──────────────
MERGE_OUT=$(sbatch \
    --job-name=peloton-gsa-merge \
    --partition=rome \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=1 \
    --time=00:30:00 \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --output="$PROJECT_ROOT/jobs/logs/peloton-gsa-merge-${ARRAY_JOB_ID}.out" \
    --error="$PROJECT_ROOT/jobs/logs/peloton-gsa-merge-${ARRAY_JOB_ID}.err" \
    --export=ALL,WORK_DIR="$WORK_DIR",ARRAY_JOB_ID="$ARRAY_JOB_ID",GIT_HASH="$GIT_HASH",DUMP_BASE="$DUMP_BASE",SAMPLES="$SAMPLES" \
    <<'MERGE_EOF'
#!/usr/bin/env bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi
module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
export PATH="$HOME/.local/bin:$PATH"

DUMP_DIR="${DUMP_BASE}/${ARRAY_JOB_ID}-${GIT_HASH}-sobol"

# Copy sample_index.csv (generated alongside X.npy) into the dump dir
cp "${WORK_DIR}/X_index.csv" "${DUMP_DIR}/sample_index.csv" 2>/dev/null || true

echo "[merge] combining Y chunks from $WORK_DIR"
uv run python -m peloton.gsa \
    --mode      merge \
    --x-file    "$WORK_DIR/X.npy" \
    --merge-dir "$WORK_DIR" \
    --samples   "$SAMPLES" \
    --out-dir   data

echo "[merge] done — gsa_sobol.csv written"
MERGE_EOF
)

MERGE_JOB_ID=$(echo "$MERGE_OUT" | awk '{print $NF}')
echo "$MERGE_OUT"

# ── Summary ───────────────────────────────────────────────────────────────────
DUMP_DIR="${DUMP_BASE}/${ARRAY_JOB_ID}-${GIT_HASH}-sobol"
echo ""
echo "[summary]"
echo "  array job : $ARRAY_JOB_ID  (tasks 0-$(( K-1 )), each on 128 cores)"
echo "  merge job : $MERGE_JOB_ID  (runs after all tasks succeed)"
echo "  dump dir  : $DUMP_DIR"
echo "  work dir  : $WORK_DIR"
echo ""
echo "[tail] to follow array task logs:"
for i in $(seq 0 $(( K - 1 ))); do
    echo "  tail -f $PROJECT_ROOT/jobs/logs/peloton-gsa-sobol-${ARRAY_JOB_ID}_${i}.out"
done
echo "[tail] to follow merge log:"
echo "  tail -f $PROJECT_ROOT/jobs/logs/peloton-gsa-merge-${ARRAY_JOB_ID}.out"
