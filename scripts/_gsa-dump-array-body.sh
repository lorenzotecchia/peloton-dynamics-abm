#!/usr/bin/env bash
# Array-task body for the Sobol agent-DUMP array. Not submitted directly: the
# job-cpu-rome-gsa-sobol-dump-array.sh wrapper passes it to `sbatch --array` with
# METHOD/WORK_DIR/CHUNK_SIZE/N_ROWS/... exported. Each task dumps one contiguous
# row chunk of WORK_DIR/X.npy. There is NO merge: every sample_<i>/ dir is
# self-contained, so all tasks just write their chunks into one shared out dir
# (keyed by the array job id, which is identical across the array).
set -euo pipefail

# Same module/uv setup as scripts/snellius-py-runtime-setup.sh.
if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi
module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
export PATH="$HOME/.local/bin:$PATH"

ROW_START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
ROW_END=$(( ROW_START + CHUNK_SIZE ))
(( ROW_END > N_ROWS )) && ROW_END=$N_ROWS

# A trailing task can be empty when K does not divide N_ROWS; nothing to do.
if (( ROW_START >= ROW_END )); then
    echo "[task $SLURM_ARRAY_TASK_ID] empty chunk ($ROW_START..$ROW_END >= $N_ROWS), skipping"
    exit 0
fi

# All tasks share one out dir, keyed by the array job id so their chunks land
# beside each other (SLURM_ARRAY_JOB_ID is identical across the array). The git
# hash is the short HEAD of the submit checkout (cwd is PROJECT_ROOT via --chdir).
GSA_OUT_BASE="${GSA_OUT_BASE:-/gpfs/work5/0/prjs2142/gsa-agent-dump-per-run}"
GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
OUT_DIR="${GSA_OUT_BASE}/${SLURM_ARRAY_JOB_ID:-local}-${GIT_HASH}-${METHOD}-dump"
mkdir -p "$OUT_DIR"

# PARQUET=1 (default) -> Parquet instead of CSV (smaller); set PARQUET=0 for CSV.
PARQUET_FLAG=()
[[ "${PARQUET:-1}" == "1" ]] && PARQUET_FLAG=(--parquet)

echo "[task $SLURM_ARRAY_TASK_ID] $METHOD dump rows $ROW_START..$ROW_END -> $OUT_DIR (procs=$SLURM_CPUS_PER_TASK, parquet=${PARQUET:-1})"
uv run python -m peloton.gsa_dump \
  --mode dump --method "$METHOD" \
  --x-file "$WORK_DIR/X.npy" \
  --row-start "$ROW_START" --row-end "$ROW_END" \
  --samples "$SAMPLES" \
  --generations "$GENERATIONS" \
  --max-steps "$MAX_STEPS" \
  --road-length "$ROAD_LENGTH" \
  --dt "$DT" \
  --group-radius "$GROUP_RADIUS" \
  --processes "$SLURM_CPUS_PER_TASK" \
  --out-dir "$OUT_DIR" \
  "${PARQUET_FLAG[@]}"
