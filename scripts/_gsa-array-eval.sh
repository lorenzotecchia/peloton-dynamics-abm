#!/usr/bin/env bash
# Array-task body for the GSA evaluate phase. Not submitted directly: the
# job-cpu-rome-gsa-{morris,sobol}-array.sh wrappers pass it to `sbatch --array`
# with METHOD/WORK_DIR/CHUNK_SIZE/N_ROWS/... exported. Each task evaluates one
# contiguous row chunk of X.npy and writes Y_<start>_<end>.npy into WORK_DIR.
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

Y_OUT="${WORK_DIR}/Y_$(printf '%06d' "$ROW_START")_$(printf '%06d' "$ROW_END").npy"
echo "[task $SLURM_ARRAY_TASK_ID] $METHOD rows $ROW_START..$ROW_END -> $Y_OUT (procs=$SLURM_CPUS_PER_TASK)"

uv run python -m peloton.gsa \
  --mode evaluate --method "$METHOD" \
  --x-file "$WORK_DIR/X.npy" \
  --row-start "$ROW_START" --row-end "$ROW_END" \
  --y-out "$Y_OUT" \
  --replicates "$REPLICATES" \
  --generations "$GENERATIONS" \
  --max-steps "$MAX_STEPS" \
  --road-length "$ROAD_LENGTH" \
  --dt "$DT" \
  --group-radius "$GROUP_RADIUS" \
  --processes "$SLURM_CPUS_PER_TASK"
