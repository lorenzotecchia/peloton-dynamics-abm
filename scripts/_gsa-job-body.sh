#!/usr/bin/env bash
# Shared sbatch body for the GSA jobs. Not submitted directly: the
# job-cpu-rome-gsa-{morris,sobol}.sh wrappers pass it to sbatch with METHOD and
# the SAMPLES/REPLICATES/... knobs exported into the environment.
set -euo pipefail

# Same module/uv setup as scripts/snellius-py-runtime-setup.sh.
if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi
module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
export PATH="$HOME/.local/bin:$PATH"

echo "[job] method=$METHOD samples=$SAMPLES replicates=$REPLICATES generations=$GENERATIONS max_steps=$MAX_STEPS road_length=$ROAD_LENGTH dt=$DT group_radius=$GROUP_RADIUS procs=$SLURM_CPUS_PER_TASK"
uv run python -m peloton.gsa \
  --method "$METHOD" \
  --samples "$SAMPLES" \
  --replicates "$REPLICATES" \
  --generations "$GENERATIONS" \
  --max-steps "$MAX_STEPS" \
  --road-length "$ROAD_LENGTH" \
  --dt "$DT" \
  --group-radius "$GROUP_RADIUS" \
  --processes "$SLURM_CPUS_PER_TASK" \
  --out-dir data
