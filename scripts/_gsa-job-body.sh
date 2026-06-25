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

# Per-run output dir: <base>/<SLURM JOB ID>-<GIT HASH>-<METHOD>. Override the base
# with GSA_OUT_BASE; the git hash is the short HEAD of the submit checkout (cwd is
# PROJECT_ROOT via sbatch --chdir), "nogit" if not a repo. SLURM_JOB_ID is "local"
# off-cluster so the script still runs for a quick local test.
GSA_OUT_BASE="${GSA_OUT_BASE:-/gpfs/work5/0/prjs2142/gsa-agent-dump-per-run}"
GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
OUT_DIR="${GSA_OUT_BASE}/${SLURM_JOB_ID:-local}-${GIT_HASH}-${METHOD}"
mkdir -p "$OUT_DIR"

echo "[job] method=$METHOD samples=$SAMPLES replicates=$REPLICATES generations=$GENERATIONS max_steps=$MAX_STEPS road_length=$ROAD_LENGTH dt=$DT group_radius=$GROUP_RADIUS procs=$SLURM_CPUS_PER_TASK"
echo "[job] out_dir=$OUT_DIR"
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
  --out-dir "$OUT_DIR"
