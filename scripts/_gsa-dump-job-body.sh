#!/usr/bin/env bash
# Shared sbatch body for the GSA *agent-dump* jobs. Not submitted directly: the
# job-cpu-rome-gsa-{morris,sobol}-dump.sh wrappers pass it to sbatch with METHOD
# and the SAMPLES/GENERATIONS/... knobs exported into the environment.
#
# Unlike _gsa-job-body.sh this computes NO sensitivity indices: it dumps the full
# per-step per-agent state for every generation of every sample (main.py-dump
# style) so the SA can be recomputed from the raw data later. No replication.
set -euo pipefail

# Same module/uv setup as scripts/snellius-py-runtime-setup.sh.
if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi
module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
export PATH="$HOME/.local/bin:$PATH"

# Per-run output dir: <base>/<SLURM JOB ID>-<GIT HASH>-<METHOD>-dump. Override the
# base with GSA_OUT_BASE; the git hash is the short HEAD of the submit checkout
# (cwd is PROJECT_ROOT via sbatch --chdir), "nogit" if not a repo. SLURM_JOB_ID is
# "local" off-cluster so the script still runs for a quick local test.
GSA_OUT_BASE="${GSA_OUT_BASE:-/gpfs/work5/0/prjs2142/gsa-agent-dump-per-run}"
GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
OUT_DIR="${GSA_OUT_BASE}/${SLURM_JOB_ID:-local}-${GIT_HASH}-${METHOD}-dump"
mkdir -p "$OUT_DIR"

# PARQUET=1 -> write Parquet instead of CSV (smaller for the big per-step tables).
PARQUET_FLAG=()
[[ "${PARQUET:-0}" == "1" ]] && PARQUET_FLAG=(--parquet)

echo "[job] method=$METHOD samples=$SAMPLES generations=$GENERATIONS max_steps=$MAX_STEPS road_length=$ROAD_LENGTH dt=$DT group_radius=$GROUP_RADIUS procs=$SLURM_CPUS_PER_TASK parquet=${PARQUET:-0}"
echo "[job] out_dir=$OUT_DIR"
uv run python -m peloton.gsa_dump \
  --method "$METHOD" \
  --samples "$SAMPLES" \
  --generations "$GENERATIONS" \
  --max-steps "$MAX_STEPS" \
  --road-length "$ROAD_LENGTH" \
  --dt "$DT" \
  --group-radius "$GROUP_RADIUS" \
  --processes "$SLURM_CPUS_PER_TASK" \
  --out-dir "$OUT_DIR" \
  "${PARQUET_FLAG[@]}"
