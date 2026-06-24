#!/usr/bin/env bash
# Submit the Morris GSA job on Snellius (Rome CPU partition).
#
# Output directory: /home/tho/prjs2142/gsa-agent-dump-per-run/<JOB-ID>-<GIT-HASH>-morris
# After submission the script tails job stdout in real-time until the job ends.
#
# Usage (from project root, on a login node):
#   bash scripts/job-gsa-morris-sbatch.sh [TRAJECTORIES] [MAX_STEPS]
# Defaults: TRAJECTORIES=20, MAX_STEPS=2000

set -euo pipefail

TRAJECTORIES="${1:-20}"
GENERATIONS="${2:-200}"
MAX_STEPS="${3:-10000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GIT_HASH="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)"
BASE_OUTDIR="/home/tho/prjs2142/gsa-agent-dump-per-run"
mkdir -p "$PROJECT_ROOT/jobs/logs"

echo "[submit] Morris GSA: trajectories=$TRAJECTORIES generations=$GENERATIONS max_steps=$MAX_STEPS git=$GIT_HASH"

JOB_ID=$(sbatch \
  --job-name=peloton-gsa-morris \
  --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=128 \
  --time=08:00:00 \
  --chdir="$PROJECT_ROOT" \
  --output="jobs/logs/peloton-gsa-morris-%j.out" \
  --error="jobs/logs/peloton-gsa-morris-%j.err" \
  --export=ALL,TRAJECTORIES="$TRAJECTORIES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",GIT_HASH="$GIT_HASH",BASE_OUTDIR="$BASE_OUTDIR" \
  --parsable \
  <<'BATCHEOF'
#!/usr/bin/env bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi
module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
export PATH="$HOME/.local/bin:$PATH"

OUT_DIR="${BASE_OUTDIR}/${SLURM_JOB_ID}-${GIT_HASH}-morris"
mkdir -p "$OUT_DIR"

echo "[job] Morris GSA starting"
echo "[job]   trajectories : $TRAJECTORIES"
echo "[job]   generations  : $GENERATIONS"
echo "[job]   max_steps    : $MAX_STEPS"
echo "[job]   cpus         : $SLURM_CPUS_PER_TASK"
echo "[job]   output dir   : $OUT_DIR"

uv run python scripts/gsa_morris.py \
  --trajectories "$TRAJECTORIES" \
  --generations  "$GENERATIONS" \
  --max-steps    "$MAX_STEPS" \
  --processes    "$SLURM_CPUS_PER_TASK" \
  --out-dir      "$OUT_DIR"

echo "[job] Morris GSA complete. Results: $OUT_DIR"
BATCHEOF
)

# sbatch --parsable may return "jobid;cluster" — keep only the job ID
JOB_ID="${JOB_ID%%;*}"
echo "[submit] job ID: $JOB_ID"

LOG="$PROJECT_ROOT/jobs/logs/peloton-gsa-morris-${JOB_ID}.out"
echo "[submit] log  : $LOG"
echo "[submit] tracking stdout (Ctrl+C to detach; job continues on Snellius) ..."

# Wait for the log file to appear (job may be queued briefly)
until [[ -f "$LOG" ]]; do sleep 3; done

tail -F "$LOG" &
TAIL_PID=$!

# Poll squeue until the job is no longer listed
while squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; do
    sleep 15
done

sleep 3
kill "$TAIL_PID" 2>/dev/null || true
echo "[submit] job $JOB_ID finished."
