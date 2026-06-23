#!/usr/bin/env bash
# Submit a parallel peloton batch run on one Snellius Rome CPU node.
#
# The peloton model is pure-Python/Mesa (no GPU), so we use the `rome` CPU
# partition and let mesa.batch_run fan the replicates across the node's cores.
#
# Usage (from the project root, on a login node):
#   bash scripts/job-cpu-rome-sbatch.sh [RUNS] [MAX_STEPS]
# Defaults: RUNS=256, MAX_STEPS=1000. Results + logs land in jobs/logs/.

set -euo pipefail

RUNS="${1:-256}"
MAX_STEPS="${2:-1000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

sbatch --job-name=peloton-sweep \
  --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=128 \
  --time=12:00:00 \
  --mail-user=lorenzo.tecchia@student.uva.nl \
  --mail-type=END,FAIL \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-sweep-%j.out \
  --error=jobs/logs/peloton-sweep-%j.err \
  --export=ALL,RUNS="$RUNS",MAX_STEPS="$MAX_STEPS" \
  <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Same module/uv setup as scripts/snellius-py-runtime-setup.sh.
if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi
module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
export PATH="$HOME/.local/bin:$PATH"

OUT="jobs/logs/results-${SLURM_JOB_ID}.csv"
echo "[job] $RUNS runs, max_steps=$MAX_STEPS, $SLURM_CPUS_PER_TASK procs -> $OUT"
uv run python -m peloton.sweep \
  --runs "$RUNS" \
  --max-steps "$MAX_STEPS" \
  --processes "$SLURM_CPUS_PER_TASK" \
  --out "$OUT"
EOF
