#!/usr/bin/env bash
# Submit a batch-learning run on one Snellius Rome CPU node.
#
# Defaults match your request:
#   - 100 replications
#   - 100 generations
# Model parameters come from src/peloton/config.py via main.py -> PelotonConfig.
#
# Usage (from project root, on a login node):
#   bash scripts/job-cpu-rome-batch-learning.sh [REPLICATIONS] [GENERATIONS] [MAX_STEPS]
# Defaults:
#   REPLICATIONS=100, GENERATIONS=100, MAX_STEPS=400

set -euo pipefail

REPLICATIONS="${1:-100}"
GENERATIONS="${2:-100}"
MAX_STEPS="${3:-400}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

sbatch --job-name=peloton-batch-learn \
  --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=32 \
  --time=12:00:00 \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-batch-learn-%j.out \
  --error=jobs/logs/peloton-batch-learn-%j.err \
  --export=ALL,REPLICATIONS="$REPLICATIONS",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS" \
  <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi
module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
export PATH="$HOME/.local/bin:$PATH"

OUT_DIR="data/batch_learning/job-${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "[job] replications=$REPLICATIONS generations=$GENERATIONS max_steps=$MAX_STEPS"
echo "[job] output_dir=$OUT_DIR"
echo "[job] model parameters from src/peloton/config.py (PelotonConfig defaults)"

uv run python batch_learning.py \
  --replications "$REPLICATIONS" \
  --generations "$GENERATIONS" \
  --max-steps "$MAX_STEPS" \
  --output-dir "$OUT_DIR"
EOF

