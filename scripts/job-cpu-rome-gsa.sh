#!/usr/bin/env bash
# Run a global sensitivity analysis on one Snellius Rome CPU node.
#
# Pure-Python/Mesa model (no GPU): the `rome` partition, with the GSA driver
# fanning samples across the node's 128 cores.
#
# Usage (from the project root, on a login node):
#   bash scripts/job-cpu-rome-gsa.sh [METHOD] [SAMPLES] [REPLICATES] [GENERATIONS] \
#                                    [MAX_STEPS] [ROAD_LENGTH] [DT] [GROUP_RADIUS]
# Defaults: METHOD=both, SAMPLES=512, REPLICATES=5, GENERATIONS=30, MAX_STEPS=500,
#           ROAD_LENGTH=100000 (100 km), DT=60 (1 min), GROUP_RADIUS=200.
# The scenario defaults are the validated 100 km / dt=60 s run (riders finish in
# ~280 steps at pack pace; 500 gives margin). Each sample runs the evolution loop
# (GENERATIONS races) so utility_decay bites. Indices land in data/gsa_<method>.csv;
# logs in jobs/logs/.

set -euo pipefail

METHOD="${1:-both}"
SAMPLES="${2:-512}"
REPLICATES="${3:-5}"
GENERATIONS="${4:-30}"
MAX_STEPS="${5:-500}"
ROAD_LENGTH="${6:-100000}"
DT="${7:-60}"
GROUP_RADIUS="${8:-200}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

sbatch --job-name=peloton-gsa \
  --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=128 \
  --time=12:00:00 \
  --mail-user=lorenzo.tecchia@student.uva.nl \
  --mail-type=END,FAIL \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-gsa-%j.out \
  --error=jobs/logs/peloton-gsa-%j.err \
  --export=ALL,METHOD="$METHOD",SAMPLES="$SAMPLES",REPLICATES="$REPLICATES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",ROAD_LENGTH="$ROAD_LENGTH",DT="$DT",GROUP_RADIUS="$GROUP_RADIUS" \
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
EOF
