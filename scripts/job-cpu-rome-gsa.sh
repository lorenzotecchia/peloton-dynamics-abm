#!/usr/bin/env bash
# Run a global sensitivity analysis on one Snellius Rome CPU node.
#
# Pure-Python/Mesa model (no GPU): the `rome` partition, with the GSA driver
# fanning samples across the node's 128 cores.
#
# Per-run agent state dumps land under DUMP_BASE, organised as:
#
#   <DUMP_BASE>/
#   └── <SLURM_JOB_ID>/
#       ├── meta.json                         job-level: method, N, G, R, steps, PROBLEM bounds, METRICS
#       ├── morris/
#       │   ├── sample_index.csv              design matrix: sample_idx → param values (one row per sample)
#       │   ├── s00000_r00/                   sample 0, replicate 0 (seed=0)
#       │   │   ├── config.json               full PelotonConfig including sampled parameter values
#       │   │   ├── agent_timeseries.parquet  per-step per-agent state for the final-generation race
#       │   │   ├── model_timeseries.parquet  model-level reporters per step
#       │   │   ├── agent_meta.csv            static physiology + strategy coeffs + finish outcome
#       │   │   ├── finish_order.csv          rank, rider uid, finish step/time
#       │   │   └── manifest.json             row counts, file descriptions, analysis hints
#       │   ├── s00000_r01/  ...
#       │   └── s<NNNNN>_r<RR>/  ...
#       └── sobol/
#           ├── sample_index.csv
#           └── s<NNNNN>_r<RR>/  ...
#
# To locate runs by parameter value: read sample_index.csv, filter the rows you
# want, then navigate to s{sample_idx:05d}_r{replicate:02d}/ for those samples.
# The agent dump reflects the evolved population from the final generation (not a
# fresh population), so strategy coefficients represent learned behaviour.
#
# Usage (from the project root, on a login node):
#   bash scripts/job-cpu-rome-gsa.sh [METHOD] [SAMPLES] [REPLICATES] [GENERATIONS] [MAX_STEPS] [DUMP_BASE]
# Defaults: METHOD=both, SAMPLES=512, REPLICATES=5, GENERATIONS=30, MAX_STEPS=2000,
#           DUMP_BASE=/projects/prjs2142/gsa-agent-dump-per-run
# Indices land in data/gsa_<method>.csv; logs in jobs/logs/.

set -euo pipefail

METHOD="${1:-both}"
SAMPLES="${2:-512}"
REPLICATES="${3:-5}"
GENERATIONS="${4:-30}"
MAX_STEPS="${5:-2000}"
DUMP_BASE="${6:-/projects/prjs2142/gsa-agent-dump-per-run}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_ROOT/jobs/logs"

sbatch --job-name=peloton-gsa \
  --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=128 \
  --time=04:00:00 \
  --chdir="$PROJECT_ROOT" \
  --output=jobs/logs/peloton-gsa-%j.out \
  --error=jobs/logs/peloton-gsa-%j.err \
  --export=ALL,METHOD="$METHOD",SAMPLES="$SAMPLES",REPLICATES="$REPLICATES",GENERATIONS="$GENERATIONS",MAX_STEPS="$MAX_STEPS",DUMP_BASE="$DUMP_BASE" \
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

echo "[job] method=$METHOD samples=$SAMPLES replicates=$REPLICATES generations=$GENERATIONS max_steps=$MAX_STEPS procs=$SLURM_CPUS_PER_TASK dump_base=$DUMP_BASE"
uv run python -m peloton.gsa \
  --method      "$METHOD" \
  --samples     "$SAMPLES" \
  --replicates  "$REPLICATES" \
  --generations "$GENERATIONS" \
  --max-steps   "$MAX_STEPS" \
  --processes   "$SLURM_CPUS_PER_TASK" \
  --out-dir     data \
  --dump-dir    "$DUMP_BASE" \
  --parquet
EOF
