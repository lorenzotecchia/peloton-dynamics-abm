#!/usr/bin/env bash
# Submit Morris and Sobol GSA as two independent Slurm jobs so they run in
# parallel on separate Rome nodes, halving wall-clock time for method=both.
#
# Each job writes its agent dumps under:
#   <DUMP_BASE>/<SLURM_JOB_ID>-<GIT_SHORT_HASH>-morris/
#   <DUMP_BASE>/<SLURM_JOB_ID>-<GIT_SHORT_HASH>-sobol/
#
# Usage (from the project root, on a login node):
#   bash scripts/job-cpu-rome-gsa-separate-jobs.sh [SAMPLES] [REPLICATES] [GENERATIONS] [MAX_STEPS] [DUMP_BASE]
# Defaults: SAMPLES=512, REPLICATES=5, GENERATIONS=30, MAX_STEPS=2000,
#           DUMP_BASE=/home/tho/prjs2142/gsa-agent-dump-per-run

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run this script with 'bash', not 'sbatch':" >&2
    echo "  bash scripts/job-cpu-rome-gsa-separate-jobs.sh [SAMPLES] ..." >&2
    exit 1
fi

SAMPLES="${1:-512}"
REPLICATES="${2:-5}"
GENERATIONS="${3:-30}"
MAX_STEPS="${4:-2000}"
DUMP_BASE="${5:-/home/tho/prjs2142/gsa-agent-dump-per-run}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

T_START=$SECONDS

# NO_TAIL=1 suppresses tail -f in the child script; both jobs submit and return immediately.
export NO_TAIL=1

bash "$SCRIPT_DIR/job-cpu-rome-gsa.sh" morris "$SAMPLES" "$REPLICATES" "$GENERATIONS" "$MAX_STEPS" "$DUMP_BASE"
bash "$SCRIPT_DIR/job-cpu-rome-gsa.sh" sobol  "$SAMPLES" "$REPLICATES" "$GENERATIONS" "$MAX_STEPS" "$DUMP_BASE"

# Collect the two job IDs from the most recent entries in jobs/logs/.
MORRIS_JOB=$(ls -t "$SCRIPT_DIR/../jobs/logs"/peloton-gsa-*.out 2>/dev/null | sed -n '2p' | grep -oP '(?<=peloton-gsa-)\d+')
SOBOL_JOB=$(ls  -t "$SCRIPT_DIR/../jobs/logs"/peloton-gsa-*.out 2>/dev/null | sed -n '1p' | grep -oP '(?<=peloton-gsa-)\d+')
LOGS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/jobs/logs"
echo "[poll] waiting for jobs ${MORRIS_JOB} (morris) and ${SOBOL_JOB} (sobol) to finish ..."
echo "[tail] to follow logs manually:"
echo "  tail -f ${LOGS_DIR}/peloton-gsa-${MORRIS_JOB}.out  # morris"
echo "  tail -f ${LOGS_DIR}/peloton-gsa-${SOBOL_JOB}.out   # sobol"

# Poll squeue until neither job appears anymore.
while squeue --noheader -j "${MORRIS_JOB},${SOBOL_JOB}" 2>/dev/null | grep -q .; do
    sleep 30
done

ELAPSED=$(( SECONDS - T_START ))
printf "[done] total wall-clock time: %dh %02dm %02ds\n" \
    $(( ELAPSED / 3600 )) $(( (ELAPSED % 3600) / 60 )) $(( ELAPSED % 60 ))
