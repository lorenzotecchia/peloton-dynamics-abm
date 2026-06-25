#!/usr/bin/env bash
# Watch a GSA array pipeline (eval array + its afterok merge) and exit when it
# ends, reporting the final Slurm states. Shared by the array wrappers; not run
# directly. The eval phase is summarised periodically (K task logs are too noisy
# to interleave); the merge log is streamed live once the array completes.
#
# Args: ARRAY_JOB_ID MERGE_JOB_ID METHOD MERGE_OUT_FILE MERGE_ERR_FILE OUT_DIR
# Note: no `set -e` -- we want to observe and report failures, not abort on them.
set -uo pipefail

ARRAY_JOB_ID="$1"; MERGE_JOB_ID="$2"; METHOD="$3"
MERGE_OUT="$4"; MERGE_ERR="$5"; OUT_DIR="$6"

job_active () { [[ -n "$(squeue -h -j "$1" -o '%T' 2>/dev/null)" ]]; }

# ── Phase 1: wait for the eval array, printing a periodic state summary ──────
echo "Watching eval array $ARRAY_JOB_ID until all tasks finish (Ctrl-C stops"
echo "watching; the jobs keep running on the cluster)..."
while job_active "$ARRAY_JOB_ID"; do
    states="$(squeue -h -j "$ARRAY_JOB_ID" -o '%T' 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
    echo "  [$(date +%H:%M:%S)] array $ARRAY_JOB_ID: ${states:-<none active>}"
    sleep 30
done

ASTATES="$(sacct -j "$ARRAY_JOB_ID" -n -X -o State 2>/dev/null | tr -d ' ' | sort -u)"
echo "Eval array $ARRAY_JOB_ID finished; task states: $(echo "$ASTATES" | tr '\n' ' ')"

if [[ -z "$ASTATES" ]]; then
    echo "WARN: sacct returned no array state; watching the merge anyway." >&2
elif [[ "$ASTATES" != "COMPLETED" ]]; then
    echo "Array did not all COMPLETE -> the afterok merge will not run." >&2
    scancel "$MERGE_JOB_ID" 2>/dev/null || true   # clear the stuck dependency
    echo "Inspect the array task logs and 'sacct -j $ARRAY_JOB_ID'." >&2
    exit 1
fi

# ── Phase 2: wait for the merge job's log to appear, then stream it ──────────
echo "Waiting for merge $MERGE_JOB_ID to start..."
while job_active "$MERGE_JOB_ID" && [[ ! -e "$MERGE_OUT" ]]; do
    if [[ "$(squeue -h -j "$MERGE_JOB_ID" -o '%r' 2>/dev/null)" == *DependencyNeverSatisfied* ]]; then
        echo "Merge $MERGE_JOB_ID will never run (dependency never satisfied)." >&2
        scancel "$MERGE_JOB_ID" 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

if [[ -e "$MERGE_OUT" ]]; then
    echo "----------------------------------------------------------------------"
    tail -n +1 -F "$MERGE_OUT" &
    TAIL_PID=$!
    while job_active "$MERGE_JOB_ID"; do sleep 10; done
    sleep 3                                  # let tail flush the final lines
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
    echo "----------------------------------------------------------------------"
fi

MSTATE="$(sacct -j "$MERGE_JOB_ID" -n -X -o State 2>/dev/null | head -n1 | tr -d ' ')"
case "$MSTATE" in
    COMPLETED)  echo "Pipeline COMPLETED. Result: $OUT_DIR/gsa_${METHOD}.csv" ;;
    ""|UNKNOWN) echo "Merge $MERGE_JOB_ID ended; sacct gave no state. Check $MERGE_ERR." ;;
    *)          echo "Merge $MERGE_JOB_ID ended in state $MSTATE -- inspect $MERGE_ERR and 'sacct -j $MERGE_JOB_ID'." >&2; exit 1 ;;
esac
