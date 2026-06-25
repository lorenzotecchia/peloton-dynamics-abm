#!/usr/bin/env bash
# Merge-phase body for the GSA array jobs. Not submitted directly: the
# job-cpu-rome-gsa-{morris,sobol}-array.sh wrappers submit it with a
# `--dependency=afterok:<array>` so it runs only once every chunk succeeded.
# Concatenates the Y_*.npy chunks in WORK_DIR and writes gsa_<METHOD>.csv to OUT_DIR.
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi
module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$OUT_DIR"
# Keep the design matrix (param values per sample row) alongside the indices.
cp "$WORK_DIR/X_index.csv" "$OUT_DIR/sample_index.csv" 2>/dev/null || true

echo "[merge] $METHOD: combining Y chunks from $WORK_DIR -> $OUT_DIR"
uv run python -m peloton.gsa \
  --mode merge --method "$METHOD" \
  --x-file "$WORK_DIR/X.npy" \
  --merge-dir "$WORK_DIR" \
  --out-dir "$OUT_DIR"
echo "[merge] done -> $OUT_DIR/gsa_${METHOD}.csv"
