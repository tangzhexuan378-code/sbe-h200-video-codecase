#!/usr/bin/env bash
set -euo pipefail

source "${VENV_DIR:-.venv}/bin/activate"
OUT_DIR="${OUT_DIR:-outputs/smoke_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
python -m sbe_h200_video.run_experiment \
  --config configs/smoke.yaml \
  --out-dir "$OUT_DIR" \
  --allow-vbench-proxy \
  2>&1 | tee "$OUT_DIR/run.log"
echo "[smoke] $OUT_DIR/main_table.md"
