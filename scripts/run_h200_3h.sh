#!/usr/bin/env bash
set -euo pipefail

source "${VENV_DIR:-.venv}/bin/activate"

CONFIG="${CONFIG:-configs/h200_3h.yaml}"
RUN_NAME="${RUN_NAME:-h200_v5_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/$RUN_NAME}"
REQUIRE_OFFICIAL_VBENCH="${REQUIRE_OFFICIAL_VBENCH:-0}"

mkdir -p "$OUT_DIR"
python scripts/check_env.py | tee "$OUT_DIR/env_status.json"

args=(--config "$CONFIG" --out-dir "$OUT_DIR")
if [ "$REQUIRE_OFFICIAL_VBENCH" = "1" ]; then
  args+=(--require-official-vbench)
else
  args+=(--allow-vbench-proxy)
fi

python -m sbe_h200_video.run_experiment "${args[@]}" 2>&1 | tee "$OUT_DIR/run.log"

echo "[run_h200_3h] Results:"
echo "  $OUT_DIR/main_table.csv"
echo "  $OUT_DIR/main_table.md"
echo "  $OUT_DIR/report.tex"
echo "  $OUT_DIR/report.pdf (if pdflatex is installed)"
