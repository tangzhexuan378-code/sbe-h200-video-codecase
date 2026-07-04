#!/usr/bin/env bash
set -euo pipefail

source "${VENV_DIR:-.venv}/bin/activate"

CONFIG="${CONFIG:-configs/h200_4h_online_continuous.yaml}"
RUN_NAME="${RUN_NAME:-h200_online_continuous_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/$RUN_NAME}"
REQUIRE_OFFICIAL_VBENCH="${REQUIRE_OFFICIAL_VBENCH:-1}"

mkdir -p "$OUT_DIR"
python scripts/check_env.py | tee "$OUT_DIR/env_status.json"

echo "[h200_4h] Generation + proxy metrics"
python -m sbe_h200_video.run_experiment \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR" \
  --allow-vbench-proxy \
  2>&1 | tee "$OUT_DIR/run.log"

echo "[h200_4h] Official VBench-4D custom-input evaluation"
if bash scripts/run_official_vbench4.sh "$OUT_DIR" 2>&1 | tee "$OUT_DIR/vbench4.log"; then
  echo "[h200_4h] Rebuild report with official VBench-4D scores"
  python -m sbe_h200_video.rebuild_report \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --official-vbench-json "$OUT_DIR/official_vbench4_scores.json" \
    2>&1 | tee "$OUT_DIR/rebuild_report.log"
else
  if [ "$REQUIRE_OFFICIAL_VBENCH" = "1" ]; then
    echo "[h200_4h] ERROR: Official VBench-4D failed and REQUIRE_OFFICIAL_VBENCH=1."
    echo "[h200_4h] No official VBench claim should be made from this run."
    exit 4
  fi
  echo "[h200_4h] WARNING: Official VBench failed. Keeping proxy-marked table only."
fi

echo "[h200_4h] Done"
echo "  $OUT_DIR/main_table.csv"
echo "  $OUT_DIR/main_table.md"
echo "  $OUT_DIR/official_vbench4_scores.json"
echo "  $OUT_DIR/per_video_rows.csv"
echo "  $OUT_DIR/generation_rows.csv"
echo "  $OUT_DIR/report.tex"
echo "  $OUT_DIR/report.pdf if pdflatex is installed"
