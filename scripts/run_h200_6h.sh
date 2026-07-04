#!/usr/bin/env bash
set -euo pipefail

source "${VENV_DIR:-.venv}/bin/activate"

CONFIG="${CONFIG:-configs/h200_6h_online_continuous.yaml}"
RUN_NAME="${RUN_NAME:-h200_online_continuous_6h_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/$RUN_NAME}"
REQUIRE_OFFICIAL_VBENCH="${REQUIRE_OFFICIAL_VBENCH:-1}"

# H200 has enough VRAM for the long-video setting. This allocator setting
# reduces fragmentation in long Wan/VBench runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_DIR"
python scripts/check_env.py | tee "$OUT_DIR/env_status.json"

python - "$OUT_DIR/env_status.json" <<'PY'
import json
import pathlib
import sys

status = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
gpu = str(status.get("gpu_name", ""))
if "H200" not in gpu.upper():
    print(f"[h200_6h] WARNING: GPU name is {gpu!r}, not H200. The run still works, but the 5-6h budget is designed for H200.")
else:
    print(f"[h200_6h] H200 detected: {gpu}")
PY

echo "[h200_6h] Generation + sampled-frame metrics"
python -m sbe_h200_video.run_experiment \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR" \
  --allow-vbench-proxy \
  2>&1 | tee "$OUT_DIR/run.log"

echo "[h200_6h] Official VBench-4D custom-input evaluation"
if bash scripts/run_official_vbench4.sh "$OUT_DIR" 2>&1 | tee "$OUT_DIR/vbench4.log"; then
  echo "[h200_6h] Rebuild report with official VBench-4D scores"
  python -m sbe_h200_video.rebuild_report \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --official-vbench-json "$OUT_DIR/official_vbench4_scores.json" \
    2>&1 | tee "$OUT_DIR/rebuild_report.log"
else
  if [ "$REQUIRE_OFFICIAL_VBENCH" = "1" ]; then
    echo "[h200_6h] ERROR: Official VBench-4D failed and REQUIRE_OFFICIAL_VBENCH=1."
    echo "[h200_6h] No official VBench claim should be made from this run."
    exit 4
  fi
  echo "[h200_6h] WARNING: Official VBench failed. Keeping proxy-marked table only."
fi

echo "[h200_6h] Done"
echo "  $OUT_DIR/main_table.csv"
echo "  $OUT_DIR/main_table.md"
echo "  $OUT_DIR/official_vbench4_scores.json"
echo "  $OUT_DIR/per_video_rows.csv"
echo "  $OUT_DIR/generation_rows.csv"
echo "  $OUT_DIR/report.tex"
echo "  $OUT_DIR/report.pdf if pdflatex is installed"
