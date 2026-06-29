#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  echo "Usage: bash scripts/run_official_vbench_custom.sh outputs/<run_name>"
  exit 2
fi

source "${VENV_DIR:-.venv}/bin/activate"

if ! command -v vbench >/dev/null 2>&1; then
  echo "[vbench] vbench command not found. Run INSTALL_VBENCH=1 bash scripts/setup.sh first."
  exit 1
fi

VIDEOS_DIR="$RUN_DIR/videos"
WORK_DIR="$RUN_DIR/official_vbench_custom"
mkdir -p "$WORK_DIR/by_method" "$WORK_DIR/logs"

python - "$RUN_DIR" "$WORK_DIR" <<'PY'
import pathlib
import shutil
import sys

run_dir = pathlib.Path(sys.argv[1])
work_dir = pathlib.Path(sys.argv[2])
videos = run_dir / "videos"
methods = [
    "baseline_12step",
    "teacache_12step_t0.2",
    "teacache_12step_t0.45",
    "fastercache_s2",
    "sbe_riskgate_v5",
    "uniform_teacache_t0.3",
    "pab_s2_c3",
]
for method in methods:
    target = work_dir / "by_method" / method
    target.mkdir(parents=True, exist_ok=True)
    for src in videos.glob(f"*_{method}.mp4"):
        dst = target / src.name
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except Exception:
                shutil.copy2(src, dst)
PY

DIMENSIONS="${DIMENSIONS:-subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality}"
for method_dir in "$WORK_DIR"/by_method/*; do
  [ -d "$method_dir" ] || continue
  method="$(basename "$method_dir")"
  if ! compgen -G "$method_dir/*.mp4" >/dev/null; then
    echo "[vbench] skip $method: no mp4 files"
    continue
  fi
  for dim in $DIMENSIONS; do
    echo "[vbench] method=$method dimension=$dim"
    (
      cd "$WORK_DIR"
      vbench evaluate \
        --dimension "$dim" \
        --videos_path "$method_dir" \
        --mode=custom_input \
        --ngpus="${GPUS:-1}"
    ) 2>&1 | tee "$WORK_DIR/logs/${method}_${dim}.log"
  done
done

echo "[vbench] Done. Raw official VBench outputs/logs are under: $WORK_DIR"
echo "[vbench] If VBench produced a JSON summary, set OFFICIAL_VBENCH_JSON=/path/to/json and run:"
echo "python -m sbe_h200_video.rebuild_report --config configs/h200_3h.yaml --out-dir $RUN_DIR --official-vbench-json /path/to/json"
