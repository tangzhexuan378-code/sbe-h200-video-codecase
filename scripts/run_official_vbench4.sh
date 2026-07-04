#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  echo "Usage: bash scripts/run_official_vbench4.sh outputs/<run_name>"
  exit 2
fi

source "${VENV_DIR:-.venv}/bin/activate"

if ! command -v vbench >/dev/null 2>&1; then
  echo "[vbench4] ERROR: vbench command not found. Run: bash scripts/bootstrap_env.sh"
  exit 1
fi

VIDEOS_DIR="$RUN_DIR/videos"
WORK_DIR="$RUN_DIR/official_vbench4"
mkdir -p "$WORK_DIR/by_method" "$WORK_DIR/logs"

python - "$RUN_DIR" "$WORK_DIR" <<'PY'
import pathlib
import shutil
import sys

run_dir = pathlib.Path(sys.argv[1])
work_dir = pathlib.Path(sys.argv[2])
videos = run_dir / "videos"
if not videos.exists():
    raise SystemExit(f"Missing generated video directory: {videos}")
methods = [
    "baseline_12step",
    "teacache_12step_t0.2",
    "teacache_12step_t0.3",
    "teacache_12step_t0.45",
    "fastercache_s2",
    "pab_s2_c3",
    "sbe_online_discrete",
    "sbe_online_continuous_no_u",
    "sbe_online_continuous_full",
]
for method in methods:
    target = work_dir / "by_method" / method
    target.mkdir(parents=True, exist_ok=True)
    for src in videos.glob(f"*_{method}.mp4"):
        dst = target / src.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)
PY

for method_dir in "$WORK_DIR"/by_method/*; do
  [ -d "$method_dir" ] || continue
  method="$(basename "$method_dir")"
  if ! compgen -G "$method_dir/*.mp4" >/dev/null; then
    echo "[vbench4] skip $method: no mp4 files"
    continue
  fi
  prompt_file="$method_dir/prompts.json"
  python - "$method_dir" "$prompt_file" <<'PY'
import json
import pathlib
import sys

method_dir = pathlib.Path(sys.argv[1])
prompt_file = pathlib.Path(sys.argv[2])
prompts = {path.name: "" for path in sorted(method_dir.glob("*.mp4"))}
prompt_file.write_text(json.dumps(prompts, indent=2), encoding="utf-8")
PY

  echo "[vbench4] method=$method dimensions=imaging_quality motion_smoothness dynamic_degree"
  vbench evaluate \
    --dimension "imaging_quality motion_smoothness dynamic_degree" \
    --videos_path "$method_dir" \
    --mode=custom_input \
    --prompt_file "$prompt_file" \
    --ngpus="${GPUS:-1}" \
    --output_path "$WORK_DIR/${method}_3d" \
    2>&1 | tee "$WORK_DIR/logs/${method}_3d.log"

  echo "[vbench4] method=$method dimension=temporal_flickering"
  vbench evaluate \
    --dimension "temporal_flickering" \
    --videos_path "$method_dir" \
    --mode=custom_input \
    --prompt_file "$prompt_file" \
    --ngpus="${GPUS:-1}" \
    --output_path "$WORK_DIR/${method}_temporal" \
    2>&1 | tee "$WORK_DIR/logs/${method}_temporal.log"
done

python -m sbe_h200_video.vbench4_summary --run-dir "$RUN_DIR" --work-dir "$WORK_DIR"
echo "[vbench4] Summary JSON: $RUN_DIR/official_vbench4_scores.json"
