#!/usr/bin/env bash
set -euo pipefail

SBE_MIN_FREE_GB="${SBE_MIN_FREE_GB:-120}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

candidate_roots=()
if [ -n "${SBE_WORK_ROOT:-}" ]; then
  candidate_roots+=("$SBE_WORK_ROOT")
fi
candidate_roots+=(
  "/root/autodl-tmp/sbe-h200-video-codecase"
  "/mnt/data/sbe-h200-video-codecase"
  "/data/sbe-h200-video-codecase"
  "$HOME/sbe-h200-video-codecase-work"
  "/tmp/sbe-h200-video-codecase"
  "$REPO_ROOT/.runtime"
)

choose_root() {
  local root free_gb
  for root in "${candidate_roots[@]}"; do
    [ -n "$root" ] || continue
    if mkdir -p "$root" "$root/cache" "$root/outputs" "$root/venvs" 2>/dev/null; then
      if touch "$root/.write_test" 2>/dev/null; then
        rm -f "$root/.write_test"
        free_gb="$(python - "$root" <<'PY'
import shutil, sys
print(int(shutil.disk_usage(sys.argv[1]).free / (1024**3)))
PY
)"
        if [ "$free_gb" -ge "$SBE_MIN_FREE_GB" ]; then
          echo "$root"
          return 0
        fi
        echo "[paths] skip $root: only ${free_gb}GB free, need ${SBE_MIN_FREE_GB}GB" >&2
      fi
    fi
  done
  return 1
}

if ! RESOLVED_SBE_WORK_ROOT="$(choose_root)"; then
  echo "[paths] ERROR: no writable runtime path with enough free space." >&2
  echo "[paths] Set SBE_WORK_ROOT=/path/with/space or lower SBE_MIN_FREE_GB for smoke tests." >&2
  return 2 2>/dev/null || exit 2
fi

export SBE_WORK_ROOT="$RESOLVED_SBE_WORK_ROOT"
export VENV_DIR="${VENV_DIR:-$SBE_WORK_ROOT/venvs/sbe-h200}"
export HF_HOME="${HF_HOME:-$SBE_WORK_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SBE_WORK_ROOT/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-$SBE_WORK_ROOT/cache/torch}"
export VBENCH_CACHE="${VBENCH_CACHE:-$SBE_WORK_ROOT/cache/vbench}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SBE_WORK_ROOT/cache/pip}"
export OUT_BASE_DIR="${OUT_BASE_DIR:-$SBE_WORK_ROOT/outputs}"

mkdir -p "$VENV_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$VBENCH_CACHE" "$PIP_CACHE_DIR" "$OUT_BASE_DIR"

echo "[paths] SBE_WORK_ROOT=$SBE_WORK_ROOT"
echo "[paths] VENV_DIR=$VENV_DIR"
echo "[paths] HF_HOME=$HF_HOME"
echo "[paths] OUT_BASE_DIR=$OUT_BASE_DIR"
