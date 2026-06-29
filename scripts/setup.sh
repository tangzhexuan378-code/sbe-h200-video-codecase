#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_VBENCH="${INSTALL_VBENCH:-1}"

echo "[setup] Python: $($PYTHON_BIN --version)"
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "[setup] Installing base dependencies"
pip install -r requirements.txt
pip install -e .

echo "[setup] Checking ffmpeg availability"
python - <<'PY'
import imageio_ffmpeg
print("imageio-ffmpeg executable:", imageio_ffmpeg.get_ffmpeg_exe())
PY

if [ "$INSTALL_VBENCH" = "1" ]; then
  echo "[setup] Trying to install official VBench. If this fails, set INSTALL_VBENCH=0 or use --allow-vbench-proxy."
  set +e
  pip install vbench
  code=$?
  if [ "$code" != "0" ]; then
    echo "[setup] pip install vbench failed, trying GitHub source"
    pip install "git+https://github.com/Vchitect/VBench.git"
    code=$?
  fi
  set -e
  if [ "$code" = "0" ]; then
    echo "[setup] VBench installed"
  else
    echo "[setup] WARNING: VBench install failed. Official VBench will be unavailable."
  fi
fi

echo "[setup] Verifying optional modules"
python scripts/check_env.py || true
echo "[setup] Done"
