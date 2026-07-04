#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_VBENCH="${INSTALL_VBENCH:-1}"
INSTALL_SPACY_MODEL="${INSTALL_SPACY_MODEL:-1}"

echo "[bootstrap] Python: $($PYTHON_BIN --version)"
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "[bootstrap] Installing base dependencies"
pip install -r requirements.txt
pip install -e .

echo "[bootstrap] Checking ffmpeg through imageio-ffmpeg"
python - <<'PY'
import imageio_ffmpeg
print("imageio-ffmpeg executable:", imageio_ffmpeg.get_ffmpeg_exe())
PY

echo "[bootstrap] Checking spaCy"
python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("spacy") else 1)
PY

if [ "$INSTALL_SPACY_MODEL" = "1" ]; then
  echo "[bootstrap] Installing spaCy English model if absent"
  set +e
  python - <<'PY'
import spacy
try:
    spacy.load("en_core_web_sm")
    print("en_core_web_sm already installed")
except Exception:
    raise SystemExit(1)
PY
  model_ok=$?
  set -e
  if [ "$model_ok" != "0" ]; then
    set +e
    python -m spacy download en_core_web_sm
    model_ok=$?
    set -e
    if [ "$model_ok" != "0" ]; then
      echo "[bootstrap] WARNING: en_core_web_sm download failed; code will use spacy.blank('en') + rule fallback."
    fi
  fi
fi

if [ "$INSTALL_VBENCH" = "1" ]; then
  echo "[bootstrap] Installing official VBench"
  set +e
  pip install vbench
  code=$?
  if [ "$code" != "0" ]; then
    echo "[bootstrap] pip install vbench failed, trying GitHub source"
    pip install "git+https://github.com/Vchitect/VBench.git"
    code=$?
  fi
  set -e
  if [ "$code" != "0" ]; then
    echo "[bootstrap] ERROR: VBench install failed. Official VBench cannot be reported."
    echo "[bootstrap] Re-run with INSTALL_VBENCH=0 only for generation/proxy debugging."
    exit 3
  fi
fi

echo "[bootstrap] Environment status"
python scripts/check_env.py
echo "[bootstrap] Done"
