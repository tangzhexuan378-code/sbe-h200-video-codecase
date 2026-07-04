#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/resolve_runtime_paths.sh"
python scripts/preflight_h200.py --json-out "$SBE_WORK_ROOT/preflight_status.json"
bash scripts/bootstrap_env.sh
bash scripts/run_h200_6h.sh
