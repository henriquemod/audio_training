#!/usr/bin/env bash
# Launch RVC's WebUI at http://localhost:7865 using the RVC venv.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVC_DIR="$PROJECT_ROOT/rvc"
RVC_PY="$RVC_DIR/.venv/bin/python"

if [[ ! -x "$RVC_PY" ]]; then
  echo "ERROR: rvc/.venv not found. Run ./scripts/setup_rvc.sh first." >&2
  exit 1
fi

echo "Starting RVC WebUI at http://localhost:7865"
echo "Dataset path to use in the WebUI: $PROJECT_ROOT/dataset/processed"
echo "Press Ctrl+C to stop."
cd "$RVC_DIR"
exec "$RVC_PY" infer-web.py --pycmd "$RVC_PY" --port 7865
