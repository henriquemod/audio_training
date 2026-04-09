#!/usr/bin/env bash
# Copy a trained RVC model from rvc/assets/weights and rvc/logs into models/.
# Usage: ./scripts/install_model.sh <experiment_name>

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_name>" >&2
  exit 2
fi

NAME="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVC_DIR="$PROJECT_ROOT/rvc"
MODELS_DIR="$PROJECT_ROOT/models"
mkdir -p "$MODELS_DIR"

# RVC saves the final generator as assets/weights/<name>.pth when you click
# "Train model" in the WebUI.
SRC_PTH="$RVC_DIR/assets/weights/${NAME}.pth"
if [[ ! -f "$SRC_PTH" ]]; then
  echo "ERROR: $SRC_PTH not found." >&2
  echo "       Did you train experiment '$NAME' and click 'Train model'?" >&2
  exit 1
fi

# Index is written under rvc/logs/<name>/added_*.index
SRC_INDEX=$(find "$RVC_DIR/logs/$NAME" -maxdepth 1 -name "added_*.index" 2>/dev/null | head -n1 || true)
if [[ -z "$SRC_INDEX" ]]; then
  echo "ERROR: no added_*.index found under $RVC_DIR/logs/$NAME/" >&2
  echo "       Did you click 'Train feature index' in the WebUI?" >&2
  exit 1
fi

DST_PTH="$MODELS_DIR/${NAME}.pth"
DST_INDEX="$MODELS_DIR/${NAME}.index"

cp -v "$SRC_PTH" "$DST_PTH"
cp -v "$SRC_INDEX" "$DST_INDEX"

echo ""
echo "Installed model '$NAME'."
echo "Add this to your .env to use it by default:"
echo "  DEFAULT_MODEL=$NAME"
