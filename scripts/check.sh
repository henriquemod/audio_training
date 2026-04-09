#!/usr/bin/env bash
# Run doctor + unit tests + integration tests + ruff.
# Skips integration tests gracefully if ffmpeg or network is unavailable.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY=".venv/bin/python"
PYTEST=".venv/bin/pytest"
RUFF=".venv/bin/ruff"

echo "=== doctor ==="
"$PY" src/doctor.py --system-only

echo ""
echo "=== unit tests ==="
"$PYTEST" tests/unit -v

echo ""
echo "=== integration tests (skipping network) ==="
"$PYTEST" tests/integration -v -m "not network"

echo ""
echo "=== ruff lint ==="
"$RUFF" check src/ tests/

echo ""
echo "=== ruff format (check only) ==="
"$RUFF" format --check src/ tests/

echo ""
echo "✓ All checks passed."
