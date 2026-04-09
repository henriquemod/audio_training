#!/usr/bin/env bash
# Clone RVC at a pinned commit, set up its venv, install deps, download weights.
# Idempotent: skips steps that are already done. Use --force to wipe and redo.

set -euo pipefail

RVC_REPO="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
RVC_COMMIT="7ef19867780cf703841ebafb565a4e47d1ea86ff"
RVC_COMMIT_DATE="2024-11-24"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVC_DIR="$PROJECT_ROOT/rvc"
LOG_FILE="$PROJECT_ROOT/scripts/setup_rvc.log"
# Use the root venv's Python (pinned to 3.10 via mise) to create rvc/.venv.
# This avoids depending on mise shell activation inside this script.
ROOT_PYTHON="$PROJECT_ROOT/.venv/bin/python"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

mkdir -p "$(dirname "$LOG_FILE")"
# Note: we deliberately do NOT use `exec > >(tee ...)` here because process
# substitution swallows non-zero exit codes from the piped stage, breaking
# `set -e`. Instead, run the whole script under a subshell piped to tee.
if [[ -z "${_SETUP_RVC_REEXEC:-}" ]]; then
  export _SETUP_RVC_REEXEC=1
  set -o pipefail
  "$0" "$@" 2>&1 | tee -a "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi

echo "=== setup_rvc.sh started at $(date) ==="
echo "RVC commit: $RVC_COMMIT ($RVC_COMMIT_DATE)"

# Pre-flight
echo ""
echo "--- Pre-flight: doctor system checks ---"
"$ROOT_PYTHON" "$PROJECT_ROOT/src/doctor.py" --system-only

# Verify root Python is 3.10 (RVC requires it)
if ! "$ROOT_PYTHON" --version | grep -q "Python 3.10"; then
  echo "ERROR: $ROOT_PYTHON is not Python 3.10. Re-create the root venv with mise-provided Python 3.10." >&2
  exit 1
fi

# Force wipe if requested
if [[ "$FORCE" -eq 1 && -d "$RVC_DIR" ]]; then
  echo ""
  echo "--- --force: removing existing rvc/ ---"
  rm -rf "$RVC_DIR"
fi

# Clone
if [[ ! -d "$RVC_DIR/.git" ]]; then
  echo ""
  echo "--- Cloning RVC ---"
  git clone "$RVC_REPO" "$RVC_DIR"
else
  echo ""
  echo "--- rvc/ already cloned, fetching latest ---"
  git -C "$RVC_DIR" fetch origin
fi

# Pin commit
echo ""
echo "--- Checking out pinned commit $RVC_COMMIT ---"
git -C "$RVC_DIR" checkout "$RVC_COMMIT"
ACTUAL_COMMIT=$(git -C "$RVC_DIR" rev-parse HEAD)
if [[ "$ACTUAL_COMMIT" != "$RVC_COMMIT" ]]; then
  echo "ERROR: checked out commit ($ACTUAL_COMMIT) does not match pinned ($RVC_COMMIT)"
  exit 1
fi

# Create RVC venv with Python 3.10 (seeded from root venv)
if [[ ! -d "$RVC_DIR/.venv" ]]; then
  echo ""
  echo "--- Creating rvc/.venv with Python 3.10 ---"
  "$ROOT_PYTHON" -m venv "$RVC_DIR/.venv"
fi

RVC_PIP="$RVC_DIR/.venv/bin/pip"
RVC_PY="$RVC_DIR/.venv/bin/python"

# Pin pip <24.1 inside RVC venv.
# fairseq 0.12.2 has legacy PEP 440 metadata (`PyYAML>=5.1.*`) that pip 24.1+
# rejects with "ResolutionImpossible". This is the documented workaround for
# RVC's unpinned fairseq dep.
echo ""
echo "--- Pinning pip<24.1 in rvc/.venv (required for fairseq 0.12.2) ---"
"$RVC_PIP" install 'pip<24.1'

# Install PyTorch 2.1 with CUDA 12.1 BEFORE RVC requirements
# (RVC's requirements.txt does not pin torch, so we control it.)
echo ""
echo "--- Installing PyTorch 2.1.2 + CUDA 12.1 ---"
"$RVC_PIP" install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2

# Install RVC's requirements
echo ""
echo "--- Installing RVC requirements ---"
"$RVC_PIP" install -r "$RVC_DIR/requirements.txt"

# Pin gradio_client to match gradio 3.34.0.
# RVC's requirements.txt pins gradio but leaves gradio_client unpinned, so
# pip resolves gradio_client to a much newer version that removed the
# `media_data` symbol gradio 3.34.0 imports. The matched pair is 0.2.7.
echo ""
echo "--- Pinning gradio_client==0.2.7 to match gradio 3.34.0 ---"
"$RVC_PIP" install 'gradio_client==0.2.7'

# Pin matplotlib<3.8 to keep FigureCanvasAgg.tostring_rgb().
# RVC's infer/lib/train/utils.py calls fig.canvas.tostring_rgb() when plotting
# spectrograms during training. That method was deprecated in matplotlib 3.8
# and removed in 3.10, so an unpinned install crashes training with
# AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'.
# 3.7.3 is the last 3.7.x release and is contemporary with RVC's pinned commit.
echo ""
echo "--- Pinning matplotlib==3.7.3 (RVC training uses removed tostring_rgb) ---"
"$RVC_PIP" install 'matplotlib==3.7.3'

# Download pretrained weights
echo ""
echo "--- Downloading pretrained weights (hubert_base.pt, rmvpe.pt, ...) ---"
cd "$RVC_DIR"
"$RVC_PY" tools/download_models.py
cd "$PROJECT_ROOT"

# Verify torch sees CUDA
echo ""
echo "--- Verifying torch + CUDA inside rvc/.venv ---"
"$RVC_PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in rvc/.venv'; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, device {torch.cuda.get_device_name(0)}')"

# Final doctor check
echo ""
echo "--- Running doctor --rvc-only ---"
"$ROOT_PYTHON" "$PROJECT_ROOT/src/doctor.py" --rvc-only

echo ""
echo "=== setup_rvc.sh completed successfully ==="
echo ""
echo "Next steps:"
echo "  1. Put your raw recordings in dataset/raw/"
echo "  2. Run: .venv/bin/python src/preprocess.py"
echo "  3. Run: ./scripts/launch_rvc_webui.sh  (train via the WebUI)"
echo "  4. After training: ./scripts/install_model.sh <experiment_name>"
echo "  5. Generate audio: .venv/bin/python src/generate.py \"Hello world\""
