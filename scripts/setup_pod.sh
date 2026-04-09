#!/usr/bin/env bash
# Bootstrap a bare Ubuntu 22.04 + NVIDIA-driver pod to fully provisioned.
# Idempotent: each layer probes the real artifact and skips if already done.
# Use --force to wipe .venv and reinstall everything.
#
# POD-ONLY script. Requires root (apt). Do not run on a laptop.
#
# Layer order (D-12 hard dependency chain):
#   1. OS detection (Ubuntu 22.04 recommended, 24.04 best-effort)
#   2. apt prerequisites (ca-certificates, wget, gnupg, software-properties-common)
#   3. CUDA toolkit 12.1 (via NVIDIA apt keyring)
#   4. Python 3.10 acquisition ladder (PATH -> mise -> deadsnakes PPA)
#   5. App venv (.venv) + editable install
#   6. RVC venv + weights (delegated to scripts/setup_rvc.sh — UNCHANGED)
#   7. Weight size floor sanity check
#   8. Final verification: doctor training-phase pre-flight (Plan 01-01 deliverable)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/scripts/setup_pod.log"
RVC_DIR="$PROJECT_ROOT/rvc"
APP_VENV="$PROJECT_ROOT/.venv"
APP_EGG_INFO="$PROJECT_ROOT/src/train_audio_model.egg-info"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

mkdir -p "$(dirname "$LOG_FILE")"
# Note: we deliberately do NOT use `exec > >(tee ...)` here because process
# substitution swallows non-zero exit codes from the piped stage, breaking
# `set -e`. Instead, run the whole script under a subshell piped to tee.
if [[ -z "${_SETUP_POD_REEXEC:-}" ]]; then
  export _SETUP_POD_REEXEC=1
  set -o pipefail
  "$0" "$@" 2>&1 | tee -a "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi

echo "=== setup_pod.sh started at $(date) ==="

# Mandatory envelope for every apt invocation in this script (D-02/D-03).
# Prevents tzdata / dpkg interactive prompts that would hang on a billing pod.
export DEBIAN_FRONTEND=noninteractive
export TZ=UTC

# ---------- Layer: OS detection ----------
echo ""
echo "--- Layer: OS detection ---"
# Ubuntu version gate. 22.04 = happy path. 24.04 = best-effort warn. Other = exit 1.
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  case "${VERSION_ID:-}" in
    "22.04")
      echo "Detected Ubuntu 22.04 — proceeding."
      ;;
    "24.04")
      echo "WARNING: Ubuntu 24.04 is best-effort. 22.04 is the recommended pod image (DOCS-02)."
      echo "         Will attempt the 22.04 apt keyring; if it fails, rent a 22.04 pod."
      ;;
    *)
      echo "ERROR: only Ubuntu 22.04 (recommended) or 24.04 (best-effort) are supported." >&2
      echo "       detected VERSION_ID=${VERSION_ID:-unknown}" >&2
      exit 1
      ;;
  esac
else
  echo "ERROR: /etc/os-release not found — cannot detect distro." >&2
  exit 1
fi

# ---------- Layer: apt prerequisites ----------
echo ""
echo "--- Layer: apt prerequisites ---"
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates wget gnupg software-properties-common

# ---------- Layer: CUDA toolkit 12.1 ----------
echo ""
echo "--- Layer: CUDA toolkit 12.1 ---"
# D-01: primary probe is nvcc --version (real artifact), not dpkg -l (package state).
# A half-installed package fails the probe and triggers reinstall.
if command -v nvcc >/dev/null 2>&1 && nvcc --version 2>/dev/null | grep -q "release 12.1"; then
  echo "CUDA 12.1 already installed ($(nvcc --version | grep release))"
else
  echo "Installing CUDA toolkit 12.1 via NVIDIA apt keyring..."
  KEYRING_DEB="/tmp/$(basename "$CUDA_KEYRING_URL")"
  wget -qO "$KEYRING_DEB" "$CUDA_KEYRING_URL"
  dpkg -i "$KEYRING_DEB"
  rm -f "$KEYRING_DEB"
  apt-get update -y
  # --no-install-recommends saves ~500 MB of X11/Nsight GUI deps we never use.
  apt-get install -y --no-install-recommends cuda-toolkit-12-1
fi

# D-04: in-script PATH only — no writes to system profile dirs or user shell rc.
# Re-rented pod = clean slate. Security-positive; no persistent PATH pollution.
export PATH="/usr/local/cuda-12.1/bin:$PATH"

# Post-install verify.
nvcc --version | grep -q "release 12.1" || {
  echo "ERROR: CUDA 12.1 install verification failed — nvcc does not report release 12.1" >&2
  exit 1
}

# ---------- Layer: Python 3.10 acquisition ladder ----------
echo ""
echo "--- Layer: Python 3.10 ---"
PY310=""

# Rung 1: app venv already 3.10 — full skip is handled by the venv layer below.
# Here we only need *some* python3.10 interpreter to seed the venv.

# Rung 2: python3.10 already on PATH (PyTorch base images often ship this).
if [[ -z "$PY310" ]] && command -v python3.10 >/dev/null 2>&1; then
  if python3.10 --version 2>&1 | grep -q "Python 3.10"; then
    PY310="$(command -v python3.10)"
    echo "Found python3.10 in PATH at $PY310"
  fi
fi

# Rung 3: mise — use `mise where` only; shell-activation helpers do not work
# in non-interactive scripts (known STATE.md pitfall).
if [[ -z "$PY310" ]] && command -v mise >/dev/null 2>&1; then
  echo "Trying mise install python@3.10..."
  mise install python@3.10
  MISE_PY_PREFIX="$(mise where python@3.10)"
  if [[ -x "$MISE_PY_PREFIX/bin/python3" ]]; then
    PY310="$MISE_PY_PREFIX/bin/python3"
    echo "Using mise-provided Python at $PY310"
  fi
fi

# Rung 4: deadsnakes PPA fallback.
if [[ -z "$PY310" ]]; then
  echo "Falling back to deadsnakes PPA..."
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -y
  apt-get install -y python3.10 python3.10-venv python3.10-dev
  PY310="/usr/bin/python3.10"
fi

"$PY310" --version | grep -q "Python 3.10" || {
  echo "ERROR: resolved Python ($PY310) is not 3.10.x" >&2
  exit 1
}
echo "Python 3.10 resolved: $PY310"

# ---------- Layer: app venv ----------
echo ""
echo "--- Layer: app venv ($APP_VENV) ---"
if [[ "$FORCE" -eq 1 && -d "$APP_VENV" ]]; then
  echo "--force: wiping existing $APP_VENV"
  rm -rf "$APP_VENV"
fi

if [[ ! -x "$APP_VENV/bin/python" ]] || \
   ! "$APP_VENV/bin/python" --version 2>&1 | grep -q "Python 3.10"; then
  echo "Creating $APP_VENV with $PY310 ..."
  "$PY310" -m venv "$APP_VENV"
else
  echo "$APP_VENV already exists and is Python 3.10 — skipping create."
fi

# Editable install probe-and-skip (D-06). pyproject.toml [project].name =
# "train-audio-model" → hyphen→underscore → src/train_audio_model.egg-info.
if [[ ! -d "$APP_EGG_INFO" ]]; then
  echo "Installing project editable (pip install -e '.[dev]') ..."
  "$APP_VENV/bin/pip" install -e "${PROJECT_ROOT}[dev]"
else
  echo "Editable install already present ($APP_EGG_INFO) — skipping."
fi

# BOOT-07: we deliberately do NOT upgrade pip in .venv, and we do NOT touch
# rvc/.venv's pip anywhere in this script. The pip<24.1 pin in rvc/.venv is
# managed entirely by scripts/setup_rvc.sh (required for fairseq 0.12.2).

# ---------- Layer: RVC venv + weights (delegated) ----------
echo ""
echo "--- Layer: RVC venv + weights (delegating to scripts/setup_rvc.sh) ---"
# D-07, BOOT-05: setup_rvc.sh is invoked unchanged. It is itself idempotent
# (probe-and-skip for clone/venv/weights) and writes its own log via its own
# re-exec+tee. It pins pip<24.1 in rvc/.venv and downloads hubert_base.pt,
# rmvpe.pt, and pretrained_v2/* via rvc/tools/download_models.py.
if [[ "$FORCE" -eq 1 ]]; then
  bash "$PROJECT_ROOT/scripts/setup_rvc.sh" --force
else
  bash "$PROJECT_ROOT/scripts/setup_rvc.sh"
fi

# ---------- Layer: weight file size floors ----------
echo ""
echo "--- Layer: weight file size floors ---"
# BOOT-08 double-check: setup_rvc.sh already downloaded these, but a truncated
# download would silently pass its step. Enforce minimum byte sizes as an
# integrity-lite guard (cryptographic hash pinning is V2 scope — T-02-08).
_check_size() {
  local path="$1"
  local min_bytes="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing $path" >&2
    exit 1
  fi
  local size
  size=$(stat -c '%s' "$path")
  if (( size < min_bytes )); then
    echo "ERROR: $path is only $size bytes (min $min_bytes) — truncated download?" >&2
    exit 1
  fi
  echo "OK: $path ($size bytes)"
}
_check_size "$RVC_DIR/assets/hubert/hubert_base.pt"           100000000
_check_size "$RVC_DIR/assets/rmvpe/rmvpe.pt"                  100000000
_check_size "$RVC_DIR/assets/pretrained_v2/f0G40k.pth"         30000000
_check_size "$RVC_DIR/assets/pretrained_v2/f0D40k.pth"         30000000

# ---------- Layer: post-install verification (doctor --training) ----------
echo ""
echo "--- Layer: post-install verification (doctor --training) ---"
# BOOT-06 / D-08: torch.cuda.is_available() check covered transitively via
# the doctor training pre-flight below → check_rvc_torch_cuda (crosses venv
# boundary via subprocess into rvc/.venv — matches the two-venv discipline).
cd "$PROJECT_ROOT"
"$APP_VENV/bin/python" src/doctor.py --training

echo ""
echo "=== setup_pod.sh completed successfully ==="
echo ""
echo "Next steps:"
echo "  1. Upload or pull training audio into dataset/raw/"
echo "  2. Run: .venv/bin/python src/train.py  (Phase 2)"
