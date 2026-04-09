#!/usr/bin/env bash
# Bootstrap a bare Ubuntu 22.04 + NVIDIA-driver pod to fully provisioned.
# Idempotent: each layer probes the real artifact and skips if already done.
# Use --force to wipe .venv and reinstall everything.
#
# POD-ONLY script. Requires root (apt). Do not run on a laptop.
#
# Layer order (D-12 hard dependency chain):
#   1. OS detection (Ubuntu 22.04 recommended, 24.04 best-effort)
#   2. apt prerequisites (ca-certificates, wget, gnupg, software-properties-common, xz-utils)
#   3. ffmpeg >= 5.0 (BtbN static build — Ubuntu 22.04 apt ships 4.x, too old)
#   4. CUDA toolkit 12.1 (via NVIDIA apt keyring)
#   5. Python 3.10 acquisition ladder (PATH -> mise -> deadsnakes PPA)
#   6. App venv (.venv) + editable install
#   7. RVC venv + weights (delegated to scripts/setup_rvc.sh — UNCHANGED)
#   8. Weight size floor sanity check
#   9. Final verification: doctor training-phase pre-flight (Plan 01-01 deliverable)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/scripts/setup_pod.log"
RVC_DIR="$PROJECT_ROOT/rvc"
APP_VENV="$PROJECT_ROOT/.venv"
APP_EGG_INFO="$PROJECT_ROOT/src/train_audio_model.egg-info"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
# BtbN static ffmpeg build — GPL variant includes afftdn / loudnorm / silencedetect.
# "latest" release is updated nightly and is ffmpeg 7.x; far above our >=5.0 floor.
FFMPEG_STATIC_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"

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

# Pod containers run as root and typically do not ship sudo. Fail fast with a
# clear message if someone runs this on a non-root laptop shell instead of
# erroring mid-apt.
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "ERROR: setup_pod.sh must run as root (pods run as root; no sudo needed)." >&2
  echo "       Current UID=${EUID:-$(id -u)}. This script is POD-ONLY." >&2
  exit 1
fi

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
  ca-certificates wget gnupg software-properties-common xz-utils

# ---------- Layer: ffmpeg >= 5.0 (BtbN static build) ----------
echo ""
echo "--- Layer: ffmpeg >= 5.0 ---"
# Ubuntu 22.04's apt ships ffmpeg 4.4 — below our >=5.0 floor (MIN_FFMPEG_VERSION
# in src/doctor.py). Install a BtbN static build into /usr/local/bin instead.
# Probe: real artifact version parse, matches how doctor.check_ffmpeg verifies.
_ffmpeg_ok() {
  command -v ffmpeg >/dev/null 2>&1 || return 1
  local ver
  ver="$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
  # Accept either a tagged release (5.1.2) or a nightly/git build (N-123884-...).
  # Nightlies are always on master and far above the 5.0 floor — treat as OK.
  # Matches src/doctor.py:parse_ffmpeg_version.
  if [[ "$ver" =~ ^N-[0-9]+ ]]; then
    :  # nightly build — version floor satisfied by construction
  else
    local major="${ver%%.*}"
    [[ "$major" =~ ^[0-9]+$ ]] || return 1
    (( major >= 5 )) || return 1
  fi
  # Filter presence check — matches REQUIRED_FFMPEG_FILTERS in src/doctor.py.
  local filters
  filters="$(ffmpeg -hide_banner -filters 2>/dev/null)"
  echo "$filters" | grep -q '\bafftdn\b'       || return 1
  echo "$filters" | grep -q '\bloudnorm\b'     || return 1
  echo "$filters" | grep -q '\bsilencedetect\b' || return 1
  return 0
}

if _ffmpeg_ok; then
  echo "ffmpeg already satisfies >=5.0 with required filters ($(ffmpeg -version | head -1))"
else
  # Remove any older apt-installed ffmpeg to avoid PATH shadowing /usr/local/bin.
  if dpkg -l ffmpeg 2>/dev/null | grep -q '^ii'; then
    echo "Removing apt-provided ffmpeg (<5.0) before installing static build..."
    apt-get remove -y ffmpeg
  fi
  echo "Installing BtbN static ffmpeg build from $FFMPEG_STATIC_URL ..."
  FFMPEG_TARBALL="/tmp/ffmpeg-static.tar.xz"
  FFMPEG_EXTRACT_DIR="/tmp/ffmpeg-static"
  rm -rf "$FFMPEG_EXTRACT_DIR"
  mkdir -p "$FFMPEG_EXTRACT_DIR"
  wget -qO "$FFMPEG_TARBALL" "$FFMPEG_STATIC_URL"
  tar -xJf "$FFMPEG_TARBALL" -C "$FFMPEG_EXTRACT_DIR" --strip-components=1
  install -m 0755 "$FFMPEG_EXTRACT_DIR/bin/ffmpeg"  /usr/local/bin/ffmpeg
  install -m 0755 "$FFMPEG_EXTRACT_DIR/bin/ffprobe" /usr/local/bin/ffprobe
  rm -rf "$FFMPEG_TARBALL" "$FFMPEG_EXTRACT_DIR"
  hash -r
fi

# Post-install verification — fail loudly if the static build is missing filters.
_ffmpeg_ok || {
  echo "ERROR: ffmpeg install verification failed — version <5.0 or required filters missing" >&2
  ffmpeg -version 2>&1 | head -3 >&2 || true
  exit 1
}
echo "ffmpeg OK: $(ffmpeg -version | head -1)"

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
echo "--- Layer: RVC venv + weights ---"
# D-07, BOOT-05: setup_rvc.sh is invoked byte-for-byte unchanged. It pins
# pip<24.1 in rvc/.venv (required for fairseq 0.12.2) and downloads
# hubert_base.pt, rmvpe.pt, and pretrained/pretrained_v2/uvr5_weights/* via
# rvc/tools/download_models.py.
#
# Weight caveat: rvc/tools/download_models.py is NOT probe-and-skip — it
# unconditionally re-downloads every weight file on every invocation. On a
# healthy pod this turns a warm re-run into a ~2+ minute network round-trip
# for bytes we already have on disk. We cannot fix download_models.py
# (rvc/ is vendored and pinned) and we cannot modify setup_rvc.sh (BOOT-05).
#
# Pod contract: pods are ephemeral (the whole FS vanishes when the pod is
# destroyed) but stable within their lifetime, and we never need to "upgrade
# to a newer model version" — RVC pretrained weights are frozen artifacts.
# So if all the provisioning state is already in place, it is safe to skip
# the setup_rvc.sh delegation entirely and let the subsequent
# `doctor --training` verification layer be the real gate.
#
# Probe surface (MUST all pass to skip the delegation):
#   1. rvc/ cloned at the pinned commit
#   2. rvc/.venv is a Python 3.10 venv
#   3. Torch importable in rvc/.venv with CUDA available
#   4. All four weight files we size-floor later exist above their floors
#      (hubert, rmvpe, pretrained_v2/f0G40k, pretrained_v2/f0D40k) plus the
#      small uvr5 dereverb sentinel vocals.onnx
# --force disables the skip: the user explicitly asked for a clean reinstall.
RVC_COMMIT_PIN="7ef19867780cf703841ebafb565a4e47d1ea86ff"
_rvc_already_provisioned() {
  [[ "$FORCE" -ne 1 ]] || return 1
  # (1) clone + pinned commit
  [[ -d "$RVC_DIR/.git" ]] || return 1
  local head
  head="$(git -C "$RVC_DIR" rev-parse HEAD 2>/dev/null || echo "")"
  [[ "$head" == "$RVC_COMMIT_PIN" ]] || return 1
  # (2) rvc/.venv is Python 3.10
  local rvc_py="$RVC_DIR/.venv/bin/python"
  [[ -x "$rvc_py" ]] || return 1
  "$rvc_py" --version 2>&1 | grep -q "Python 3.10" || return 1
  # (3) torch importable with CUDA
  "$rvc_py" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1 || return 1
  # (4) weight sentinels above their floors
  local path size min
  for entry in \
    "$RVC_DIR/assets/hubert/hubert_base.pt:100000000" \
    "$RVC_DIR/assets/rmvpe/rmvpe.pt:100000000" \
    "$RVC_DIR/assets/pretrained_v2/f0G40k.pth:30000000" \
    "$RVC_DIR/assets/pretrained_v2/f0D40k.pth:30000000" \
    "$RVC_DIR/assets/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx:1000000"; do
    path="${entry%:*}"
    min="${entry##*:}"
    [[ -f "$path" ]] || return 1
    size=$(stat -c '%s' "$path" 2>/dev/null || echo 0)
    (( size >= min )) || return 1
  done
  return 0
}

if _rvc_already_provisioned; then
  echo "RVC already provisioned (clone, venv, torch+CUDA, weights) — skipping setup_rvc.sh."
  echo "Use --force to wipe and reinstall."
else
  echo "Delegating to scripts/setup_rvc.sh..."
  if [[ "$FORCE" -eq 1 ]]; then
    bash "$PROJECT_ROOT/scripts/setup_rvc.sh" --force
  else
    bash "$PROJECT_ROOT/scripts/setup_rvc.sh"
  fi
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
