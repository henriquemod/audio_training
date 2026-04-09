# Stack Research

**Domain:** Headless RVC voice-model training on rented GPU pods (bare Ubuntu + NVIDIA driver)
**Researched:** 2026-04-09
**Confidence:** HIGH for RVC training scripts (sourced from vendored clone); MEDIUM for CUDA install (NVIDIA docs don't publish exact 12.1 commands; verified via community sources + NVIDIA keyring infrastructure); HIGH for mise/Python; MEDIUM for pod base-image landscape.

---

## IMPORTANT SCOPE NOTE

This file covers ONLY the **additional** stack needed for the pod-bootstrap and training-orchestration milestone. The following are fixed upstream constraints — do NOT re-research or change them:

- Python 3.10 (pinned in `pyproject.toml` and `.mise.toml`)
- torch 2.1.2 + torchaudio 2.1.2 + torchvision 0.16.2, CUDA 12.1 wheels
- fairseq 0.12.2 (installed via `rvc/requirements.txt`, requires `pip<24.1`)
- gradio 3.34.0 + gradio_client 0.2.7 + matplotlib 3.7.3 (pinned in `setup_rvc.sh`)
- RVC pinned at commit `7ef19867780cf703841ebafb565a4e47d1ea86ff`
- Two-venv split (`./.venv` app venv + `./rvc/.venv` RVC venv)

---

## 1. CUDA Toolkit Install on a Pod

### Method: NVIDIA apt repository (network deb) + `cuda-toolkit-12-1` meta-package

**Confidence: MEDIUM** — NVIDIA's current installation guide only documents CUDA 13.x. The CUDA 12.1 apt method is well-documented via community sources and NVIDIA's keyring infrastructure (same mechanism, versioned package name). The 12.1 repo packages are still served at NVIDIA's CDN.

#### Ubuntu 22.04 (Jammy) — Fully Supported

```bash
# Idempotent detection: skip if nvcc already reports 12.1
if nvcc --version 2>/dev/null | grep -q "release 12.1"; then
    echo "CUDA 12.1 toolkit already installed, skipping."
else
    # NVIDIA network repo keyring
    KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${KEYRING_DEB}" \
        -O "/tmp/${KEYRING_DEB}"
    sudo dpkg -i "/tmp/${KEYRING_DEB}"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cuda-toolkit-12-1
    # Add to PATH (persists for current session; bootstrap writes to /etc/profile.d/)
    export PATH="/usr/local/cuda-12.1/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}"
fi
```

Key points:
- `cuda-toolkit-12-1` installs ONLY the toolkit (compiler, headers, libraries) — NOT the driver. The driver is already present on the pod.
- `--no-install-recommends` skips ~3 GB of documentation and sample packages.
- `DEBIAN_FRONTEND=noninteractive` prevents debconf from blocking on license prompts.
- `nvcc --version` is the right probe — `nvidia-smi` shows the driver's "max supported CUDA version", not the installed toolkit version.
- Detection logic: check `/usr/local/cuda-12.1/` existence OR `nvcc --version | grep "release 12.1"`.

#### Ubuntu 24.04 (Noble) — CUDA 12.1 NOT Available via Official Apt Repo

**CUDA 12.1 is not packaged for Ubuntu 24.04.** NVIDIA's repository for `ubuntu2404` starts at CUDA 12.4+. The `cuda-toolkit-12-1` package does not exist in the Noble apt repo.

Options for Ubuntu 24.04:
1. **Runfile installer (preferred for version-pinning on 24.04):**
   ```bash
   wget -q "https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run"
   sudo sh cuda_12.1.1_530.30.02_linux.run --toolkit --silent --override
   ```
   The `--toolkit --silent --override` flags skip the driver install (already present), suppress interactive prompts, and bypass driver version mismatch warnings.

2. **Use Ubuntu 22.04 pod images** — this is the pragmatic choice. All three major providers (RunPod, Vast.ai, Lambda Labs) offer Ubuntu 22.04-based images. Since CUDA 12.1 apt support is clean on 22.04, prefer 22.04 base images in documentation and bootstrap checks.

#### Detection Logic (works across both methods)

```bash
cuda_toolkit_installed() {
    # 1. nvcc reports 12.1
    if nvcc --version 2>/dev/null | grep -q "release 12.1"; then return 0; fi
    # 2. toolkit directory exists
    if [ -d "/usr/local/cuda-12.1" ]; then return 0; fi
    # 3. Pip torch already installed with CUDA 12.1 in rvc/.venv (skip toolkit)
    # torch 2.1.2+cu121 ships its own CUDA runtime — toolkit install only needed
    # for nvcc (compilation). If we're only running pre-built torch wheels, the
    # toolkit is optional but strongly recommended for doctor checks.
    return 1
}
```

#### Pod Base-Image Reality (2025)

Most providers ship pods with one of:
- **Bare driver only** (`nvidia-smi` works, `nvcc` absent, no `/usr/local/cuda/`)
- **CUDA runtime image** (libs present at `/usr/local/cuda-X.Y/lib64/`, no compiler/nvcc)
- **CUDA devel image** (full toolkit including nvcc — bootstrap is a no-op)
- **PyTorch base image** (torch pre-installed in system Python, probably wrong Python version for us)

The bootstrap must probe all four and short-circuit correctly. The apt method handles cases 1 and 2. Case 3 is already done. Case 4 requires setting up venvs from scratch but the toolkit is already present.

---

## 2. Python 3.10 Install on a Pod

### Method: mise (preferred) with apt fallback

**Confidence: HIGH** — mise install is a one-liner curl, shims work in non-interactive scripts, Python 3.10 is available via mise's CPython plugin. `.mise.toml` is already committed. No reason to deviate.

#### mise on a pod

```bash
# Install mise if not present
if ! command -v mise &>/dev/null && [ ! -f "$HOME/.local/bin/mise" ]; then
    curl -fsSL https://mise.run | sh
    export PATH="$HOME/.local/bin:${PATH}"
fi

# Install Python 3.10 using .mise.toml (already in repo root)
# mise reads .mise.toml automatically when run from PROJECT_ROOT
"$HOME/.local/bin/mise" install python    # installs the version declared in .mise.toml
"$HOME/.local/bin/mise" reshim python     # rebuild shims after install

# Resolve the Python 3.10 binary without relying on shell activation
MISE_PYTHON="$("$HOME/.local/bin/mise" where python)/bin/python3"
```

Key points:
- mise install is non-interactive by default. No `MISE_YES=1` needed for `mise install` (it only prompts for trust, not for version choices when the version is declared in a config file).
- **DO NOT rely on `mise activate bash` in the bootstrap script.** `activate` hooks into the shell prompt, which never fires in a non-interactive script. Use `mise exec` or the absolute path from `mise where python` instead.
- `mise reshim` is needed after install if shims are used.
- The `.mise.toml` is already committed with `python = "3.10"`, so `mise install` on a fresh box will fetch CPython 3.10.x (latest patch) automatically.

#### mise vs alternatives on a pod

| Method | Non-interactive | Speed | Idempotent | Notes |
|--------|----------------|-------|------------|-------|
| **mise** (recommended) | Yes (curl+shims) | ~2-3 min first time | Yes | User's stated preference; `.mise.toml` already committed |
| deadsnakes PPA + apt | Yes (`-y` flag) | ~1 min | Yes | Python 3.10 is NOT in deadsnakes for Ubuntu 22.04 (system default) or Ubuntu 24.04 (in standard repos) — deadsnakes is irrelevant for 3.10 on these distros |
| System apt `python3.10` | Yes | <1 min | Yes | Ubuntu 22.04 ships Python 3.10 in standard repos; Ubuntu 24.04 also ships 3.10. Viable fallback but bypasses mise pinning |
| pyenv | Yes (`PYENV_ROOT`, curl install) | Similar to mise | Yes | More setup steps than mise; no advantage given mise is already the user's tool |

**Recommendation:** Use mise as primary. Add a fallback: if mise install fails (e.g., no network, build tools missing), fall back to `apt-get install -y python3.10 python3.10-venv python3.10-dev`. The apt fallback works on both Ubuntu 22.04 and 24.04 for Python 3.10 without the deadsnakes PPA.

#### One critical mise gotcha on pods

mise compiles CPython from source by default unless a pre-built binary is available. On a pod with no build tools, `mise install python` can fail. The bootstrap script must ensure build dependencies are present:

```bash
sudo apt-get install -y --no-install-recommends \
    build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev libffi-dev liblzma-dev
```

Alternatively, mise can be told to use python-build-standalone (pre-built): `MISE_PYTHON_DEFAULT_PACKAGES_FILE='' mise install` — but the cleanest approach is to ensure build deps are present so the standard compile path works.

---

## 3. Remote Audio Pull

### Method: `curl` / `wget` for HTTP/HTTPS URLs; `rclone` only if pre-configured

**Confidence: HIGH** — per PROJECT.md, no boto3/awscli/rclone as hard deps. `curl` and `wget` are present on every Ubuntu server image.

#### Decision Matrix

| Source Type | Tool | Notes |
|-------------|------|-------|
| Pre-uploaded local directory | None — already present | Most common case; check `[ -d "$DATASET_DIR" ]` |
| HTTP/HTTPS URL (single file or .tar.gz) | `curl -fL -o` | `-f` fails on HTTP errors, `-L` follows redirects |
| S3/R2 presigned URL | `curl -fL -o` | Presigned URLs are just HTTPS — `curl` works without any S3 SDK |
| S3/R2 with credentials (non-presigned) | `rclone` — only if already configured | If `rclone` is not configured, fail early with a clear error and instructions |

#### Implementation pattern

```bash
pull_dataset() {
    local source="$1"
    local dest_dir="$2"
    if [ -d "$source" ]; then
        # Already a local directory — nothing to do
        DATASET_DIR="$source"
    elif echo "$source" | grep -qE "^https?://"; then
        # HTTP/HTTPS URL — download and extract if .tar.gz/.zip
        mkdir -p "$dest_dir"
        local fname; fname="$(basename "$source" | cut -d? -f1)"
        curl -fL --progress-bar -o "$dest_dir/$fname" "$source"
        case "$fname" in
            *.tar.gz|*.tgz) tar -xzf "$dest_dir/$fname" -C "$dest_dir" --strip-components=1 ;;
            *.zip)           unzip -q "$dest_dir/$fname" -d "$dest_dir" ;;
        esac
        DATASET_DIR="$dest_dir"
    else
        echo "ERROR: --dataset must be a local directory or HTTP/HTTPS URL" >&2
        exit 2
    fi
}
```

**DO NOT add boto3, awscli-v2, or the AWS CLI** as dependencies. They bring a large install surface, require credentials management, and are not needed when presigned URLs work. Document the "generate a presigned URL and pass it with `--dataset`" pattern in the README instead.

---

## 4. Resumable Long-Running CLI Processes

### Method: RVC's built-in checkpoint mechanism (no additional framework needed)

**Confidence: HIGH** — verified directly in the vendored clone at `rvc/infer/modules/train/train.py` lines 208-224 and `rvc/infer/lib/train/utils.py:latest_checkpoint_path`.

#### How RVC's checkpoint resume works (from source)

`train.py` wraps its checkpoint-load in a try/except at startup:

```python
try:  # auto-resume if checkpoints exist
    _, _, _, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
    )
    _, _, _, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
    )
    global_step = (epoch_str - 1) * len(train_loader)
except:  # first run — load pretrained base weights instead
    epoch_str = 1
    global_step = 0
    # ... loads pretrainG / pretrainD
```

`latest_checkpoint_path` selects the checkpoint with the highest numeric step count using `sort(key=lambda f: int("".join(filter(str.isdigit, f))))`.

Checkpoint save behavior is controlled by `-l` / `--if_latest`:
- `-l 1` (save-latest mode): always writes to `G_2333333.pth` and `D_2333333.pth` (fixed filenames). Only the latest checkpoint is kept. Resume works by overwriting the same file.
- `-l 0` (save-all mode): writes `G_<global_step>.pth` and `D_<global_step>.pth` per save. Multiple checkpoints accumulate. Resume picks the numerically highest.

**Recommendation for pod training:** Use `-l 1` (save-latest). It limits checkpoint disk usage to ~2× model size and ensures the resume target is always unambiguous.

#### Lock file / PID file convention

RVC itself has no lock file. Add a sentinel at the `src/train.py` wrapper level:

```bash
LOCK_FILE="$RVC_DIR/logs/$EXPERIMENT_NAME/.training.lock"
if [ -f "$LOCK_FILE" ] && kill -0 "$(cat "$LOCK_FILE")" 2>/dev/null; then
    echo "ERROR: Training already running (PID $(cat "$LOCK_FILE"))" >&2
    exit 1
fi
echo "$$" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT INT TERM
```

This is sufficient for single-pod, single-run usage. No fancier locking (flock, fcntl) needed.

#### The `--save-every` flag matters for pod kills

With default `-se 10` (save every 10 epochs), a pod killed after epoch 9 loses 9 epochs of work. For expensive training runs on minute-billed pods, recommend `-se 5` as the default (configurable via CLI flag).

---

## 5. Headless Progress Reporting

### Method: Python stdlib `logging` (matching RVC's existing approach) + `rich.Console` for the wrapper layer

**Confidence: HIGH**

#### What RVC's training pipeline already emits

- `infer/lib/train/utils.py` sets up `logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)` globally.
- `train.py` uses a per-experiment `logging.FileHandler` writing to `logs/<exp>/train.log` AND a `StreamHandler` to stdout.
- Every epoch logs: `"====> Epoch: {epoch} [{elapsed}]"` via `EpochRecorder`.
- Batch-level loss is logged every N batches to stdout.
- The preprocess/extract scripts use `print()` to stdout (no logging setup).

#### Recommendation for the `src/train.py` wrapper

Do NOT reinvent progress bars for what is already a log-line-per-epoch output. The pattern that works best in a terminal-over-SSH pod session:

1. **Pipe and tee subprocess output** (same pattern as `setup_rvc.sh`):
   ```python
   proc = subprocess.Popen(cmd, cwd=RVC_DIR, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
   with open(log_file, "a") as log:
       for line in proc.stdout:
           print(line, end="", flush=True)  # forward to terminal
           log.write(line)                  # persist to file
   ```
   This gives live output in SSH, survives `tail -f`, and is recoverable after a disconnect.

2. **`rich.Console` for the wrapper-level status** (already a dependency in `.venv`): use it for the pre-training checklist and post-training export summary. Do not use `rich.progress` for the subprocess output — RVC's log lines are not structured enough to drive a progress bar reliably.

3. **DO NOT use tqdm** for the subprocess-forwarding layer. tqdm is a `.venv` app-layer library, not in `rvc/.venv`, and the progress events come from the subprocess's stdout as text — tqdm can't parse them without custom parsing that adds fragility.

#### What "progress reporting" needs to answer at 2 AM

- "Is it still running?" → Process liveness check + tail of log file
- "How far along?" → `tail -20 rvc/logs/<exp>/train.log` shows current epoch
- "When will it finish?" → EpochRecorder timestamps let you do `elapsed / epochs_done × epochs_remaining` in a status command

The `src/train.py` wrapper should emit a one-line summary at startup: `"Training <exp>: epochs 1-200, save every 5, log: rvc/logs/<exp>/train.log"`. That's all a pod operator needs.

---

## 6. RVC Training Entry Points (Verified Against Vendored Clone)

**Confidence: HIGH** — all script paths, argv signatures, and behavior verified by directly reading the files at commit `7ef19867780cf703841ebafb565a4e47d1ea86ff`.

**CRITICAL NOTE:** The training scripts are NOT in `rvc/tools/`. They are in `rvc/infer/modules/train/`. The PROJECT.md requirement mentioning `trainset_preprocess_pipeline_print.py` uses the old script names from earlier RVC versions. At the pinned commit, these scripts have been reorganized.

### Stage 1: Trainset Preprocess

**Script:** `rvc/infer/modules/train/preprocess.py`

**Invocation (from `infer-web.py` line 223):**
```
python infer/modules/train/preprocess.py \
    "<inp_root>" \
    <sr_int> \
    <n_p> \
    "<exp_dir>" \
    <noparallel_bool> \
    <per_float>
```

**Positional argv (no flags, all positional):**

| `sys.argv` index | Name | Type | Description |
|---|---|---|---|
| 1 | `inp_root` | str (path) | Input directory of raw WAV files |
| 2 | `sr` | int | Target sample rate (32000, 40000, or 48000) |
| 3 | `n_p` | int | Number of CPU processes for parallel slicing |
| 4 | `exp_dir` | str (path) | Full path to `logs/<experiment_name>` directory |
| 5 | `noparallel` | "True"/"False" | Disable multiprocessing (use "False" for GPU pods) |
| 6 | `per` | float | Segment length in seconds (default 3.7) |

**Writes to:** `<exp_dir>/0_gt_wavs/`, `<exp_dir>/1_16k_wavs/`, `<exp_dir>/preprocess.log`

**sr_dict mapping (from `infer-web.py` lines 187-191):**
```python
sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}
```
Pass the integer, not the string key.

### Stage 2: F0 Extraction

**Script:** `rvc/infer/modules/train/extract/extract_f0_print.py` (for non-GPU f0 methods)

**Invocation (from `infer-web.py` line 266):**
```
python infer/modules/train/extract/extract_f0_print.py \
    "<exp_dir>" \
    <n_p> \
    <f0method>
```

**Positional argv:**

| `sys.argv` index | Name | Type | Description |
|---|---|---|---|
| 1 | `exp_dir` | str (path) | Full path to `logs/<experiment_name>` |
| 2 | `n_p` | int | Number of CPU processes |
| 3 | `f0method` | str | One of: `"pm"`, `"harvest"`, `"dio"`, `"rmvpe"` |

**Writes to:** `<exp_dir>/2a_f0/`, `<exp_dir>/2b-f0nsf/`, appends to `<exp_dir>/extract_f0_feature.log`

**For `rmvpe` GPU method** (recommended for quality): use `rvc/infer/modules/train/extract/extract_f0_rmvpe.py` instead:
```
python infer/modules/train/extract/extract_f0_rmvpe.py \
    <n_gpus> <gpu_idx> <gpu_id> "<exp_dir>" <is_half_bool>
```
For single-GPU headless use, `n_gpus=1`, `gpu_idx=0`, `gpu_id=0`.

**Recommendation:** Use `rmvpe` (GPU variant) for quality; use `harvest` as a CPU fallback if GPU VRAM is tight during extraction.

### Stage 3: Feature Extraction (HuBERT embeddings)

**Script:** `rvc/infer/modules/train/extract_feature_print.py`

**Invocation (from `infer-web.py` line 359):**
```
python infer/modules/train/extract_feature_print.py \
    <device> \
    <n_part> \
    <i_part> \
    <i_gpu> \
    "<exp_dir>" \
    <version> \
    <is_half>
```

**Positional argv (7-arg form with explicit GPU):**

| `sys.argv` index | Name | Type | Description |
|---|---|---|---|
| 1 | `device` | str | Device string: `"cuda"`, `"cpu"`, `"mps"` |
| 2 | `n_part` | int | Total number of parallel workers |
| 3 | `i_part` | int | This worker's 0-based index |
| 4 | `i_gpu` | str | GPU index (e.g., `"0"`) — sets `CUDA_VISIBLE_DEVICES` |
| 5 | `exp_dir` | str (path) | Full path to `logs/<experiment_name>` |
| 6 | `version` | str | `"v1"` or `"v2"` |
| 7 | `is_half` | str | `"True"` or `"False"` (half-precision) |

**Note:** The script also accepts a 6-arg form (no `i_gpu`), used when `device` already reflects which GPU. For single-GPU headless use, the 7-arg form is cleaner.

**Reads from:** `<exp_dir>/1_16k_wavs/` + `assets/hubert/hubert_base.pt`

**Writes to:** `<exp_dir>/3_feature256/` (v1) or `<exp_dir>/3_feature768/` (v2)

**Appends to:** `<exp_dir>/extract_f0_feature.log`

### Stage 4: Model Training

**Script:** `rvc/infer/modules/train/train.py`

**Invocation (from `infer-web.py` lines 572, 592):**
```
python infer/modules/train/train.py \
    -e "<experiment_name>" \
    -sr <sample_rate_str> \
    -f0 <1_or_0> \
    -bs <batch_size> \
    -g <gpu_ids> \
    -te <total_epoch> \
    -se <save_every_epoch> \
    [-pg <path_to_pretrainG>] \
    [-pd <path_to_pretrainD>] \
    -l <0_or_1> \
    -c <0_or_1> \
    -sw <0_or_1> \
    -v <version>
```

**Named flags (from `infer/lib/train/utils.py:get_hparams` lines 308-366):**

| Flag | Required | Type | Description |
|------|----------|------|-------------|
| `-e` / `--experiment_dir` | Yes | str | Experiment name (NOT a path — resolves to `./logs/<name>`) |
| `-sr` / `--sample_rate` | Yes | str | `"32k"`, `"40k"`, or `"48k"` |
| `-f0` / `--if_f0` | Yes | int | `1` = use f0 (recommended), `0` = no f0 |
| `-bs` / `--batch_size` | Yes | int | Batch size (typical: 4–16 depending on VRAM) |
| `-g` / `--gpus` | No | str | GPU IDs separated by `-` (e.g., `"0"`, `"0-1"`). Default: `"0"` |
| `-te` / `--total_epoch` | Yes | int | Total epochs to train |
| `-se` / `--save_every_epoch` | Yes | int | Checkpoint save frequency in epochs |
| `-pg` / `--pretrainG` | No | str | Path to pretrained Generator weight |
| `-pd` / `--pretrainD` | No | str | Path to pretrained Discriminator weight |
| `-l` / `--if_latest` | Yes | int | `1` = save only latest G/D checkpoints (recommended for pods) |
| `-c` / `--if_cache_data_in_gpu` | Yes | int | `1` = cache dataset in GPU memory (speeds up training if VRAM allows) |
| `-sw` / `--save_every_weights` | No | str | `"1"` = save extractable weight each checkpoint. Default: `"0"` |
| `-v` / `--version` | Yes | str | `"v1"` or `"v2"` |

**Checkpoint auto-resume behavior (verified from source):**
On startup, `train.py` tries to load `G_*.pth` and `D_*.pth` from `logs/<experiment_name>/` using `latest_checkpoint_path` (sorts by embedded step number, takes highest). If found, training resumes from that epoch. If not found (first run), it falls back to `-pg`/`-pd` pretrained bases. This is fully automatic — no separate "resume" flag needed. **Re-invocation with the same `-e` name is all that's required to resume.**

**Writes to:** `logs/<experiment_name>/G_*.pth`, `D_*.pth`, per-epoch weights to `assets/weights/<name>.pth` (if `-sw 1`), `logs/<experiment_name>/train.log`

### Stage 5: FAISS Index Build (inline in infer-web.py — no separate script)

The `train_index` function in `infer-web.py` (lines 616-700) runs entirely in-process: it reads `<exp_dir>/3_feature768/` (v2) or `3_feature256/` (v1), builds a FAISS IVF index, and writes:
- `<exp_dir>/trained_IVF<N>_Flat_nprobe_1_<name>_<version>.index` (intermediate)
- `<exp_dir>/added_IVF<N>_Flat_nprobe_1_<name>_<version>.index` (final, the `.index` file used at inference)

**To call this headlessly:** the logic must be extracted into a standalone Python script (or re-implemented in `src/train.py`), since there is no `tools/train-index.py` that accepts clean argv. The `tools/infer/train-index-v2.py` file has hardcoded paths and is not a general-purpose CLI. Recommend implementing index-building inline in `src/train.py` by importing the logic or replicating the 30-line faiss block.

### Pretrained Base Weights Required for Training

Located at `rvc/assets/pretrained_v2/` (downloaded by `tools/download_models.py` during `setup_rvc.sh`):

| File | Used For |
|------|---------|
| `f0G40k.pth` | Pretrained Generator, 40k sample rate, with f0 (most common) |
| `f0D40k.pth` | Pretrained Discriminator, 40k sample rate, with f0 |
| `f0G32k.pth` | Pretrained Generator, 32k sample rate, with f0 |
| `f0D32k.pth` | Pretrained Discriminator, 32k sample rate, with f0 |
| `f0G48k.pth` | Pretrained Generator, 48k sample rate, with f0 |
| `f0D48k.pth` | Pretrained Discriminator, 48k sample rate, with f0 |
| `G40k.pth` | Generator, 40k, no f0 (rare usage) |
| `D40k.pth` | Discriminator, 40k, no f0 |

Paths passed to `-pg`/`-pd` must be relative to `cwd=RVC_DIR` (i.e., `"assets/pretrained_v2/f0G40k.pth"`).

`download_models.py` already handles downloading all of these. The bootstrap script must verify they exist before starting training.

### Full Headless Training Invocation Sequence

Executed with `cwd=RVC_DIR`, all via `rvc/.venv/bin/python`:

```bash
EXP="my_experiment"
EXP_DIR="$RVC_DIR/logs/$EXP"
SR=40000         # integer for preprocess
SR_STR="40k"     # string for train.py
N_CPU=$(nproc)   # parallelism
DEVICE="cuda"
VERSION="v2"
IS_HALF="True"
GPU_ID="0"
EPOCHS=200
SAVE_EVERY=5
BATCH_SIZE=8

mkdir -p "$EXP_DIR"

# Stage 1: preprocess
"$RVC_PY" infer/modules/train/preprocess.py \
    "$DATASET_DIR" "$SR" "$N_CPU" "$EXP_DIR" "False" 3.7

# Stage 2: F0 extraction (rmvpe GPU)
"$RVC_PY" infer/modules/train/extract/extract_f0_rmvpe.py \
    1 0 0 "$EXP_DIR" "$IS_HALF"

# Stage 3: HuBERT feature extraction
"$RVC_PY" infer/modules/train/extract_feature_print.py \
    "$DEVICE" 1 0 "$GPU_ID" "$EXP_DIR" "$VERSION" "$IS_HALF"

# Stage 4: generate filelist (must be done inline — infer-web.py does this between
#           stage 3 and train.py; requires replicating the filelist.txt generation logic)
# (implemented in src/train.py wrapper)

# Stage 5: train
"$RVC_PY" infer/modules/train/train.py \
    -e "$EXP" -sr "$SR_STR" -f0 1 -bs "$BATCH_SIZE" -g "$GPU_ID" \
    -te "$EPOCHS" -se "$SAVE_EVERY" \
    -pg "assets/pretrained_v2/f0G${SR_STR}.pth" \
    -pd "assets/pretrained_v2/f0D${SR_STR}.pth" \
    -l 1 -c 0 -sw 1 -v "$VERSION"

# Stage 6: FAISS index build (inline Python — no script to call)
# (implemented in src/train.py wrapper)
```

**Critical detail:** Between stage 3 and stage 4, `infer-web.py` generates a `filelist.txt` in `<exp_dir>/` (lines 500-546). This file lists `wav_path|feature_npy_path|speaker_id` for every training sample. `train.py` reads this file via `hps.data.training_files`. The `src/train.py` wrapper MUST replicate this filelist generation step — it is not done by any of the four subprocess scripts.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| CUDA install on Ubuntu 22.04 | `apt + cuda-toolkit-12-1` | runfile installer | Runfile installs driver + toolkit by default; requires `--toolkit --silent --override` flags; more fragile on pods where driver is already installed by the provider |
| CUDA install on Ubuntu 24.04 | runfile `12.1.1_530.30.02_linux.run` with `--toolkit --silent --override` | Skip Ubuntu 24.04 entirely (use 22.04 images) | 22.04 is the pragmatic choice; document 24.04 as unsupported |
| Python version management | mise (primary) + apt fallback | pyenv | pyenv has no advantage over mise; user's preference is mise; `.mise.toml` already committed |
| Remote audio pull | `curl -fL` | boto3, awscli-v2, rclone | boto3/awscli require credential management and large install; rclone requires separate config; presigned URLs make `curl` sufficient for the stated use cases |
| Progress reporting | stdlib `logging` + pipe-and-tee | tqdm, rich.progress | RVC already emits line-per-epoch log output; wrapping it in a progress bar requires fragile stdout parsing; tqdm not in rvc/.venv |
| Checkpoint resume | RVC's native G_/D_ mechanism | Manual epoch tracking, external state | RVC already does automatic resume natively (verified in source) — no additional framework needed |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `boto3` / `awscli-v2` in app venv | Large dep chain, credential management, breaks venv hygiene; out of scope per PROJECT.md | `curl -fL` with presigned S3/R2 URLs |
| `rclone` as a hard dep | Requires separate `rclone.conf` config and `rclone` binary install; adds setup complexity for a use case that presigned URLs cover | `curl` for URLs; document rclone as an optional advanced path |
| `pip install` into `rvc/.venv` for any new packages | Risks breaking the pinned fairseq/gradio/matplotlib combo; `pip<24.1` constraint is load-bearing | Any new runtime dep goes in `.venv` (app) only |
| `shell=True` in any subprocess call | Existing discipline; injection risk; breaks on paths with spaces | `subprocess.run(list_of_args, cwd=RVC_DIR)` |
| `mise activate bash` in bootstrap scripts | `activate` hooks into shell prompt — does not fire in non-interactive scripts | Use `mise exec` or absolute path via `mise where python` |
| `tensorboard` for pod monitoring | Requires a port-forward to view; not compatible with headless pod usage | `tail -f rvc/logs/<exp>/train.log` and a `--status` subcommand |
| `torch.distributed` / multi-GPU setup | `train.py` uses DDP but single-GPU is the supported case per PROJECT.md | Pass `-g 0` (single GPU); leave multi-GPU for a future milestone |
| `infer-web.py` for headless training | Requires gradio, opens port 7865, expects manual browser interaction | Direct subprocess invocation of the training scripts as documented above |

---

## Version Compatibility

| Component | Version | Constraint Source | Notes |
|-----------|---------|------------------|-------|
| Python | 3.10.x | `.mise.toml`, `pyproject.toml` | Fixed |
| torch | 2.1.2+cu121 | `setup_rvc.sh` | Fixed in rvc/.venv |
| pip (rvc/.venv) | <24.1 | `setup_rvc.sh` | Required for fairseq 0.12.2 PEP 440 metadata |
| CUDA toolkit | 12.1 | Matches torch cu121 wheels | `nvcc` version must match torch's CUDA ABI |
| NVIDIA driver | >=530.30 | CUDA 12.1 minimum requirement | Verified via `nvidia-smi`; pod providers typically ship 535+ |
| Ubuntu | 22.04 LTS (preferred) or 24.04 | 22.04: apt `cuda-toolkit-12-1` available; 24.04: requires runfile | Prefer 22.04 pod images |
| faiss-cpu | installed via `rvc/requirements.txt` | RVC dep | The index build step uses faiss — do not re-install or upgrade |
| matplotlib | 3.7.3 | `setup_rvc.sh` | Pinned; 3.8+ removes `tostring_rgb` used in training utils |

---

## Sources

- `rvc/infer/modules/train/preprocess.py` — argv signatures verified (Stage 1)
- `rvc/infer/modules/train/extract/extract_f0_print.py` — argv signatures verified (Stage 2)
- `rvc/infer/modules/train/extract_feature_print.py` — argv signatures verified (Stage 3)
- `rvc/infer/modules/train/train.py` + `rvc/infer/lib/train/utils.py:get_hparams` — named flags + auto-resume logic verified (Stage 4)
- `rvc/infer-web.py` lines 218-609 — exact invocation strings used by the webui; ground truth for subprocess construction
- `rvc/assets/pretrained_v2/` — directory listing; confirmed presence of all 8 weight files
- NVIDIA CUDA Installation Guide (current, 13.x): `https://docs.nvidia.com/cuda/cuda-installation-guide-linux/` — keyring infrastructure (`cuda-keyring_1.1-1_all.deb`) confirmed still used; specific 12.1 package name (`cuda-toolkit-12-1`) confirmed via community guides [MEDIUM confidence]
- mise.jdx.dev/installing-mise.html — `curl https://mise.run | sh` is the canonical non-interactive install [HIGH confidence]
- mise.jdx.dev/dev-tools/shims.html — shims pattern for non-interactive scripts confirmed [HIGH confidence]
- Deadsnakes PPA documentation — confirmed Python 3.10 is NOT in deadsnakes for Ubuntu 22.04 (system default) or 24.04 [HIGH confidence]
- RVC GitHub issues #837, #953, #1199, #2417 — checkpoint resume behavior and known issues documented [MEDIUM confidence]

---

*Stack research for: headless RVC training, pod bootstrap layer*
*Researched: 2026-04-09*
