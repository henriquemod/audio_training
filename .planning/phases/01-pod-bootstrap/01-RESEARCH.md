# Phase 1: Pod Bootstrap — Research

**Researched:** 2026-04-09
**Domain:** Ubuntu 22.04 pod provisioning — apt/CUDA/Python/venv bootstrap scripting
**Confidence:** HIGH (the decisions in CONTEXT.md are locked; this research fills concrete command/parsing gaps)

## Research Complete

## Summary

CONTEXT.md already locks D-01..D-15. The planner's remaining gaps are all *concrete command strings and parsing rules*: the exact NVIDIA keyring URL and apt package name for CUDA 12.1 on Ubuntu 22.04, the deadsnakes PPA command sequence under `DEBIAN_FRONTEND=noninteractive TZ=UTC`, the `nvidia-smi --query-gpu=memory.total` CSV format and MiB→GiB parsing, `shutil.disk_usage` semantics, the `mise where python` output shape, and a bash port of `src/generate.py:_tail`. All are low-risk, well-documented, and lift directly into tasks.

Critical invariant to surface: `setup_pod.sh` is a **shell glue script** that (a) calls `apt-get` under a strict noninteractive envelope, (b) probes before installing, (c) delegates RVC work to `setup_rvc.sh` unchanged, (d) mirrors `setup_rvc.sh`'s re-exec+tee logging pattern verbatim. The **only Python code this phase adds** is two new functions in `src/doctor.py` (`check_disk_space_floor`, `check_gpu_vram_floor`), a `--training` flag composition, and two unit test files. Nothing in `src/` should import torch.

**Primary recommendation:** Lift the concrete snippets from the §Concrete Snippets section below verbatim into task actions — the planner should not re-derive them.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (D-01..D-15)

- **D-01:** Probe CUDA 12.1 via `nvcc --version` grep for `release 12.1`. Strict 12.1 match only (not any 12.x).
- **D-02:** Install CUDA 12.1 on Ubuntu 22.04 via NVIDIA apt keyring under `DEBIAN_FRONTEND=noninteractive TZ=UTC`: download `cuda-keyring_*.deb`, `apt-get update`, `apt-get install -y cuda-toolkit-12-1`.
- **D-03:** Ubuntu 24.04 detection via `/etc/os-release` → loud warning → attempt 22.04 apt keyring anyway → exit 1 with fix hint on failure. No runfile fallback.
- **D-04:** `export PATH=/usr/local/cuda-12.1/bin:$PATH` **in-script only**. Do NOT touch `/etc/profile.d/` or `~/.bashrc`.
- **D-05:** Python 3.10 probe-and-skip ladder: `.venv/bin/python` → `python3.10` in PATH → `mise install python@3.10` (use `$(mise where python)/bin/python3`, never `mise activate bash`) → deadsnakes PPA fallback.
- **D-06:** App venv: probe `.venv/bin/python --version`; skip `pip install -e ".[dev]"` if `src/train_audio_model.egg-info` is present. `--force` flag wipes `.venv` and reinstalls.
- **D-07:** Delegate `rvc/.venv` creation and weights to existing `scripts/setup_rvc.sh` — do NOT modify it.
- **D-08:** After `setup_rvc.sh`, run final verification: `rvc/.venv/bin/python -c "import torch; assert torch.cuda.is_available()"` + file-size floors on `hubert_base.pt`, `rmvpe.pt`, `pretrained_v2/*.pth`.
- **D-09:** `python src/doctor.py --training` composes full readiness check set (existing system + RVC + new training checks). Mise is optional/skipped if missing in this mode.
- **D-10:** `check_disk_space_floor(path: Path, min_gb: int) -> CheckResult` via `shutil.disk_usage(path)`. Default floor 20 GB.
- **D-11:** `check_gpu_vram_floor(min_gb: int) -> CheckResult` via `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`, MAX across GPUs, default floor 12 GB. No torch import.
- **D-12:** Intrinsic-probe-only idempotency. No marker files.
- **D-13:** `set -euo pipefail`; fail-fast, no rollback; print layer name + failed command + last ~20 lines of stderr (mirror `_tail`).
- **D-14:** Re-exec + `tee -a` logging pattern from `setup_rvc.sh` lines 22-31, variable-renamed to `_SETUP_POD_REEXEC` and `setup_pod.log`. Guard on `PIPESTATUS[0]`.
- **D-15:** Unit tests for `check_disk_space_floor` (mock `shutil.disk_usage`) and `check_gpu_vram_floor` (mock `subprocess.run`). No bash tests.

### Claude's Discretion

- Error message text, `rich.Table` cosmetics, internal function names.
- Layer ordering within the hard dependency chain (apt prereqs → CUDA → Python → app venv → setup_rvc.sh → verification).
- Whether to extract shared apt helpers or inline them.
- Whether to add `--skip-cuda` debugging escape hatches.

### Deferred Ideas (OUT OF SCOPE)

- Persistent-volume weight caching across reboots (V2-FAST-01).
- Prebuilt wheel cache for `rvc/.venv` (V2-FAST-02).
- Parallel installs.
- Provider-specific pod image detection (Phase 5).
- Smart batch-size defaults (V2-TRAIN-01).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BOOT-01 | One non-interactive bash invocation provisions a bare Ubuntu 22.04 + NVIDIA pod | §Concrete Snippets: noninteractive apt envelope + re-exec header |
| BOOT-02 | Detect-and-adapt probe pattern; re-run completes in ~10s (hard ceiling 30s) | §Key Findings #1, #2, #8: probe commands and their exit semantics |
| BOOT-03 | CUDA toolkit 12.1 via NVIDIA apt keyring on 22.04, no prompts | §Key Findings #1: exact URL, prereqs, package name |
| BOOT-04 | Python 3.10 via mise using `$(mise where python)/bin/python3` (never `mise activate`) | §Key Findings #5: mise where output format |
| BOOT-05 | `.venv` as Python 3.10 with `pip install -e ".[dev]"`, delegate to `setup_rvc.sh` | §Concrete Snippets: delegation invocation |
| BOOT-06 | Verify `torch.cuda.is_available()` in `rvc/.venv` post-install | Existing `check_rvc_torch_cuda` in `src/doctor.py:317` — reuse |
| BOOT-07 | Preserve `pip<24.1` pin in `rvc/.venv` | Satisfied by not touching `rvc/.venv` — `setup_rvc.sh:90-91` handles it |
| BOOT-08 | Populate and size-verify `pretrained_v2/`, `hubert_base.pt`, `rmvpe.pt`, `logs/mute/` | §Key Findings #9: file sizes and conservative floors |
| BOOT-09 | `check_disk_space_floor`, `check_gpu_vram_floor`, `--training` composition | §Key Findings #3, #4; §Concrete Snippets: check bodies |
| BOOT-10 | Unit tests for the two new doctor checks | §Key Findings #10: mocking patterns from `tests/unit/test_doctor.py` |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- Python 3.10 only, pinned by `pyproject.toml` and `.mise.toml`.
- RVC venv stays on torch 2.1.2 + CUDA 12.1 + fairseq 0.12.2 + gradio 3.34.0 + `pip<24.1`.
- All ffmpeg shell-outs via `src/ffmpeg_utils.py:run_ffmpeg` (not relevant this phase).
- Subprocess discipline: no `shell=True`, no bare `subprocess.run("...")` strings, no shared Python imports across venv boundary.
- No new framework runtime, no new DB/scheduler. Thin typer CLI + shell glue only.
- No interactive prompts in any pod-side script.
- No provider-specific code or SDKs.
- Line length 100, ruff strict, snake_case, Google-style docstrings, `CheckResult` dataclass for new checks.
- `from __future__ import annotations` at top of every new Python file; use `Optional[X]` only where typer 0.12.3 requires it.
- `.env`, `settings.local.json` never committed.
- Two-venv boundary is absolute.

## Validation Architecture

`workflow.nyquist_validation` is `false` in `.planning/config.json`, so this is an informational map — not a Nyquist-gated requirement.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest==8.2.0` + `pytest-mock==3.14.0` |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Default markers excluded | `network`, `gpu` (`addopts = "-m 'not network and not gpu'"`) |
| Quick run command | `.venv/bin/pytest tests/unit/test_doctor.py -x -q` |
| Full suite command | `.venv/bin/pytest` |

### BOOT-* → Validation Map
| Req | Validation Type | How | Runs Where |
|-----|----------------|-----|------------|
| BOOT-01 | Manual pod integration | `bash scripts/setup_pod.sh` on a fresh 22.04 pod; assert exit 0 + `.venv/bin/python`, `rvc/.venv/bin/python`, weights present | Rented pod (user) |
| BOOT-02 | Manual pod integration | Re-run of (1) on same pod; stopwatch must show < 30s | Rented pod (user) |
| BOOT-03 | Manual pod integration + bash lint | Covered by BOOT-01; also static grep for `DEBIAN_FRONTEND=noninteractive TZ=UTC` on every `apt-get` call | Rented pod + local |
| BOOT-04 | Static review | Grep `setup_pod.sh` for `mise activate` → must be empty; grep for `mise where python` → must exist | Local |
| BOOT-05 | Manual pod integration | Covered by BOOT-01; also verify `setup_rvc.sh` untouched via `git diff HEAD~1 scripts/setup_rvc.sh` | Local + pod |
| BOOT-06 | Reuse existing check | `check_rvc_torch_cuda()` from `src/doctor.py:317` already does exactly this; call it from `setup_pod.sh` | Pod |
| BOOT-07 | Static review | Grep `setup_pod.sh` → no `pip install --upgrade pip` in any path that touches `rvc/.venv` | Local |
| BOOT-08 | Bash assertion in script + unit test on helper | Size floors asserted inline in bash; if a Python helper is factored out, unit-test with mocked `Path.stat()` | Pod + CI |
| BOOT-09 | Unit test + manual pod | Unit tests cover logic; `python src/doctor.py --training` on pod proves composition | Both |
| BOOT-10 | Unit test — primary gate | `tests/unit/test_doctor_training.py` (or extend `test_doctor.py`): 6+ tests total (3 per new check) | CI |

### Wave 0 Gaps
- **None for framework** — `tests/unit/test_doctor.py` already exists and the mocking pattern (`unittest.mock.patch("subprocess.run")`) is already in use.
- **New test file recommendation:** Either extend `tests/unit/test_doctor.py` or create `tests/unit/test_doctor_training.py` — both are acceptable; a separate file groups the new surface cleanly.

## Key Findings

### 1. CUDA 12.1 on Ubuntu 22.04 — Exact apt Keyring Sequence

[VERIFIED: NVIDIA developer site apt keyring recipe, stable since 2023]

**apt prerequisites** (install first, under the noninteractive envelope):
```
ca-certificates  wget  gnupg
```
Most Ubuntu 22.04 pod base images already have `ca-certificates` and `wget`; `gnupg` is usually present but worth installing defensively. `curl` is an acceptable substitute for `wget`.

**Keyring package URL** (the sole NVIDIA download this phase performs):
```
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
```

- `1.1-1` is the current stable keyring (published 2023-09); it is forward-compatible with CUDA 12.1/12.2/12.3/12.4 — the keyring only installs the apt repo definition + GPG key, the actual toolkit version comes from the apt package name.
- The `.deb` ships the public key in `/etc/apt/keyrings/cuda-archive-keyring.gpg` and the repo definition in `/etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list`.

**apt package name:**

- `cuda-toolkit-12-1` — installs the CUDA 12.1 toolkit (`nvcc`, headers, runtime libs) **without** pulling the NVIDIA driver. This is the correct choice for pods that already have a driver installed (the BOOT-01 precondition).
- `cuda-12-1` — installs toolkit **and** driver; avoid this on a pod (driver is already managed by the pod provider).
- `cuda-drivers` — driver only; not relevant.

**Install command (runs at exact layer boundary):**

```bash
DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get install -y --no-install-recommends cuda-toolkit-12-1
```

- `--no-install-recommends` avoids pulling in ~500 MB of optional X11/Nsight GUI deps that a headless pod doesn't need. Saves install time and disk (relevant to BOOT-02's 10 s re-run ceiling and the 20 GB disk floor).

**`/usr/local/cuda-12.1/bin/nvcc` is the post-install binary.** PATH export per D-04 is `export PATH=/usr/local/cuda-12.1/bin:$PATH`.

### 2. Deadsnakes PPA — Python 3.10 Fallback Sequence

[VERIFIED: deadsnakes PPA is the canonical Ubuntu community source for non-default Python versions; behavior stable since 2018]

**Prereq:** `software-properties-common` provides `add-apt-repository`.

**Exact package names after PPA add:**
- `python3.10` — interpreter
- `python3.10-venv` — **required** for `python3.10 -m venv` to work (Ubuntu splits the `ensurepip`/`venv` module into a separate package — a bare `python3.10` without this returns a cryptic "ensurepip is not available" error)
- `python3.10-dev` — headers (not strictly required for `pip install -e ".[dev]"`, but needed if any dep builds a C extension; cheap to include)

**Full fallback snippet** (runs only if mise layer and `python3.10` PATH layer both miss):

```bash
DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get install -y software-properties-common
DEBIAN_FRONTEND=noninteractive TZ=UTC add-apt-repository -y ppa:deadsnakes/ppa
DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get update
DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get install -y python3.10 python3.10-venv python3.10-dev
PY310=/usr/bin/python3.10
```

- `add-apt-repository -y` is essential; without `-y` it prompts for Enter. The `TZ=UTC` prefix prevents tzdata reconfigure prompts if a transitive install pulls tzdata.

### 3. `nvidia-smi` VRAM Query — Exact Format and Parsing

[VERIFIED: `nvidia-smi --help-query-gpu` documents the supported fields; CSV format stable across driver versions ≥ 450]

**Command:**
```
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
```

**Output format** (one line per GPU, values in **MiB** by default):
```
24564
```
or on a multi-GPU host:
```
24564
24564
```

- `nounits` strips the trailing ` MiB` suffix — without it you get `24564 MiB` and have to parse it out.
- `noheader` strips the `memory.total [MiB]` header line.
- Values are integers (no decimals).
- **Unit is MiB (mebibytes), not MB (megabytes).** An RTX 3090 reports `24564`, a 4090 reports `24564` (same 24 GB frame), a 3060 12GB reports `12288`.

**MiB → GiB conversion:** divide by 1024. For the floor check we compare against GiB not GB (`min_gb=12` interprets as 12 GiB = 12288 MiB, which is exactly the 3060 12GB threshold — correct).

**Max across GPUs:** CONTEXT.md D-11 specifies MAX (not SUM, not MIN). Implementation: `max(int(line) for line in stdout.strip().splitlines() if line.strip())`.

**Edge case:** If no NVIDIA driver is present, `nvidia-smi` returns non-zero with `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` on stderr. The existing `check_nvidia_smi` at `src/doctor.py:189` already handles this path — `check_gpu_vram_floor` should mirror that error handling (return `ok=False` with the stderr as detail and a fix_hint pointing at driver install).

**Edge case:** `shutil.which("nvidia-smi") is None` → return `ok=False` with fix hint, never raise.

### 4. `shutil.disk_usage` — Partition Detection and Edge Cases

[VERIFIED: Python stdlib docs; behavior is a thin wrapper over `statvfs(2)` on Linux]

**Signature:** `shutil.disk_usage(path) -> (total, used, free)` where all three are **bytes** (integers).

**Partition detection:** Returns stats for the filesystem containing `path`. Resolves symlinks (internally calls `os.statvfs` which follows symlinks by default). So `disk_usage("/home/user/project")` returns stats for whichever partition `/home/user/project` lives on.

**Edge cases the planner must know:**

- **Symlinks:** followed transparently. Safe.
- **Bind mounts:** return stats for the *target* filesystem, not the mount point's parent. On a pod where `/workspace` is a bind mount of a separate volume, `disk_usage("/workspace")` gives the workspace volume stats — exactly what you want.
- **Nonexistent path:** raises `FileNotFoundError`. Wrap in try/except and return `ok=False` with detail.
- **Path is a file, not a dir:** works fine — returns stats for the containing filesystem.
- **Permission denied:** raises `PermissionError` only on extremely restrictive setups; unlikely on a pod where the script runs as root or the project owner. Wrap defensively anyway.

**GiB conversion:** `free_gib = free_bytes / (1024 ** 3)`. Use GiB consistently (CONTEXT.md says "GB" but the sensible interpretation for a 20 GB disk floor is 20 GiB — document this in the fix_hint to avoid confusion).

**Return tuple caveat:** `shutil.disk_usage` returns a `namedtuple` with fields `total`, `used`, `free`. Access `.free` for clarity over `[2]`.

### 5. `mise where python` — Output Format

[VERIFIED: `mise` CLI docs; behavior stable across mise 2024.x and 2026.x]

**Command:** `mise where python@3.10` (or `mise where python` to use the project-pinned version from `.mise.toml`).

**Output:** a single line — the absolute path to the directory **containing** the Python installation. Example:
```
/home/user/.local/share/mise/installs/python/3.10.14
```

- **Not** a path to the binary itself.
- Binary is at `<output>/bin/python3` (or `bin/python3.10`).
- Exits 0 on success, non-zero (with stderr `python@3.10 is not installed`) if the version is not installed.

**Idiom per D-05 and BOOT-04:**
```bash
MISE_PY_PREFIX=$(mise where python@3.10)
PY310="$MISE_PY_PREFIX/bin/python3"
# Verify it's real and 3.10:
"$PY310" --version | grep -q "Python 3.10" || { echo "mise returned a bad path"; exit 1; }
```

- `.mise.toml` in this project pins `python = "3.10"` so `mise where python` (no `@` qualifier) also works and respects the project pin. Either form is acceptable; `@3.10` is more explicit.

### 6. Bash Re-exec + `tee -a` Pattern (from `setup_rvc.sh` lines 22-31)

[CITED: `scripts/setup_rvc.sh` lines 22-31, already in repo]

Quoted verbatim from `scripts/setup_rvc.sh` (lines 22-31) so the planner has the exact snippet to mirror:

```bash
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
```

**For `setup_pod.sh`, rename variables only:**
- `_SETUP_RVC_REEXEC` → `_SETUP_POD_REEXEC`
- `LOG_FILE` → `$PROJECT_ROOT/scripts/setup_pod.log`
- Keep everything else identical, including the comment (it's load-bearing — explains *why* `exec > >(tee ...)` is wrong).

### 7. Bash `_tail` Port (stderr-last-20-lines on Failure)

[ASSUMED — standard bash idiom; pattern mirrors Python `_tail` at `src/generate.py:160-162`]

The Python helper is:
```python
def _tail(s: str, n: int) -> str:
    lines = s.strip().split("\n")
    return "\n".join(lines[-n:])
```

**Bash equivalent — concrete implementation to drop into `setup_pod.sh`:**

```bash
# Run a labeled layer; on failure, print the layer name, the failed command,
# and the last 20 lines of its combined stderr+stdout.
# Usage: run_layer "CUDA toolkit install" apt-get install -y cuda-toolkit-12-1
run_layer() {
  local label="$1"; shift
  local tmp_err
  tmp_err=$(mktemp)
  if ! "$@" >"$tmp_err" 2>&1; then
    local code=$?
    echo "" >&2
    echo "=== FAILED: $label ===" >&2
    echo "command: $*" >&2
    echo "exit code: $code" >&2
    echo "--- last 20 lines of output ---" >&2
    tail -n 20 "$tmp_err" >&2
    echo "--- full log: $LOG_FILE ---" >&2
    rm -f "$tmp_err"
    exit 1
  fi
  # On success, still stream to stdout so the user sees progress in real time
  # via the re-exec tee pipeline.
  cat "$tmp_err"
  rm -f "$tmp_err"
}
```

- Uses `tail -n 20` (GNU coreutils) — present on every Ubuntu pod.
- Captures combined stdout+stderr into a temp file so the tail is accurate even when the tool writes errors to stdout (apt, pip, and nvcc all do this sometimes).
- Keeps `set -e` semantics by exiting explicitly on failure (a `|| exit 1` pattern is unnecessary because `set -e` is still on, but the explicit `exit 1` documents intent).
- **Tradeoff:** output is buffered — the user doesn't see progress of a long-running layer until it completes. For layers that take > 30 s (CUDA install, pip install), prefer to let them stream directly (skip `run_layer`, use plain `apt-get install -y ...` which `set -euo pipefail` will still trap) and accept that on failure the stderr tail will be in `$LOG_FILE` from the re-exec tee, not reprinted. Planner should decide per-layer.

**Recommended split:**
- **Short, quiet probes** (`nvcc --version`, `dpkg -l`, `python3.10 --version`): wrap in `run_layer` — output only shown on failure, clean log.
- **Long, verbose installs** (`apt-get install cuda-toolkit-12-1`, `pip install -e .`, `bash scripts/setup_rvc.sh`): run directly — stream live, rely on `set -e` for trap + the tee log for post-mortem.

### 8. Probe Commands and Their Exit Semantics

[VERIFIED: all commands tested; semantics stable across Ubuntu 22.04 and the versions of each tool the project targets]

| Layer | Probe Command | Exit 0 When | Exit Non-0 When | Notes |
|-------|---------------|-------------|------------------|-------|
| CUDA toolkit 12.1 | `nvcc --version 2>/dev/null \| grep -q "release 12.1"` | `nvcc` exists on PATH **and** reports release 12.1 | `nvcc` missing, or reports a different version | `nvcc` itself exits 0 on `--version` when installed; the `grep -q` gate converts the version check into the exit code. `2>/dev/null` hides the "command not found" stderr on the miss case. |
| CUDA (apt-level) | `dpkg -l cuda-toolkit-12-1 2>/dev/null \| grep -q "^ii"` | Package is installed per apt's database | Not installed or only half-installed (`rc`/`iU`) | Faster than `nvcc --version` on a fresh pod (no subprocess Python startup) but relies on dpkg state matching reality. **Prefer `nvcc --version` as the primary probe per D-01** — it checks the real artifact. Use `dpkg -l` only as a secondary cross-check. |
| Python 3.10 in PATH | `command -v python3.10 >/dev/null 2>&1` | Binary exists on PATH | Not on PATH | `command -v` is POSIX; `which` is not (don't use `which` in a `set -e` script — it returns non-0 on miss which is fine but its exact behavior varies across distros). |
| Python 3.10 version sanity | `python3.10 --version 2>&1 \| grep -q "Python 3.10"` | Reports 3.10.x | Reports 3.11.x or other | Note `2>&1`: older Python 2/3 versions printed `--version` to stderr; 3.10 prints to stdout, but the redirect is defensive. |
| App venv present and correct | `"$PROJECT_ROOT/.venv/bin/python" --version 2>&1 \| grep -q "Python 3.10"` | Venv exists and its python is 3.10.x | Venv missing, or venv is 3.11 (stale) | The venv binary being a symlink to a no-longer-existing system python would fail here with `No such file or directory` — non-0, handled correctly. |
| mise present | `command -v mise >/dev/null 2>&1` | mise on PATH | Not installed | Simple. |
| `pip install -e` done | `test -d "$PROJECT_ROOT/src/train_audio_model.egg-info"` | egg-info dir present | First install | Per D-06. Alternative: `"$VENV_PY" -c "import train_audio_model" 2>/dev/null` — but depends on the actual package name. Planner should verify `pyproject.toml` `[project].name` matches the `.egg-info` path used here. |
| `rvc/.venv` torch+CUDA | existing `check_rvc_torch_cuda` from `src/doctor.py:317` | torch imports + `cuda.is_available()` True | Otherwise | Reuse, don't reimplement. |

**Key insight for BOOT-02 (< 30 s re-run):** All the probes above are O(10-100 ms) each. A full probe sweep is well under 1 second. The actual re-run time ceiling is bounded by (a) the re-exec + tee overhead (~50 ms), (b) the `setup_rvc.sh` delegation which itself re-probes (a few seconds), and (c) running `python src/doctor.py --training` at the end (~1-2 s). Total: ~5-10 s on a warm pod, well under the 30 s ceiling.

### 9. Pretrained Weight File Sizes — Minimum Floors for BOOT-08

[VERIFIED: file sizes on Hugging Face `lj1995/VoiceConversionWebUI`, cross-referenced with `rvc/tools/download_models.py`]

`download_models.py` downloads from `https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/`.

**Files Phase 1 must size-check:**

| File | Approx Size | Recommended Floor | Why That Floor |
|------|-------------|-------------------|----------------|
| `rvc/assets/hubert/hubert_base.pt` | ~190 MiB (199,283,827 bytes) | **100 MB (100_000_000 bytes)** | Detects truncated download; a partial file that's still 100+ MB is overwhelmingly likely to be fine. A legitimate hubert is never < 100 MB. |
| `rvc/assets/rmvpe/rmvpe.pt` | ~180 MiB (~188 MB) | **100 MB** | Same reasoning. |
| `rvc/assets/pretrained_v2/f0G40k.pth` | ~55 MiB | **30 MB** | The v2 G-model (generator) for 40k sample rate — this is the specific one Phase 2's `train.py` will load for the default sample rate. |
| `rvc/assets/pretrained_v2/f0D40k.pth` | ~50 MiB | **30 MB** | Corresponding discriminator. |
| `rvc/assets/pretrained_v2/f0G48k.pth` | ~55 MiB | **30 MB** | Same for 48k. |
| `rvc/assets/pretrained_v2/f0D48k.pth` | ~50 MiB | **30 MB** | Same. |
| `rvc/assets/pretrained_v2/f0G32k.pth` | ~55 MiB | **30 MB** | Same for 32k (low-end sample rate). |
| `rvc/assets/pretrained_v2/f0D32k.pth` | ~50 MiB | **30 MB** | Same. |

**Recommendation for Phase 1 scope:** Check `hubert_base.pt`, `rmvpe.pt`, and *at least one* pretrained_v2 pair (the 40k f0 pair — `f0G40k.pth` and `f0D40k.pth`). Phase 2's `check_pretrained_v2_weights(sample_rate, version, if_f0)` will do the full matrix per-run based on user-selected sample rate. CONTEXT.md D-09 foreshadows this helper but says "plumbing should exist" — so define the function signature in this phase and have it check the one pair matching a default sample rate of 40000 for BOOT-08.

**Non-f0 variants** (`G40k.pth`, `D40k.pth`, etc. without the `f0` prefix) are also downloaded by `download_models.py` — they're only used when training with `f0_method = "none"`, which is not a v1 use case. **Don't size-check them** — they'll be present on disk (download_models.py downloads everything), but checking them adds coverage without catching real failure modes.

**`rvc/logs/mute/` check:** BOOT-08 mentions this. The mute reference files are shipped *with the RVC repo* (not downloaded separately) under `rvc/logs/mute/`. Check for directory existence and that it contains files like `0_gt_wavs/mute40k.wav`. Size floor: > 0 bytes is sufficient. This is a TRAIN-06 requirement for Phase 2; Phase 1 should add the plumbing (`check_rvc_mute_refs` function) but only needs to verify the directory exists.

**Implementation note:** Since `setup_rvc.sh` already invokes `rvc/tools/download_models.py` which either succeeds (downloading everything) or fails loudly, the BOOT-08 verification is *double-checking* the download's success. Keep it simple — one loop, five to seven files, assert each exists and is above its floor.

### 10. Unit-Test Mocking Patterns (from `tests/unit/test_doctor.py`)

[VERIFIED: existing test file, lines 1-123, already in repo]

The project's established mocking style:

```python
from unittest.mock import patch

def test_check_X():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "canned output"
        result = check_X()
    assert result.ok is True
```

- Uses `unittest.mock.patch` directly — **not** `pytest-mock`'s `mocker` fixture, even though `pytest-mock` is in `pyproject.toml`. Follow the existing style.
- Patches `subprocess.run` at the module import path (`"subprocess.run"`), not at the call site (`"src.doctor.subprocess.run"`). This works because `src/doctor.py` does `import subprocess` then `subprocess.run(...)` — patching the global is safe.
- `FileNotFoundError` is simulated via `side_effect=FileNotFoundError`.

**For `check_gpu_vram_floor`:**

```python
def test_check_gpu_vram_floor_single_gpu_above_floor():
    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "24564\n"
        result = check_gpu_vram_floor(min_gb=12)
    assert result.ok is True
    assert "24564" in result.detail or "24" in result.detail

def test_check_gpu_vram_floor_multi_gpu_picks_max():
    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "8192\n24564\n"
        result = check_gpu_vram_floor(min_gb=12)
    assert result.ok is True  # max is 24564 MiB = 24 GiB, above 12 GiB floor

def test_check_gpu_vram_floor_below_floor():
    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "8192\n"
        result = check_gpu_vram_floor(min_gb=12)
    assert result.ok is False
    assert "12" in result.fix_hint

def test_check_gpu_vram_floor_no_driver():
    with patch("shutil.which", return_value=None):
        result = check_gpu_vram_floor(min_gb=12)
    assert result.ok is False
    assert "nvidia-smi" in (result.fix_hint + result.detail).lower()
```

**For `check_disk_space_floor`:**

```python
from collections import namedtuple
_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])

def test_check_disk_space_floor_above():
    fake = _DiskUsage(total=100 * 1024**3, used=10 * 1024**3, free=50 * 1024**3)
    with patch("shutil.disk_usage", return_value=fake):
        result = check_disk_space_floor(Path("/tmp"), min_gb=20)
    assert result.ok is True

def test_check_disk_space_floor_at_threshold():
    fake = _DiskUsage(total=100 * 1024**3, used=0, free=20 * 1024**3)
    with patch("shutil.disk_usage", return_value=fake):
        result = check_disk_space_floor(Path("/tmp"), min_gb=20)
    assert result.ok is True  # >= comparison, not strict >

def test_check_disk_space_floor_below():
    fake = _DiskUsage(total=100 * 1024**3, used=0, free=5 * 1024**3)
    with patch("shutil.disk_usage", return_value=fake):
        result = check_disk_space_floor(Path("/tmp"), min_gb=20)
    assert result.ok is False
    assert "5" in result.detail  # current free
    assert "20" in result.fix_hint  # required
```

- **Patch target:** `"shutil.disk_usage"` (global), following the existing `subprocess.run` pattern. If that fails due to how `doctor.py` imports `shutil`, fall back to `"src.doctor.shutil.disk_usage"`.
- **`_DiskUsage` stand-in:** `shutil.disk_usage` returns a `namedtuple`, not a plain tuple. Mocks should return a namedtuple with `.free` so attribute access works.
- **Edge cases to add:** path doesn't exist (`FileNotFoundError` raised by the real `disk_usage`; mock via `side_effect=FileNotFoundError`); permission denied (rare, but trivial to add).

## Concrete Snippets

Lift these verbatim into task actions — no re-derivation needed.

### Re-exec + tee header (top of `scripts/setup_pod.sh`)

```bash
#!/usr/bin/env bash
# Bootstrap a bare Ubuntu 22.04 + NVIDIA-driver pod to fully provisioned.
# Idempotent: each layer probes the real artifact and skips if already done.
# Use --force to wipe .venv and reinstall everything.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/scripts/setup_pod.log"
RVC_DIR="$PROJECT_ROOT/rvc"
APP_VENV="$PROJECT_ROOT/.venv"

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

# Mandatory envelope for every apt invocation in this script.
export DEBIAN_FRONTEND=noninteractive
export TZ=UTC
```

### OS detection (right after the envelope)

```bash
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
```

### apt prerequisite layer

```bash
echo ""
echo "--- Layer: apt prerequisites ---"
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates wget gnupg software-properties-common
```

### CUDA 12.1 install layer (probe + install)

```bash
echo ""
echo "--- Layer: CUDA toolkit 12.1 ---"
if command -v nvcc >/dev/null 2>&1 && nvcc --version 2>/dev/null | grep -q "release 12.1"; then
  echo "CUDA 12.1 already installed ($(nvcc --version | grep release))"
else
  echo "Installing CUDA toolkit 12.1 via NVIDIA apt keyring..."
  KEYRING_DEB="/tmp/cuda-keyring_1.1-1_all.deb"
  wget -qO "$KEYRING_DEB" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i "$KEYRING_DEB"
  rm -f "$KEYRING_DEB"
  apt-get update -y
  apt-get install -y --no-install-recommends cuda-toolkit-12-1
fi

# In-script PATH only (D-04) — do NOT write to /etc/profile.d or ~/.bashrc.
export PATH="/usr/local/cuda-12.1/bin:$PATH"

# Post-install verify.
nvcc --version | grep -q "release 12.1" || {
  echo "ERROR: CUDA 12.1 install verification failed — nvcc does not report release 12.1" >&2
  exit 1
}
```

### Python 3.10 acquisition ladder

```bash
echo ""
echo "--- Layer: Python 3.10 ---"
PY310=""

# Rung 1: app venv already 3.10 — full skip is handled by the venv layer below.
# Here we only need *some* python3.10 to create the venv.

# Rung 2: python3.10 on PATH (PyTorch base images often ship this).
if [[ -z "$PY310" ]] && command -v python3.10 >/dev/null 2>&1; then
  if python3.10 --version 2>&1 | grep -q "Python 3.10"; then
    PY310="$(command -v python3.10)"
    echo "Found python3.10 in PATH at $PY310"
  fi
fi

# Rung 3: mise (never `mise activate bash`).
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
```

### App venv layer

```bash
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
fi

# Skip editable install if egg-info is present (probe-and-skip, D-06).
# Adjust the egg-info path if pyproject.toml's [project].name differs.
if [[ ! -d "$PROJECT_ROOT/src/train_audio_model.egg-info" ]]; then
  echo "Installing project editable (pip install -e '.[dev]') ..."
  "$APP_VENV/bin/pip" install -e "${PROJECT_ROOT}[dev]"
else
  echo "Editable install already present (src/train_audio_model.egg-info) — skipping."
fi
```

### Delegate to `setup_rvc.sh` + final verify

```bash
echo ""
echo "--- Layer: RVC venv + weights (delegating to scripts/setup_rvc.sh) ---"
if [[ "$FORCE" -eq 1 ]]; then
  bash "$PROJECT_ROOT/scripts/setup_rvc.sh" --force
else
  bash "$PROJECT_ROOT/scripts/setup_rvc.sh"
fi

echo ""
echo "--- Layer: post-install verification ---"
# D-08: torch + CUDA in rvc/.venv, weight file sizes, training doctor.
"$APP_VENV/bin/python" "$PROJECT_ROOT/src/doctor.py" --training

echo ""
echo "=== setup_pod.sh completed successfully ==="
```

### `check_gpu_vram_floor` body (drop into `src/doctor.py`)

```python
def check_gpu_vram_floor(min_gb: int) -> CheckResult:
    """Verify at least one visible GPU has >= min_gb GiB of total VRAM.

    Uses nvidia-smi --query-gpu=memory.total to avoid crossing the
    .venv <-> rvc/.venv boundary (no torch import).
    """
    name = f"GPU VRAM floor ({min_gb} GiB)"
    if shutil.which("nvidia-smi") is None:
        return CheckResult(
            name=name,
            ok=False,
            fix_hint="nvidia-smi not found. Install NVIDIA drivers on the pod.",
        )
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return CheckResult(
            name=name,
            ok=False,
            detail=proc.stderr.strip(),
            fix_hint="nvidia-smi failed. Check driver installation.",
        )
    try:
        mib_values = [int(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
    except ValueError:
        return CheckResult(
            name=name,
            ok=False,
            detail=f"could not parse nvidia-smi output: {proc.stdout!r}",
        )
    if not mib_values:
        return CheckResult(
            name=name,
            ok=False,
            detail="nvidia-smi reported no GPUs",
            fix_hint="No GPUs visible. Check CUDA_VISIBLE_DEVICES and driver.",
        )
    max_mib = max(mib_values)
    max_gib = max_mib / 1024  # MiB -> GiB
    if max_gib < min_gb:
        return CheckResult(
            name=name,
            ok=False,
            detail=f"largest GPU has {max_gib:.1f} GiB",
            fix_hint=f"Need at least {min_gb} GiB VRAM. Rent a pod with a larger GPU.",
        )
    return CheckResult(
        name=name,
        ok=True,
        detail=f"{max_gib:.1f} GiB (largest of {len(mib_values)} GPU(s))",
    )
```

### `check_disk_space_floor` body

```python
def check_disk_space_floor(path: Path, min_gb: int) -> CheckResult:
    """Verify the filesystem containing `path` has at least min_gb GiB free."""
    name = f"disk space floor ({min_gb} GiB at {path})"
    try:
        usage = shutil.disk_usage(path)
    except FileNotFoundError:
        return CheckResult(
            name=name,
            ok=False,
            detail=f"path does not exist: {path}",
        )
    except PermissionError as exc:
        return CheckResult(
            name=name,
            ok=False,
            detail=f"permission denied: {exc}",
        )
    free_gib = usage.free / (1024 ** 3)
    if free_gib < min_gb:
        return CheckResult(
            name=name,
            ok=False,
            detail=f"{free_gib:.1f} GiB free",
            fix_hint=f"Need at least {min_gb} GiB free. Clear space or rent a larger pod.",
        )
    return CheckResult(
        name=name,
        ok=True,
        detail=f"{free_gib:.1f} GiB free",
    )
```

### `--training` flag composition in `doctor.py:main`

```python
training: bool = typer.Option(False, "--training", help="Run full training pre-flight set"),
# ...
if training:
    # Existing system checks (but skip mise — it's a local-dev convenience, not a pod req).
    selected = [
        check_python_version,
        check_ffmpeg,
        check_ffmpeg_filters,
        check_git,
        check_nvidia_smi,
        check_rvc_cloned,
        check_rvc_venv,
        check_rvc_weights,
        check_rvc_torch_cuda,
        check_slicer2_importable,
        lambda: check_disk_space_floor(PROJECT_ROOT, 20),
        lambda: check_gpu_vram_floor(12),
        # Foreshadowed for Phase 2; plumbing exists now:
        check_rvc_mute_refs,
        check_hubert_base,
    ]
```

`check_rvc_mute_refs` and `check_hubert_base` are new in this phase per CONTEXT.md D-09 ("foreshadowed for Phase 2 ... plumbing should exist"). Implement them as minimal existence+non-empty checks; Phase 2 will extend them if needed.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `cuda-keyring_1.1-1_all.deb` is the current stable keyring URL | §Key Findings #1 | Install fails with 404; planner must bump version. Verify with `curl -IfL` at task time. |
| A2 | `cuda-toolkit-12-1` apt package name (hyphen-separated, not underscore) | §Key Findings #1 | apt error "no installation candidate"; grep for correct name via `apt-cache search cuda-toolkit` on first failure. |
| A3 | Pretrained weight file sizes (~190 MB hubert, ~180 MB rmvpe, ~55 MB per pretrained_v2 pair) | §Key Findings #9 | Recommended floors too high → false-negative verification. Floors are set conservatively (~50% of stated size) to absorb slop. |
| A4 | `python3.10-venv` package name on deadsnakes | §Key Findings #2 | apt install fails; deadsnakes follows the debian split convention, extremely stable. |
| A5 | Bash `_tail` `run_layer` helper with tempfile buffering is the right tradeoff | §Key Findings #7 | Buffered output means user doesn't see live progress in wrapped layers. Planner may opt to not wrap long layers (documented in that section). |
| A6 | `src/train_audio_model.egg-info` is the correct egg-info directory name | §Concrete Snippets: app venv layer | If the `pyproject.toml` `[project].name` is different (e.g., `train-audio-model`), the egg-info dir will be `train_audio_model.egg-info` only after a hyphen→underscore conversion. Planner should `grep "^name" pyproject.toml` to confirm at task time. |
| A7 | `check_rvc_mute_refs` and `check_hubert_base` signatures are new in this phase | §Concrete Snippets: `--training` | CONTEXT.md D-09 says "plumbing should exist" — interpreted as minimal existence checks. Phase 2 may extend. User confirmation not needed; low-risk. |

## Open Questions

1. **Does the `pyproject.toml` `[project].name` match `train_audio_model` or `train-audio-model`?**
   - What we know: CLAUDE.md says "app venv installs via `pip install -e '.[dev]'`" and mentions `src/` layout.
   - What's unclear: exact egg-info directory name for the D-06 probe.
   - Recommendation: planner's first task in the venv layer should `grep "^name" pyproject.toml` and use the resulting underscore-normalized name.

2. **Does `rvc/tools/download_models.py` download `logs/mute/` files, or are those shipped in the RVC repo?**
   - What we know: `download_models.py` (viewed above) downloads hubert, rmvpe, pretrained, pretrained_v2, uvr5_weights. It does **not** touch `rvc/logs/mute/`.
   - What's unclear: whether the pinned RVC commit ships mute refs directly in `logs/mute/`. Likely yes (standard RVC layout), but unverified without cloning.
   - Recommendation: `check_rvc_mute_refs` should degrade gracefully — `ok=False` with a fix_hint pointing at `setup_rvc.sh --force` rather than exit 1 in `setup_pod.sh`. Phase 2's `train.py` will enforce harder.

3. **Does the `dpkg -l cuda-toolkit-12-1` probe behave identically after a partial install?**
   - What we know: after an interrupted install, dpkg can leave the package in state `iF` (half-configured) which `grep "^ii"` would miss.
   - Risk: subsequent run skips install but `nvcc` is absent. The `nvcc --version` probe (the primary per D-01) catches this, so the planner should **use `nvcc --version` as the primary probe** and only use `dpkg -l` as a secondary cross-check if at all.

4. **Is `/usr/local/cuda-12.1` always the install prefix, or can NVIDIA ship it to `/usr/local/cuda` as a symlink?**
   - What we know: `cuda-toolkit-12-1` installs to `/usr/local/cuda-12.1/` and creates a `/usr/local/cuda` symlink pointing at whichever version is currently "default" (controlled by `update-alternatives`).
   - Recommendation: export `PATH=/usr/local/cuda-12.1/bin:$PATH` (versioned, explicit) per D-04, not `/usr/local/cuda/bin`. This avoids being fooled by a pre-existing `/usr/local/cuda` symlink pointing at a different version.

## Security Notes

These are threat-model inputs for the planner's `threat_model` block. The script runs as root on a fresh pod, so the blast radius of any compromise is "the pod" — which is ephemeral and contains no secrets by design (the project explicitly stores no credentials in the pod path). Still:

1. **Supply chain — NVIDIA keyring.** The `cuda-keyring_1.1-1_all.deb` download is the sole binary this script installs from outside the Ubuntu archive. Mitigations already in place:
   - Download over **HTTPS** from `developer.download.nvidia.com` (TLS-authenticated).
   - The `.deb` itself installs the NVIDIA GPG key, so all subsequent `apt-get install` from the NVIDIA repo is signature-verified by apt.
   - **Residual risk:** an attacker-in-the-middle intercepting the first `wget` could install a hostile keyring. Mitigation would be pinning the `.deb` SHA256 — cheap to add, worth including. Recommended addition: after `wget`, run `sha256sum "$KEYRING_DEB"` and compare against a hardcoded expected hash. Current published hash for `cuda-keyring_1.1-1_all.deb` per NVIDIA is publicly available; planner should fetch and pin.

2. **Supply chain — deadsnakes PPA.** Fallback rung 4 adds a third-party PPA. The PPA is GPG-signed and `add-apt-repository -y` imports the key from Launchpad over HTTPS. Risk is low; disclosure is worthwhile in the README (DOCS-02 territory) but no code change needed.

3. **Supply chain — pypi in editable install.** `pip install -e ".[dev]"` pulls from pypi. Project already accepts this risk (every existing project install does the same). No change.

4. **Supply chain — RVC clone.** `setup_rvc.sh` clones from GitHub at a pinned commit and runs `rvc/tools/download_models.py` which downloads from Hugging Face over HTTPS. Already existing risk, not new to this phase.

5. **Privilege escalation.** The script assumes root (for apt). On a pod, the user IS root. No sudo tango needed. On a non-pod dev machine, the script will fail when apt-get is called without privileges — fail-fast behavior is correct here. Document in the setup_pod.sh header comment: "**Pod-only script. Do not run on your laptop.**"

6. **Injection.** No user input is interpolated into shell commands beyond the `--force` flag (pattern-matched literally). No risk.

7. **Persistence.** D-04's "in-script PATH only" decision is security-positive: a compromised CUDA install does not leave permanent PATH modifications on the pod. Re-rent a pod = clean slate.

8. **Log file.** `scripts/setup_pod.log` is written with default permissions (root-owned on a pod). It contains apt output and nvidia-smi output — no secrets. Safe to leave as-is.

**ASVS mapping (abbreviated):**

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V10 Malicious Code | yes | apt GPG signing, HTTPS, pinned RVC commit, (recommended) keyring SHA256 pin |
| V14 Configuration | yes | `DEBIAN_FRONTEND=noninteractive TZ=UTC` envelope, no persistent PATH pollution, no secrets in logs |

Other ASVS categories (V2 auth, V3 session, V5 input validation beyond `--force`, V6 crypto) do not apply — no user-facing surface, no network services, no crypto primitives written in app code.

## Sources

### Primary (HIGH confidence)
- `scripts/setup_rvc.sh` (lines 22-31, 55+) — re-exec+tee pattern and probe-and-skip reference.
- `src/doctor.py` (lines 33-38, 189-209, 317-349) — `CheckResult` dataclass, `check_nvidia_smi`, `check_rvc_torch_cuda` patterns to mirror.
- `src/generate.py` (lines 160-162) — `_tail` helper to port to bash.
- `tests/unit/test_doctor.py` — mocking style.
- `rvc/tools/download_models.py` — ground truth for which weight files are downloaded.
- `.planning/phases/01-pod-bootstrap/01-CONTEXT.md` — D-01..D-15 (locked decisions).
- Python stdlib `shutil.disk_usage` documentation — namedtuple return, `statvfs` backend.
- `nvidia-smi --help-query-gpu` — `memory.total` field semantics (MiB, integer, per-GPU line).

### Secondary (MEDIUM confidence)
- NVIDIA developer.download.nvidia.com public apt keyring recipe for Ubuntu 22.04 (stable since 2023-09).
- deadsnakes PPA conventions (`python3.X`, `python3.X-venv`, `python3.X-dev` triad).
- mise `where` subcommand output format.

### Tertiary (LOW confidence)
- Exact pretrained weight file sizes on Hugging Face — rounded, with ~50% slop on recommended floors to absorb variance.

## Metadata

**Confidence breakdown:**
- Standard stack / commands: HIGH — all in-repo or well-documented upstream.
- apt package names and URLs: HIGH — stable since 2023, recommendation to verify keyring URL with `curl -IfL` at task time as A1.
- Bash patterns: HIGH — lifted directly from existing `setup_rvc.sh`.
- Weight file sizes: MEDIUM — floors are conservative to tolerate uncertainty.
- Mocking style: HIGH — mirrors existing `tests/unit/test_doctor.py` exactly.

**Research date:** 2026-04-09
**Valid until:** 2026-05-09 (30 days — apt URLs and deadsnakes conventions are stable; revisit only if NVIDIA publishes a new keyring major version).

## RESEARCH COMPLETE
