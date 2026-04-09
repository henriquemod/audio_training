---
phase: 01-pod-bootstrap
plan: 02
name: setup-pod-script
type: execute
wave: 2
depends_on: [01-01-doctor-training-checks]
files_modified:
  - scripts/setup_pod.sh
requirements: [BOOT-01, BOOT-02, BOOT-03, BOOT-04, BOOT-05, BOOT-06, BOOT-07, BOOT-08]
autonomous: false
tags: [bootstrap, bash, cuda, apt, pod, mise, python]

must_haves:
  truths:
    - "`bash scripts/setup_pod.sh` on a clean Ubuntu 22.04 + NVIDIA-driver pod exits 0 non-interactively, leaving `.venv/bin/python` (3.10), `rvc/.venv/bin/python`, and pretrained weights"
    - "Re-running `bash scripts/setup_pod.sh` on an already-provisioned pod exits 0 in well under 30 seconds without mutating any venv"
    - "Every `apt-get` invocation in the script is prefixed by `DEBIAN_FRONTEND=noninteractive TZ=UTC` (via exported envelope or inline)"
    - "The script uses `nvcc --version | grep -q 'release 12.1'` as the primary CUDA probe (strict 12.1 match, not any 12.x)"
    - "The script uses `$(mise where python@3.10)/bin/python3` for the mise rung, never `mise activate bash`"
    - "The script delegates `rvc/.venv` creation to `bash scripts/setup_rvc.sh` without modifying it"
    - "The script's final layer runs `.venv/bin/python src/doctor.py --training` (using the flag added by Plan 01-01)"
    - "The script has `set -euo pipefail` at the top and uses the re-exec + `tee -a setup_pod.log` pattern (`PIPESTATUS[0]` exit propagation)"
    - "`scripts/setup_rvc.sh` is byte-for-byte unchanged after this plan"
  artifacts:
    - path: "scripts/setup_pod.sh"
      provides: "Idempotent pod bootstrap (CUDA→Python→app venv→RVC→verify)"
      min_lines: 180
      contains: "_SETUP_POD_REEXEC"
    - path: "scripts/setup_pod.sh"
      provides: "Executable permission bit set"
      contains: "#!/usr/bin/env bash"
    - path: "scripts/setup_pod.log"
      provides: "Populated on first run via tee (gitignored — do not commit)"
  key_links:
    - from: "scripts/setup_pod.sh CUDA layer"
      to: "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
      via: "wget -qO"
      pattern: "cuda-keyring_1\\.1-1_all\\.deb"
    - from: "scripts/setup_pod.sh RVC delegation layer"
      to: "scripts/setup_rvc.sh"
      via: "bash subprocess"
      pattern: "bash .*scripts/setup_rvc\\.sh"
    - from: "scripts/setup_pod.sh final verification"
      to: "src/doctor.py --training"
      via: ".venv/bin/python"
      pattern: "src/doctor\\.py --training"
---

<objective>
Create `scripts/setup_pod.sh` — a single non-interactive bash script that takes a bare Ubuntu 22.04 + NVIDIA-driver pod to fully provisioned in one invocation. Idempotent via intrinsic probing (nvcc, python --version, venv presence, weight file sizes); re-runs on a healthy pod complete in ~10 seconds. Delegates RVC venv + weight download to existing `scripts/setup_rvc.sh` unchanged. Final verification runs `.venv/bin/python src/doctor.py --training` (the flag added by Plan 01-01).

Purpose: BOOT-01..BOOT-08. This is the user-facing pod bootstrap primitive. Everything downstream (Phase 2 train.py, Phase 4 train.sh) assumes this script has run successfully.

Output: `scripts/setup_pod.sh` (executable) + a test run of the script on a real pod (manual gate — see the checkpoint task).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/01-pod-bootstrap/01-CONTEXT.md
@.planning/phases/01-pod-bootstrap/01-RESEARCH.md
@scripts/setup_rvc.sh
@CLAUDE.md

<interfaces>
The consumed interface from Plan 01-01 is `python src/doctor.py --training` — a no-argument CLI flag that exits 0 on a ready pod and 1 on any missing check. Plan 01-01 MUST be merged before Plan 02 task 1 runs.

The delegated interface to `scripts/setup_rvc.sh` (already in repo, DO NOT MODIFY):
- Invocation: `bash scripts/setup_rvc.sh` or `bash scripts/setup_rvc.sh --force`
- Exit 0 on success, non-zero on failure
- Idempotent itself (probe-and-skip for clone/venv/weights)
- Writes its own log to `scripts/setup_rvc.log` via its own re-exec+tee
- Preserves `pip<24.1` pin in `rvc/.venv` (lines 90-91 of setup_rvc.sh)
- Calls `rvc/tools/download_models.py` which downloads hubert_base.pt, rmvpe.pt, pretrained_v2/*
- Final verifies torch+CUDA inside rvc/.venv

The `scripts/setup_rvc.sh` re-exec+tee pattern to MIRROR (lines 22-31):

```bash
mkdir -p "$(dirname "$LOG_FILE")"
if [[ -z "${_SETUP_RVC_REEXEC:-}" ]]; then
  export _SETUP_RVC_REEXEC=1
  set -o pipefail
  "$0" "$@" 2>&1 | tee -a "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi
```

Rename `_SETUP_RVC_REEXEC` → `_SETUP_POD_REEXEC` and `LOG_FILE` → `$PROJECT_ROOT/scripts/setup_pod.log`. Everything else identical, including the "deliberately do NOT use `exec > >(tee ...)`" comment (it is load-bearing and explains why process substitution is wrong).

Project `[project].name` in `pyproject.toml` is `train-audio-model` → the egg-info directory for the D-06 probe is `src/train_audio_model.egg-info` (hyphen→underscore normalization, standard setuptools behavior).
</interfaces>

Absolute paths:
- `/home/henrique/Development/train_audio_model/scripts/setup_pod.sh` (will create)
- `/home/henrique/Development/train_audio_model/scripts/setup_rvc.sh` (delegated, do not modify)
- `/home/henrique/Development/train_audio_model/.planning/phases/01-pod-bootstrap/01-RESEARCH.md` §"Concrete Snippets" (lines 457-653) — verbatim bash to lift
- `/home/henrique/Development/train_audio_model/.planning/phases/01-pod-bootstrap/01-CONTEXT.md` §Decisions D-01..D-08, D-12, D-13, D-14
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create scripts/setup_pod.sh end-to-end</name>
  <files>scripts/setup_pod.sh</files>

  <read_first>
    - scripts/setup_rvc.sh (FULL file — lines 22-31 are the re-exec+tee pattern to mirror byte-for-byte with variable renames; lines 55-80 are the probe-and-skip reference implementation)
    - .planning/phases/01-pod-bootstrap/01-RESEARCH.md §"Concrete Snippets" lines 457-653 (all of it — lift bash verbatim into the script, only adjusting layer ordering and the `run_layer`/`tail-on-failure` choice per §Key Findings #7)
    - .planning/phases/01-pod-bootstrap/01-CONTEXT.md §Decisions D-01 through D-14
    - .planning/phases/01-pod-bootstrap/01-RESEARCH.md §"Key Findings #1" (CUDA 12.1 apt keyring URL + package name) and §"Key Findings #2" (deadsnakes PPA fallback sequence)
    - pyproject.toml (verify `[project].name` — should be `train-audio-model`, meaning egg-info dir is `src/train_audio_model.egg-info`)
  </read_first>

  <action>
Create `/home/henrique/Development/train_audio_model/scripts/setup_pod.sh` as a single bash file. **Lift the concrete bash snippets from 01-RESEARCH.md §"Concrete Snippets" verbatim** — that section contains the fully-composed script with only a small number of gluing decisions left to the implementer.

**Script structure (enforce this layer order — it's the hard dependency chain from D-12):**

1. **Shebang + header comment** (`#!/usr/bin/env bash`). Header comment must include:
   - "Bootstrap a bare Ubuntu 22.04 + NVIDIA-driver pod to fully provisioned."
   - "Idempotent: each layer probes the real artifact and skips if already done."
   - "Use --force to wipe .venv and reinstall everything."
   - "**POD-ONLY script. Requires root (apt). Do not run on a laptop.**" (per research §Security Notes item 5)

2. **`set -euo pipefail`** (fail-fast, no rollback — D-13).

3. **Variables block:**
   ```bash
   PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
   LOG_FILE="$PROJECT_ROOT/scripts/setup_pod.log"
   RVC_DIR="$PROJECT_ROOT/rvc"
   APP_VENV="$PROJECT_ROOT/.venv"
   APP_EGG_INFO="$PROJECT_ROOT/src/train_audio_model.egg-info"
   CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
   ```

4. **`--force` flag parse** (mirror `setup_rvc.sh` lines 17-20 — only accepts bare `--force` as `$1`).

5. **Re-exec + `tee -a` logging block** — lift verbatim from `setup_rvc.sh` lines 22-31, rename `_SETUP_RVC_REEXEC` → `_SETUP_POD_REEXEC`, rename `LOG_FILE` as set above. KEEP the multi-line comment explaining why `exec > >(tee ...)` is wrong — it is load-bearing. This implements D-14.

6. **Startup banner:** `echo "=== setup_pod.sh started at $(date) ==="`.

7. **Noninteractive apt envelope** (exported once, covers every subsequent apt-get call — implements D-02/D-03 envelope requirement):
   ```bash
   export DEBIAN_FRONTEND=noninteractive
   export TZ=UTC
   ```

8. **OS detection layer** — lift verbatim from 01-RESEARCH.md §"OS detection". Sources `/etc/os-release`, branches on `VERSION_ID`. `22.04` → proceed. `24.04` → loud warning, proceed anyway (D-03). Anything else → exit 1 with a clear fix hint. If `/etc/os-release` is unreadable → exit 1.

9. **apt prerequisites layer** — lift verbatim from 01-RESEARCH.md §"apt prerequisite layer":
   ```bash
   apt-get update -y
   apt-get install -y --no-install-recommends ca-certificates wget gnupg software-properties-common
   ```

10. **CUDA 12.1 layer** — lift verbatim from 01-RESEARCH.md §"CUDA 12.1 install layer (probe + install)". Key points:
    - Primary probe: `command -v nvcc >/dev/null 2>&1 && nvcc --version 2>/dev/null | grep -q "release 12.1"` (D-01: strict 12.1 match).
    - On miss: `wget -qO /tmp/cuda-keyring_1.1-1_all.deb "$CUDA_KEYRING_URL"`, `dpkg -i`, `rm -f`, `apt-get update -y`, `apt-get install -y --no-install-recommends cuda-toolkit-12-1` (implements D-02 and BOOT-03; `--no-install-recommends` saves ~500 MB of X11/Nsight GUI deps per research §Key Findings #1).
    - Post-install: `export PATH="/usr/local/cuda-12.1/bin:$PATH"` — IN-SCRIPT ONLY (D-04). Do NOT write to `/etc/profile.d/` or `~/.bashrc`.
    - Post-install verify: `nvcc --version | grep -q "release 12.1" || { echo ...; exit 1; }`.

11. **Python 3.10 acquisition ladder** — lift verbatim from 01-RESEARCH.md §"Python 3.10 acquisition ladder". Four rungs in priority order (D-05):
    - Rung 1 (skipped here — handled by the app venv layer below).
    - Rung 2: `command -v python3.10 >/dev/null 2>&1` and `python3.10 --version | grep -q "Python 3.10"` → `PY310="$(command -v python3.10)"`.
    - Rung 3: `command -v mise >/dev/null 2>&1` → `mise install python@3.10`, `MISE_PY_PREFIX="$(mise where python@3.10)"`, `PY310="$MISE_PY_PREFIX/bin/python3"`. NEVER `mise activate bash` (STATE.md pitfall, D-05).
    - Rung 4 (deadsnakes fallback): `add-apt-repository -y ppa:deadsnakes/ppa`, `apt-get update -y`, `apt-get install -y python3.10 python3.10-venv python3.10-dev`, `PY310="/usr/bin/python3.10"`.
    - Post-ladder assert: `"$PY310" --version | grep -q "Python 3.10" || { echo ERROR; exit 1; }`.

12. **App venv layer** — lift verbatim from 01-RESEARCH.md §"App venv layer". Key points:
    - If `--force` and `.venv` exists → `rm -rf "$APP_VENV"`.
    - Probe: `[[ ! -x "$APP_VENV/bin/python" ]] || ! "$APP_VENV/bin/python" --version 2>&1 | grep -q "Python 3.10"` → create: `"$PY310" -m venv "$APP_VENV"`.
    - Editable install probe: `[[ ! -d "$APP_EGG_INFO" ]]` → `"$APP_VENV/bin/pip" install -e "${PROJECT_ROOT}[dev]"`. Else echo "already installed — skipping" (D-06 probe-and-skip). CONFIRMED: `pyproject.toml` has `name = "train-audio-model"` → egg-info dir is `src/train_audio_model.egg-info`.
    - Do NOT upgrade pip in `.venv` anywhere in this layer. The `pip<24.1` pin in `rvc/.venv` is a DIFFERENT venv, managed by `setup_rvc.sh`, and must not be touched here (BOOT-07 compliance = "don't interact with rvc/.venv pip at all in this script").

13. **RVC delegation layer** — lift verbatim from 01-RESEARCH.md §"Delegate to `setup_rvc.sh` + final verify":
    ```bash
    if [[ "$FORCE" -eq 1 ]]; then
      bash "$PROJECT_ROOT/scripts/setup_rvc.sh" --force
    else
      bash "$PROJECT_ROOT/scripts/setup_rvc.sh"
    fi
    ```
    This implements D-07 (delegate unchanged) and transitively BOOT-05, BOOT-06, BOOT-07, BOOT-08 (setup_rvc.sh already downloads weights, verifies torch+CUDA, and preserves the pip pin).

14. **Weight size floor layer** — BOOT-08 double-check (setup_rvc.sh already downloaded them, this verifies integrity-lite). Check the files and sizes from 01-RESEARCH.md §"Key Findings #9":
    ```bash
    echo ""
    echo "--- Layer: weight file size floors ---"
    _check_size() {
      local path="$1"; local min_bytes="$2"
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
    ```
    Four files (hubert 100 MB floor, rmvpe 100 MB floor, f0G40k 30 MB floor, f0D40k 30 MB floor — per research §Key Findings #9). Do not size-check non-f0 variants (not used in v1). Do not size-check all 6 f0 pairs (phase 2's `check_pretrained_v2_weights` handles per-sample-rate matrix).

15. **Final verification layer — INVOKES PLAN 01-01 DELIVERABLE:**
    ```bash
    echo ""
    echo "--- Layer: post-install verification (doctor --training) ---"
    "$APP_VENV/bin/python" "$PROJECT_ROOT/src/doctor.py" --training
    ```
    This is the crucial hand-off point: the `--training` flag is added in Plan 01-01 Task 2. If Plan 01-01 hasn't merged, this script cannot be tested end-to-end.

16. **Success banner:**
    ```bash
    echo ""
    echo "=== setup_pod.sh completed successfully ==="
    echo ""
    echo "Next steps:"
    echo "  1. Upload or pull training audio into dataset/raw/"
    echo "  2. Run: .venv/bin/python src/train.py  (Phase 2)"
    ```

**Post-create:** `chmod +x scripts/setup_pod.sh` so the script is executable.

**Line-length guidance:** bash has no hard limit, but keep lines readable. Long apt commands may be backslash-continued as in `setup_rvc.sh`.

**Failure mode (D-13, fail-fast):** Rely on `set -euo pipefail` for most failure propagation. For the weight-size layer, use explicit `exit 1` with messages to stderr. The re-exec + tee captures everything into `setup_pod.log` for post-mortem. No rollback. No marker files (D-12 intrinsic-probe-only). No `run_layer` wrapper — per research §Key Findings #7 the tradeoff favors letting long layers stream live, so use direct invocations throughout.

**Do NOT:**
- Modify `scripts/setup_rvc.sh` in any way (D-07, BOOT-05).
- Touch `rvc/.venv`'s pip directly (BOOT-07).
- Write to `/etc/profile.d/` or `~/.bashrc` (D-04).
- Use `mise activate bash` (STATE.md pitfall).
- Use `shell=True` semantics — this is bash, but apply the same discipline: no `eval "$foo"`, no shell interpolation into command strings from user input (the only user input is `--force`).
- Use marker files for idempotency (D-12).
- Add `sudo` prefixes — script assumes root on the pod (per research §Security Notes #5).
  </action>

  <verify>
    <automated>cd /home/henrique/Development/train_audio_model &amp;&amp; bash -n scripts/setup_pod.sh &amp;&amp; test -x scripts/setup_pod.sh &amp;&amp; git diff --exit-code scripts/setup_rvc.sh</automated>
  </verify>

  <acceptance_criteria>
    - `test -x scripts/setup_pod.sh` succeeds (executable bit set)
    - `bash -n scripts/setup_pod.sh` exits 0 (bash syntax check)
    - `head -1 scripts/setup_pod.sh` is `#!/usr/bin/env bash`
    - `grep -c "set -euo pipefail" scripts/setup_pod.sh` is at least 1
    - `grep -c "_SETUP_POD_REEXEC" scripts/setup_pod.sh` is at least 2 (the guard + the export)
    - `grep -c "PIPESTATUS\[0\]" scripts/setup_pod.sh` is exactly 1
    - `grep -c "DEBIAN_FRONTEND=noninteractive" scripts/setup_pod.sh` is at least 1 (env export approach)
    - `grep -c "TZ=UTC" scripts/setup_pod.sh` is at least 1
    - `grep -c "cuda-keyring_1.1-1_all.deb" scripts/setup_pod.sh` is exactly 1
    - `grep -c "cuda-toolkit-12-1" scripts/setup_pod.sh` is at least 1
    - `grep -c "release 12.1" scripts/setup_pod.sh` is at least 2 (probe + post-install verify)
    - `grep -c "/usr/local/cuda-12.1/bin" scripts/setup_pod.sh` is at least 1
    - `grep -n "mise activate" scripts/setup_pod.sh` returns zero matches (forbidden)
    - `grep -c "mise where python@3.10" scripts/setup_pod.sh` is at least 1
    - `grep -c "ppa:deadsnakes/ppa" scripts/setup_pod.sh` is at least 1
    - `grep -c "python3.10-venv" scripts/setup_pod.sh` is at least 1
    - `grep -c "bash .*scripts/setup_rvc.sh" scripts/setup_pod.sh` is at least 2 (force and non-force branches)
    - `grep -c "src/doctor.py --training" scripts/setup_pod.sh` is exactly 1
    - `grep -c "train_audio_model.egg-info" scripts/setup_pod.sh` is at least 1
    - `grep -c "hubert_base.pt" scripts/setup_pod.sh` is at least 1
    - `grep -c "f0G40k.pth" scripts/setup_pod.sh` is at least 1
    - `grep -c "f0D40k.pth" scripts/setup_pod.sh` is at least 1
    - `grep -n "/etc/profile.d" scripts/setup_pod.sh` returns zero matches
    - `grep -n "~/.bashrc\|HOME/.bashrc" scripts/setup_pod.sh` returns zero matches
    - `grep -n "^sudo \| sudo " scripts/setup_pod.sh` returns zero matches
    - `grep -n "pip install --upgrade pip" scripts/setup_pod.sh` returns zero matches (BOOT-07)
    - `grep -n "rvc/.venv/bin/pip" scripts/setup_pod.sh` returns zero matches (script must not touch rvc pip — BOOT-07)
    - `git diff --exit-code scripts/setup_rvc.sh` exits 0 (unchanged — BOOT-05)
    - `shellcheck scripts/setup_pod.sh` reports no errors (warnings OK — matching setup_rvc.sh which has `# shellcheck disable` comments). If `shellcheck` is not installed, skip this check.
  </acceptance_criteria>

  <done>
    `scripts/setup_pod.sh` exists, is executable, passes `bash -n`, contains every required probe/command/URL, does not touch `setup_rvc.sh`, and calls `src/doctor.py --training` as its final verification.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 2: Manual pod integration verification (BOOT-01, BOOT-02)</name>

  <what-built>
    `scripts/setup_pod.sh` — fully automated by Claude in Task 1. It has been syntax-checked (`bash -n`), all acceptance-criteria greps pass, and it invokes `src/doctor.py --training` (Plan 01-01 deliverable) as its final verification layer. What Claude CANNOT automate is running it on a real rented GPU pod to verify the integration success criteria from ROADMAP Phase 1: "Running `bash scripts/setup_pod.sh` on a clean Ubuntu 22.04 + NVIDIA-driver image completes without any interactive prompt and exits 0" (BOOT-01) and "Re-running on an already-provisioned pod completes in under 30 seconds" (BOOT-02).
  </what-built>

  <how-to-verify>
    Rent a fresh Ubuntu 22.04 + NVIDIA-driver pod (RunPod, Vast.ai, or Lambda Labs — any provider that gives you a bare 22.04 + driver image).

    **Run 1 — cold bootstrap (BOOT-01):**
    1. `git clone <repo> && cd train_audio_model`
    2. `time bash scripts/setup_pod.sh 2>&1 | tee /tmp/cold.log`
    3. Expected: exits 0. No `?` / `Y/N` / `Continue?` interactive prompts appear in stdout. Total wall time probably 10-25 minutes dominated by apt + pip + RVC weight download.
    4. After success, confirm artifacts:
       - `.venv/bin/python --version` reports `Python 3.10.x`
       - `rvc/.venv/bin/python --version` reports `Python 3.10.x`
       - `ls -la rvc/assets/hubert/hubert_base.pt rvc/assets/rmvpe/rmvpe.pt rvc/assets/pretrained_v2/f0G40k.pth rvc/assets/pretrained_v2/f0D40k.pth` — all files exist and each is above its size floor (100 MB, 100 MB, 30 MB, 30 MB respectively)
       - `.venv/bin/python src/doctor.py --training` exits 0 (passes on provisioned pod — ROADMAP Phase 1 success criterion 3)

    **Run 2 — warm re-run (BOOT-02):**
    1. `time bash scripts/setup_pod.sh 2>&1 | tee /tmp/warm.log`
    2. Expected: exits 0. `real` time reported by `time` is **under 30 seconds** (target ~10 s).
    3. Grep the log for probe-skip confirmations:
       - `grep -i "already" /tmp/warm.log` should show at least: CUDA already installed, python3.10 found (or venv already 3.10), editable install already present, RVC already cloned (from setup_rvc.sh).
       - `grep -ci "creating\|installing\|downloading" /tmp/warm.log` should be low (some echo lines with these words for skipped layers are OK; the actual install subcommands should not have run).

    **Cross-checks:**
    - Verify `scripts/setup_rvc.sh` is byte-for-byte unchanged on the pod vs the committed version (`git diff scripts/setup_rvc.sh` empty on the pod clone).
    - Verify `rvc/.venv/bin/pip --version` reports pip < 24.1 (BOOT-07 — setup_pod.sh must not have touched it).
    - Verify no new files under `/etc/profile.d/` matching `cuda*` (D-04 — no persistent PATH pollution).

    **Paste to approve:** the output of `time bash scripts/setup_pod.sh` for Run 2, confirming exit 0 and real time < 30s.
  </how-to-verify>

  <files>scripts/setup_pod.sh (no file edits — manual verification only)</files>
  <action>
    Human-only action: rent a fresh Ubuntu 22.04 + NVIDIA-driver GPU pod, clone the repo, and execute the verification steps in `<how-to-verify>` below. Claude cannot automate this — it requires a real rented GPU pod that costs money to provision. Everything Claude could automate (script creation, syntax check, grep-based criteria) is in Task 1.
  </action>
  <verify>
    <automated>MISSING — this is a manual human-verify checkpoint. See `<how-to-verify>` for the required manual steps and `<resume-signal>` for the paste-back protocol.</automated>
  </verify>
  <done>
    User has pasted Run 2 timing output showing `real` time under 30 seconds and exit 0, confirmed `src/doctor.py --training` exits 0 on the pod, confirmed `scripts/setup_rvc.sh` unchanged on the pod clone, and typed "approved".
  </done>
  <resume-signal>Paste Run 2 timing output, or describe issues encountered. Type "approved" if both runs succeed and warm re-run is under 30s.</resume-signal>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| internet→pod | `wget` pulls `cuda-keyring_1.1-1_all.deb` over HTTPS from `developer.download.nvidia.com` (supply chain entry) |
| apt→pod | `apt-get install` pulls CUDA toolkit, deadsnakes Python, apt prerequisites from Ubuntu archive + NVIDIA apt repo + Launchpad PPA |
| script→root | `setup_pod.sh` runs as root on the pod (pod user IS root); apt requires elevation |
| setup_pod.sh→setup_rvc.sh | Delegation via bash subprocess; setup_rvc.sh in turn pulls from GitHub (RVC) and Hugging Face (weights) |
| pod→log file | `scripts/setup_pod.log` captures all command output (stdout+stderr) via tee |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-02-01 | Tampering | NVIDIA cuda-keyring `.deb` download over HTTPS | mitigate | Download URL pinned to `https://developer.download.nvidia.com/.../cuda-keyring_1.1-1_all.deb` (TLS-authenticated). The `.deb` installs NVIDIA's GPG key; all subsequent `apt-get install` of `cuda-toolkit-12-1` is signature-verified by apt. Residual MITM risk on the first `wget` is accepted for v1 (ASVS L1 accept — pod is ephemeral, no secrets, no persistent identity). SHA256 pinning is noted as a future hardening in 01-RESEARCH.md §Security Notes #1 but is V2 scope. |
| T-02-02 | Tampering | deadsnakes PPA fallback rung | mitigate | `add-apt-repository -y ppa:deadsnakes/ppa` imports the GPG key from Launchpad over HTTPS. PPA is a well-known Ubuntu community source with stable maintainer key. Accepted risk — no code change needed. |
| T-02-03 | Tampering | RVC clone + HuggingFace weight download | transfer | Delegated to `scripts/setup_rvc.sh` which is already in repo and already accepted. No change to its threat surface. Phase 1 does not modify it. |
| T-02-04 | Denial of Service | Interactive apt/dpkg prompts blocking on a billing pod | mitigate | Mandatory `export DEBIAN_FRONTEND=noninteractive; export TZ=UTC` envelope before any apt call. `add-apt-repository -y` used. `apt-get install -y` used everywhere. Enforced by acceptance criterion greps. |
| T-02-05 | Elevation of Privilege | Script assumes root | accept | Script is explicitly "POD-ONLY" per header comment and 01-RESEARCH.md §Security Notes #5. On the pod, the user IS root; no sudo tango. On a non-pod machine, `apt-get` without privileges will fail-fast which is correct behavior. Documented in script header. |
| T-02-06 | Information Disclosure | `scripts/setup_pod.log` captures apt/nvidia-smi/pip output via tee | accept | Log contains apt install output, nvidia-smi output, pip resolver output — no secrets, no PII. Log is root-owned on the pod and the pod is ephemeral. Safe per research §Security Notes #8. Log is gitignored (confirm — add to .gitignore in this task if missing). |
| T-02-07 | Denial of Service | Partial/half-installed CUDA state from interrupted run (`iF` dpkg state) | mitigate | D-01 primary probe is `nvcc --version | grep -q "release 12.1"` — checks the real artifact, not dpkg state. A half-installed package fails the probe and triggers reinstall. `dpkg -l` is explicitly NOT used as primary per research §Key Findings #8 and Open Question #3. |
| T-02-08 | Tampering | Weight file size floors are integrity-light, not cryptographic | accept | A 100 MB attacker-crafted sentinel could pass the floor. Accepted because upstream is HTTPS HuggingFace (TLS-authenticated), same as accepted for T-01-01. Cryptographic hash pinning is V2. Phase 2's `check_pretrained_v2_weights` will extend this for the full sample-rate matrix; cryptographic hashing is deferred further. |
| T-02-09 | Elevation of Privilege | `--force` is the only user input and is pattern-matched as literal `$1` | mitigate | Only `[[ "${1:-}" == "--force" ]]` check; no interpolation into shell commands beyond this boolean. No injection surface. Per research §Security Notes #6. |
| T-02-10 | Configuration drift | In-script `PATH` export leaks into no persistent state | mitigate | D-04: `export PATH="/usr/local/cuda-12.1/bin:$PATH"` is inside the running bash process only. No writes to `/etc/profile.d/`, no writes to `~/.bashrc`. Enforced by acceptance criterion greps. Re-rented pod = clean slate. Security-positive. |

ASVS L1 mapping:
- **V10 Malicious Code:** apt GPG signing (inherited), HTTPS keyring download (pinned URL), pinned RVC commit (delegated) — all addressed.
- **V14 Configuration:** noninteractive envelope (T-02-04), no persistent path pollution (T-02-10), no secrets in logs (T-02-06) — all addressed.

No high-severity items remain.
</threat_model>

<verification>
**Autonomous (runs in CI / on any dev box):**
- `bash -n scripts/setup_pod.sh` — syntax valid
- `test -x scripts/setup_pod.sh` — executable
- `git diff --exit-code scripts/setup_rvc.sh` — unchanged (BOOT-05)
- All acceptance-criteria greps in Task 1 pass
- `shellcheck scripts/setup_pod.sh` (if installed) reports no errors

**Manual (requires real pod — Task 2 checkpoint):**
- BOOT-01: cold bootstrap on Ubuntu 22.04 + NVIDIA-driver pod exits 0, no interactive prompts, leaves `.venv`, `rvc/.venv`, weights
- BOOT-02: warm re-run exits 0 in under 30 seconds
- BOOT-03 (transitively verified in BOOT-01): CUDA 12.1 apt keyring install works non-interactively
- BOOT-04 (transitively verified): Python 3.10 acquired via `mise where` or earlier rung, never `mise activate`
- BOOT-05 (verified by git diff on pod): setup_rvc.sh byte-for-byte unchanged
- BOOT-06 (verified by `doctor.py --training` passing): torch.cuda.is_available() True in rvc/.venv
- BOOT-07 (verified by `rvc/.venv/bin/pip --version` on pod): pip < 24.1 preserved
- BOOT-08 (verified by Task 1's size-floor layer AND by `doctor.py --training` `check_hubert_base`): weights populated and above size floors
</verification>

<success_criteria>
- `scripts/setup_pod.sh` created, executable, bash-syntax-clean, lifts concrete snippets verbatim from 01-RESEARCH.md.
- Every BOOT requirement 01-08 has a clear implementation path in the script.
- `scripts/setup_rvc.sh` is unchanged (BOOT-05).
- Final verification layer invokes Plan 01-01's `--training` flag, giving end-to-end readiness confirmation on the pod.
- Pod integration checkpoint (Task 2) confirms cold bootstrap BOOT-01 and warm re-run BOOT-02 on a real Ubuntu 22.04 pod.
</success_criteria>

<output>
After completion, create `.planning/phases/01-pod-bootstrap/01-02-SUMMARY.md` covering:
- Final layer list and ordering in the script
- Approximate line count
- Output of Task 2's manual pod verification (Run 1 cold time, Run 2 warm time, `doctor --training` result)
- Any deviations from 01-RESEARCH.md's concrete snippets and their rationale
- Requirement coverage: BOOT-01..BOOT-08 ✓ (BOOT-09, BOOT-10 covered by Plan 01-01)
- Confirmation that `scripts/setup_rvc.sh` was not modified (`git diff HEAD~N scripts/setup_rvc.sh` empty)
</output>
