# Phase 1: Pod Bootstrap - Context

**Gathered:** 2026-04-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver `scripts/setup_pod.sh` — a single non-interactive bash script that takes a bare Ubuntu 22.04 + NVIDIA-driver pod to a fully provisioned state (CUDA toolkit 12.1 → Python 3.10 → app venv → RVC venv → RVC clone → pretrained weights), following a probe-and-skip detect-and-adapt pattern so re-runs complete in under 30 seconds. Also deliver new training-specific doctor checks (`check_disk_space_floor`, `check_gpu_vram_floor`) and a `python src/doctor.py --training` flag that composes the full training-readiness check set.

**Out of scope for Phase 1:** `src/train.py` itself (Phase 2), FAISS index shim (Phase 3), `scripts/train.sh` + remote pull (Phase 4), README docs (Phase 5).

</domain>

<decisions>
## Implementation Decisions

### CUDA Toolkit Handling
- **D-01:** Probe for existing CUDA 12.1 toolkit via `nvcc --version` grep for `release 12.1`. If matched, skip the CUDA install layer entirely. Strict 12.1 match only — not any 12.x — to avoid silent drift against RVC's `torch 2.1.2+cu121` pin.
- **D-02:** When CUDA install is needed on Ubuntu 22.04, use the NVIDIA apt keyring method under `DEBIAN_FRONTEND=noninteractive TZ=UTC`: download `cuda-keyring_*.deb`, `apt-get update`, `apt-get install -y cuda-toolkit-12-1`. This matches BOOT-03 verbatim.
- **D-03:** On Ubuntu 24.04 (detected via `/etc/os-release`), print a loud warning (`"Ubuntu 24.04 is best-effort — 22.04 recommended per DOCS-02"`) then attempt the 22.04 apt keyring anyway. If the apt install fails, exit 1 with a clear fix hint (`"rent a 22.04 pod or install CUDA 12.1 manually, then re-run setup_pod.sh"`). No runfile fallback — runfile recipe is only medium confidence per STATE.md and adds a second install code path.
- **D-04:** After a successful CUDA install, export `PATH=/usr/local/cuda-12.1/bin:$PATH` **in-script only** (within setup_pod.sh's own process) so subsequent probes see `nvcc`. Do NOT write to `/etc/profile.d/` or `~/.bashrc` — pod shells are ephemeral and RVC's torch wheels bundle their own CUDA runtime libs. Minimal system pollution.

### Python 3.10 Acquisition Ladder
- **D-05:** `mise` is a local-dev convenience, not a pod requirement. The pod script uses a probe-and-skip ladder; first match wins, subsequent layers are skipped:
  1. `.venv/bin/python --version` reports `Python 3.10.x` → full skip (idempotent re-run case).
  2. `python3.10 --version` or `which python3.10` in PATH → use it to create `.venv` (PyTorch base images and some pod templates ship 3.10 preinstalled).
  3. `mise` binary present in PATH → `mise install python@3.10` (respects the project `.mise.toml`), capture `MISE_PY=$(mise where python)/bin/python3`, use for venv creation. **Never** `mise activate bash` (STATE.md pitfall).
  4. Fallback: add the deadsnakes PPA under `DEBIAN_FRONTEND=noninteractive TZ=UTC`, `apt install -y python3.10 python3.10-venv python3.10-dev`, use `/usr/bin/python3.10`.
- **D-06:** App venv creation: probe `.venv/bin/python --version`. If 3.10.x → skip layer. Otherwise call `$PY310 -m venv .venv`. After venv exists, skip `pip install -e ".[dev]"` if `src/train_audio_model.egg-info` (or equivalent dist-info under site-packages) is present. This is what keeps the idempotent re-run under the BOOT-02 ceiling. A `--force` flag on setup_pod.sh wipes `.venv` and reinstalls.

### RVC Venv + Weights
- **D-07:** setup_pod.sh delegates the entire `rvc/.venv` creation and weight download to the existing `scripts/setup_rvc.sh` — it does NOT modify or reimplement any of that logic (BOOT-05). Bootstrap's job is to ensure `.venv/bin/python` exists and is 3.10, then shell out to `bash scripts/setup_rvc.sh`. `setup_rvc.sh`'s own probe-and-skip handles re-runs, preserves the `pip<24.1` pin (BOOT-07), and runs `rvc/tools/download_models.py`.
- **D-08:** After `setup_rvc.sh` returns successfully, setup_pod.sh runs a final verification layer: `rvc/.venv/bin/python -c "import torch; assert torch.cuda.is_available()"` (BOOT-06), and asserts file sizes on `rvc/assets/hubert/hubert_base.pt`, `rvc/assets/rmvpe.pt`, and `rvc/assets/pretrained_v2/*.pth` (BOOT-08). On any failure, print diagnostic and exit 1.

### Doctor --training Check Set
- **D-09:** `python src/doctor.py --training` composes a **full readiness check set** (not just the new additions). It runs: existing system checks (ffmpeg, git, nvidia-smi, mise is optional/skipped if missing), existing RVC checks (rvc cloned, rvc venv, rvc weights, rvc torch CUDA, slicer2 importable), and the NEW training checks: `check_disk_space_floor(PROJECT_ROOT, 20)`, `check_gpu_vram_floor(12)`, `check_pretrained_v2_weights` (foreshadowed for Phase 2, not required as full check in Phase 1 but the plumbing should exist), `check_hubert_base`, `check_rvc_mute_refs`. One command = "is this pod ready to train?". Phase 2's `train.py` will call this exact set pre-flight.
- **D-10:** `check_disk_space_floor(path: Path, min_gb: int) -> CheckResult` — measures free space on the partition containing `path` via `shutil.disk_usage(path)`. Default path is `PROJECT_ROOT`. Default floor for the `--training` composition is **20 GB** (covers ~5 GB RVC weights+venv, ~5 GB dataset raw+processed, ~5 GB training intermediates, ~5 GB export headroom). Caller can override.
- **D-11:** `check_gpu_vram_floor(min_gb: int) -> CheckResult` — measures total VRAM via `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`, parses MiB, converts to GB, takes the MAX across GPUs (single-GPU target anyway). Default floor is **12 GB** — locks out 8 GB cards, admits RTX 3060 12GB and above. No torch import, no `rvc/.venv` cross-boundary hop — pure shell-out matching the existing `check_nvidia_smi` pattern.

### Idempotency + Error Handling + Logging
- **D-12:** Idempotency is **intrinsic-probe-only**. No marker files. Each layer probes the real artifact (nvcc version, python3.10 --version, .venv/bin/python version, rvc/.git HEAD commit, rvc/.venv torch import, file sizes on weights). If the real artifact is correct, layer is done. Zero stale-marker risk, no cleanup code. Matches existing `setup_rvc.sh` pattern.
- **D-13:** Failure mode is **fail-fast, no rollback**. `set -euo pipefail` at the top of setup_pod.sh. On any command failure, print: the layer name, the command that failed, and the last ~20 lines of its stderr (mirror the `_tail` pattern from `src/generate.py`). Exit 1. User re-runs setup_pod.sh; the probe-and-skip design means completed layers are no-ops and retry continues where it left off. No attempt to undo a partial apt install or partial venv — rollback code is fragile and pods are cheap to re-provision.
- **D-14:** Logging follows the exact re-exec + `tee -a` pattern from `scripts/setup_rvc.sh`: guard with a `_SETUP_POD_REEXEC` env var, re-exec the script piped to `tee -a scripts/setup_pod.log`, use `set -o pipefail` and check `PIPESTATUS[0]` so `tee` doesn't swallow non-zero exits. User sees real-time output AND gets a full log for post-mortem (critical on billing pods where the terminal session may be gone before debugging is possible).

### Unit Testing (BOOT-10)
- **D-15:** BOOT-10 is satisfied by unit tests in `tests/unit/` for the two new doctor functions:
  - `test_check_disk_space_floor`: mocks `shutil.disk_usage` to return varying free-space tuples and asserts `ok`/`detail`/`fix_hint` correctness for the "above floor", "at floor", and "below floor" cases.
  - `test_check_gpu_vram_floor`: mocks `subprocess.run` for the `nvidia-smi` call, returning canned CSV output, asserts the correct max-across-GPUs parsing and floor comparison.
  - No unit tests for the bash script itself (matches project convention — `setup_rvc.sh` has none either).

### Claude's Discretion
- Exact error message text, `rich.Table` cosmetic formatting, and internal function naming within `src/doctor.py` and `scripts/setup_pod.sh` — planner and executor may choose what reads best.
- Layer ordering within setup_pod.sh past the hard dependency chain (apt prerequisites → CUDA → Python → app venv → setup_rvc.sh → final verification).
- Whether to extract shared apt helpers in the bash script or inline them.
- Whether to add a `--skip-cuda` or similar debugging escape hatches — not required by any BOOT-* but low-cost if it helps local iteration.

### Folded Todos
(None — no pending todos matched this phase.)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Planning (in-repo)
- `.planning/REQUIREMENTS.md` §Bootstrap — BOOT-01 through BOOT-10, the authoritative requirement text.
- `.planning/ROADMAP.md` §"Phase 1: Pod Bootstrap" — success criteria (4 items) that VERIFICATION must hit.
- `.planning/PROJECT.md` §Constraints — mise preference, two-venv boundary, pip<24.1 pin, no-interactive-prompts rule.
- `.planning/STATE.md` §"Critical Pitfalls to Remember" — `mise activate bash` forbidden, `DEBIAN_FRONTEND=noninteractive TZ=UTC` mandatory.

### Existing Code To Extend (not modify)
- `scripts/setup_rvc.sh` — the delegated-to script. setup_pod.sh calls this; does not replicate or modify it. Read to understand the re-exec + tee logging pattern (lines 22-31) which setup_pod.sh mirrors, and the probe-and-skip pattern for clone/venv/weights (lines 55+).
- `src/doctor.py` — where the new `check_disk_space_floor` and `check_gpu_vram_floor` functions live, and where the `--training` flag is added. Follow the `CheckResult` dataclass shape and the existing `check_nvidia_smi` pattern for subprocess-based checks.
- `src/generate.py` `_tail` helper — the stderr-tail-on-failure pattern to mirror in setup_pod.sh's failure diagnostic.
- `src/ffmpeg_utils.py` — *not* touched by Phase 1, but read to confirm the "pure arg builder + thin runner" pattern the rest of the codebase follows (relevant for Phase 2's planner, not this phase).

### External References
- NVIDIA CUDA installation for Ubuntu 22.04 (apt keyring method): <https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network>. Commands are already stabilized in the community; the planner can extract the exact `cuda-keyring_*.deb` URL from that page.
- deadsnakes PPA for Python 3.10 on Ubuntu: `ppa:deadsnakes/ppa` — standard, no auth required.

### Vendored RVC (read-only black box)
- `rvc/tools/download_models.py` — called by `setup_rvc.sh`; Phase 1 verifies its outputs (file sizes) but does not invoke it directly.
- RVC pinned commit: `7ef19867780cf703841ebafb565a4e47d1ea86ff` (2024-11-24) — must not change.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`scripts/setup_rvc.sh`** — the re-exec + `tee -a` logging pattern at lines 22-31 is copy-pasteable into setup_pod.sh with only variable renames (`_SETUP_POD_REEXEC`, `setup_pod.log`). The probe-and-skip pattern for RVC clone/venv is the reference implementation for setup_pod.sh's CUDA/Python layers.
- **`src/doctor.py` `CheckResult` dataclass (lines 33-38)** — shared return shape for all new checks. `name`, `ok`, `detail`, `fix_hint`. No changes needed.
- **`src/doctor.py` `check_nvidia_smi`** — the reference pattern for a subprocess-based check that parses tool output, to be mirrored by `check_gpu_vram_floor`.
- **`src/doctor.py` `check_ffmpeg_filters`** — reference for composing multiple sub-probes under one named check.
- **`src/generate.py` `_tail`** — the 20-line stderr tail helper for failure diagnostics; conceptually ported to bash in setup_pod.sh's error trap.

### Established Patterns
- **Two-venv boundary is absolute** — no Python imports cross `./.venv` ↔ `./rvc/.venv`. Bootstrap enforces this structurally by having separate venv creation and only talking to `rvc/.venv` via subprocess.
- **Pure arg builders + thin runners** — does not apply to Phase 1's bash script directly, but `src/doctor.py` functions stay pure (return `CheckResult`, no side effects, no raising).
- **Fail-fast at script top** — every existing bash script in `scripts/` uses `set -euo pipefail`. Inherit.
- **`DEBIAN_FRONTEND=noninteractive TZ=UTC`** must prefix every apt invocation in setup_pod.sh. No exceptions.

### Integration Points
- `scripts/setup_pod.sh` → `scripts/setup_rvc.sh` (shell-out delegation for RVC venv + weights).
- `scripts/setup_pod.sh` → `src/doctor.py --system-only` (existing) for an early pre-apt sanity check.
- `scripts/setup_pod.sh` → `src/doctor.py --training` (new, this phase) for final verification after all layers complete.
- `src/doctor.py` → `shutil.disk_usage` (stdlib) for disk floor.
- `src/doctor.py` → `subprocess.run("nvidia-smi ...")` for VRAM floor.

</code_context>

<specifics>
## Specific Ideas

- **Idempotent re-run target is ~10 seconds, hard ceiling 30 seconds** (BOOT-02 says "under 30 seconds"). Every layer's probe must be a single cheap command. Counter-example to avoid: running `apt list --installed` on the full apt database would blow past this — use `dpkg -l cuda-toolkit-12-1 2>/dev/null` or probe `nvcc --version` directly.
- **Exit code discipline** (matches project convention in `src/generate.py` and `src/doctor.py`):
  - `0` on success
  - `1` on config/setup error (missing driver, OS detection fail, CUDA install fail, venv creation fail, weight download fail)
  - `2` on user-input error (currently unused in setup_pod.sh since there are no input flags beyond optional `--force`; reserve for the future)
  - `3` on subprocess runtime error (setup_rvc.sh fails, `rvc/.venv` torch check fails)
- **Force flag:** `scripts/setup_pod.sh --force` wipes `.venv` and forwards `--force` to `setup_rvc.sh`. Does NOT wipe the CUDA apt install (that's an OS-level change the user should undo manually if they really want a clean slate).
- **OS probe hierarchy:** `source /etc/os-release` then branch on `VERSION_ID`. If `VERSION_ID="22.04"` → happy path. If `VERSION_ID="24.04"` → warn + attempt. Any other value → exit 1 with "Only Ubuntu 22.04 tested; 24.04 best-effort".

</specifics>

<deferred>
## Deferred Ideas

- **Aggressive caching of pretrained weights across pod reboots via persistent volume** — already captured as V2-FAST-01 in REQUIREMENTS.md. Not in this milestone.
- **Prebuilt wheel cache for `rvc/.venv` installs** — already captured as V2-FAST-02. Not in this milestone.
- **Parallel installs** — explicitly deprioritized in PROJECT.md ("Correctness over speed").
- **Provider-specific pod image detection (RunPod, Vast.ai templates)** — out of scope per REQUIREMENTS.md "Out of Scope" and ROADMAP.md Phase 5 scope.
- **Smart batch-size default based on detected GPU VRAM** — captured as V2-TRAIN-01. Separate phase.

### Reviewed Todos (not folded)
(None — no todos were matched against Phase 1.)

</deferred>

---

*Phase: 01-pod-bootstrap*
*Context gathered: 2026-04-09*
