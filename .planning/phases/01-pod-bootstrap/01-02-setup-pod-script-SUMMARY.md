---
phase: 01-pod-bootstrap
plan: 02
name: setup-pod-script
subsystem: infra
tags: [bootstrap, bash, cuda, apt, pod, ffmpeg, python, mise, rvc]
dependency_graph:
  requires:
    - phase: 01-01-doctor-training-checks
      provides: "python src/doctor.py --training"
  provides:
    - "scripts/setup_pod.sh (idempotent pod bootstrap)"
    - "src.doctor.parse_ffmpeg_version (nightly-build aware)"
    - "src.doctor.parse_ffmpeg_version_display"
    - "src.doctor.check_mise (soft-OK on missing binary)"
  affects:
    - "Phase 2 train.py (assumes setup_pod.sh has run)"
    - "Phase 4 train.sh (invokes setup_pod.sh as first step)"
tech_stack:
  added:
    - "BtbN static ffmpeg build (nightly master, ffmpeg 7.x GPL variant) — installed to /usr/local/bin"
  patterns:
    - "Re-exec + tee -a logging with _SETUP_POD_REEXEC guard and PIPESTATUS[0] exit propagation"
    - "Probe-and-skip layering: every layer intrinsically probes the real artifact (nvcc --version, stat size, dpkg Status, torch.cuda.is_available) before mutating state"
    - "DEBIAN_FRONTEND=noninteractive + TZ=UTC envelope exported once at top; inherited by every apt-get call"
    - "Hard-dependency layer ordering (D-12): OS → apt prereqs → ffmpeg → CUDA → Python → app venv → RVC → weight floors → doctor --training"
    - "Root-only pod-side scripts fail fast on non-root EUID (pods run as root; no sudo)"
    - "soft-OK pattern for optional dev-only dependencies (check_mise) — missing binary returns ok=True, broken install still fails"
key_files:
  created:
    - "scripts/setup_pod.sh (382 lines)"
  modified:
    - "src/doctor.py (parse_ffmpeg_version nightly support, parse_ffmpeg_version_display, check_mise soft-OK)"
    - "tests/unit/test_doctor.py (nightly parse/display tests + mise soft-OK/soft-fail tests)"
    - ".gitignore (scripts/setup_pod.log)"
decisions:
  - "Install ffmpeg via BtbN static build rather than apt — Ubuntu 22.04 ships 4.4, below our >=5.0 floor"
  - "Accept BtbN nightly version tags (N-123884-...) in both doctor parse and shell probe, with a sentinel (9999,0,0) tuple satisfying floor comparisons"
  - "check_mise returns soft-OK when binary is missing — mise is laptop-dev tooling, check_python_version is the real gate on pods"
  - "Skip the entire setup_rvc.sh delegation when clone + venv + torch+CUDA + all weight sentinels are already in place (rvc/tools/download_models.py is not probe-and-skip and would re-download ~35 files every warm run)"
  - "Skip apt update+install when all prereqs are already dpkg-installed (removes the only remaining network I/O on warm runs)"
  - "Pod script fails fast if EUID != 0 (pods run as root without sudo; running on a laptop shell would die mid-apt with a confusing error)"
metrics:
  duration: "~1h (Task 1 ~20min + Task 2 real-pod verification + 5 deviation fixes)"
  tasks: 2
  files_touched: 4
  tests_added: 8
  completed: "2026-04-09"
---

# Phase 1 Plan 02: setup-pod-script Summary

**Idempotent non-interactive pod bootstrap script (`scripts/setup_pod.sh`) that takes a bare Ubuntu 22.04 + NVIDIA-driver pod to a fully provisioned training environment in one invocation; warm re-runs complete in ~26s under the 30s target.**

## Performance

- **Duration:** ~1h end-to-end (Task 1 code ~20min + real-pod verification + 5 follow-up deviation commits)
- **Tasks:** 2
- **Files modified:** 4 (1 created, 3 modified)
- **Final warm re-run timing (BOOT-02):** `real 0m25.810s  user 0m3.516s  sys 0m8.834s`

## Accomplishments

- `scripts/setup_pod.sh` — 382-line idempotent pod bootstrap covering 9 layers:
  1. OS detection (22.04 happy path, 24.04 best-effort warn, else exit)
  2. apt prerequisites (ca-certificates, wget, gnupg, software-properties-common, xz-utils) with dpkg probe-and-skip
  3. ffmpeg >= 5.0 via BtbN static build (GPL variant; afftdn/loudnorm/silencedetect filters verified)
  4. CUDA toolkit 12.1 via NVIDIA apt keyring (strict `nvcc --version | grep "release 12.1"` probe)
  5. Python 3.10 acquisition ladder: PATH → `mise where python@3.10` → deadsnakes PPA
  6. App venv (`.venv`) + editable install with `train_audio_model.egg-info` probe
  7. RVC venv + weights (delegated to `scripts/setup_rvc.sh` unchanged — with a whole-delegation skip probe)
  8. Weight file size floors on hubert_base.pt, rmvpe.pt, f0G40k.pth, f0D40k.pth
  9. Final verification: `.venv/bin/python src/doctor.py --training` (14 checks, all OK on warm run)
- Real-pod verification (BOOT-01 cold / BOOT-02 warm) completed on RunPod Ubuntu 22.04 + RTX 4090.
- `scripts/setup_rvc.sh` unchanged byte-for-byte (BOOT-05 contract preserved).
- Doctor hardened against real-world ffmpeg/mise edge cases discovered on the pod.

## Task Commits

1. **Task 1: Write `scripts/setup_pod.sh`** — `60cb113` (feat: add scripts/setup_pod.sh pod bootstrap)
2. **Task 2: Real-pod verification (BOOT-01 + BOOT-02)** — no code commit; verification produced 5 follow-up deviation commits below.

**Deviation follow-ups (all in response to real-pod testing):**

- `8366a70` — fix(01-02): install ffmpeg>=5 on pod + fail fast when not root
- `597887a` — fix(01-02): accept nightly ffmpeg build tags in version parse
- `c11f6ee` — fix(01-02): treat missing mise as soft-OK on pods
- `62ac65b` — perf(01-02): skip setup_rvc.sh delegation on already-provisioned pod
- `d0c8736` — perf(01-02): skip apt update+install when all prereqs already satisfied

## Files Created/Modified

- `scripts/setup_pod.sh` (created) — 382-line pod bootstrap (9 idempotent layers)
- `src/doctor.py` (modified) — `parse_ffmpeg_version` accepts nightly `N-<digits>` tags (sentinel `(9999, 0, 0)`); new `parse_ffmpeg_version_display` helper returns raw version token for honest user output; `check_mise` returns soft-OK when binary missing (fails only on broken install)
- `tests/unit/test_doctor.py` (modified) — 8 new tests: nightly parse, display (stable/nightly/unknown), check_ffmpeg accepting nightly output, `test_check_mise_missing_is_soft_ok`, `test_check_mise_broken_install_is_soft_fail`
- `.gitignore` (modified) — `scripts/setup_pod.log` (T-02-06)

## Verification

**Cold run (BOOT-01):** Real RunPod Ubuntu 22.04 + RTX 4090. `bash scripts/setup_pod.sh` completed in ~25+ minutes (dominated by the RVC venv `pip install torch==2.1.2+cu121` ~2.2 GB wheel → ~5.5 GB unpacked). All 14 `doctor --training` checks passed.

**Warm re-run (BOOT-02):**

```
$ time bash scripts/setup_pod.sh
real    0m25.810s
user    0m3.516s
sys     0m8.834s
```

Exit 0. Well under the 30s target. Layer trace:

- OS detection → OK (Ubuntu 22.04)
- apt prerequisites → "All apt prerequisites already installed — skipping apt update/install."
- ffmpeg >= 5.0 → "ffmpeg already satisfies >=5.0 with required filters (N-123884-gd3d0b7a5ee-20260409)"
- CUDA toolkit 12.1 → already installed (V12.1.105)
- Python 3.10 → /usr/bin/python3.10
- app venv → `.venv` already exists, egg-info present, skipped
- RVC venv + weights → "RVC already provisioned — skipping setup_rvc.sh."
- weight file size floors → 4/4 OK
- doctor --training → all 14 checks OK (Python 3.10.12, ffmpeg nightly, CUDA 12.1, RTX 4090 24 GB VRAM, torch 2.1.2+cu121, disk floor, rvc_mute_refs, hubert_base.pt, ...)

## Decisions Made

All five deviations below are key decisions forced by real-world pod conditions; see the Deviations section for rationale.

## Deviations from Plan

Five real-world deviations were applied after Task 1 during Task 2 pod verification. All are Rule 1 (Bug) or Rule 2 (Missing Critical) fixes — no architectural changes, no auth gates, no new scope.

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Add ffmpeg >= 5.0 installation layer (BtbN static build) + fail-fast on non-root**

- **Found during:** Task 2 (cold-run BOOT-01 on real pod)
- **Issue:** The original script had no ffmpeg layer at all — it assumed the base image shipped a working ffmpeg >= 5.0. Ubuntu 22.04 apt ships ffmpeg 4.4, below the `MIN_FFMPEG_VERSION = 5.0` floor in `src/doctor.py`, so the final `doctor --training` verification failed at the end of an otherwise successful ~25-minute provision. Additionally, running the script on a non-root laptop shell died mid-apt with a confusing error because pod containers run as root without sudo.
- **Fix:**
  - Added a new ffmpeg layer installing the BtbN static build (GPL variant, ffmpeg 7.x nightly) into `/usr/local/bin`. The probe verifies both version and required filters (`afftdn`, `loudnorm`, `silencedetect`) and is idempotent.
  - Removes any apt-installed ffmpeg before extracting the static build to avoid PATH shadowing.
  - Added `xz-utils` to apt prerequisites (needed for `.tar.xz` extraction).
  - Added an `EUID != 0` guard at the top that fails fast with a clear "POD-ONLY" error message.
- **Files modified:** `scripts/setup_pod.sh`
- **Commit:** `8366a70`

**2. [Rule 1 - Bug] Accept BtbN nightly ffmpeg version tags in doctor parse**

- **Found during:** Task 2 (after deviation #1)
- **Issue:** BtbN static builds report their version as a git tag like `ffmpeg version N-123884-gd3d0b7a5ee-20260409`. The existing regex `ffmpeg version (\d+)\.(\d+)` in `parse_ffmpeg_version` rejected this, causing both the shell probe in `setup_pod.sh` AND the doctor `check_ffmpeg` to fail with "version <5.0 or required filters missing" even though the binary is a post-7.x master build, far above the 5.0 floor.
- **Fix:**
  - `parse_ffmpeg_version`: added a second regex branch matching `N-<digits>` nightly builds, returning a sentinel `(9999, 0, 0)` tuple that satisfies any floor comparison.
  - New `parse_ffmpeg_version_display` helper returns the raw first-line version token so `check_ffmpeg`'s `detail` field shows the honest build tag (e.g. `N-123884-gd3d0b7a5ee-20260409`) instead of a misleading `9999.0.0`.
  - Mirrored the same parse logic in `scripts/setup_pod.sh`'s `_ffmpeg_ok` shell probe so script and doctor agree byte-for-byte.
  - Added 4 unit tests (nightly parse, display stable/nightly/unknown, check_ffmpeg accepting a nightly build output).
- **Files modified:** `src/doctor.py`, `scripts/setup_pod.sh`, `tests/unit/test_doctor.py`
- **Commit:** `597887a`

**3. [Rule 1 - Bug] Treat missing `mise` as soft-OK in `check_mise`**

- **Found during:** Task 2 (during cold-run BOOT-01)
- **Issue:** `scripts/setup_pod.sh` delegates to `scripts/setup_rvc.sh`, which runs `doctor --system-only` as its own pre-flight. `check_mise` was hard-failing on pods because mise is not installed there (it's a laptop-dev convenience for pinning Python 3.10 — pods acquire Python via distro PATH or deadsnakes). The hard failure aborted `setup_rvc.sh` and killed the whole pod bootstrap. Per BOOT-05, `setup_rvc.sh` cannot be modified, so the fix had to be in `check_mise` itself.
- **Fix:**
  - Missing `mise` binary → `CheckResult(ok=True, detail="not installed (optional — only needed on dev laptops)")`. Doctor table shows OK, no "Fix the following" banner, no non-zero exit.
  - Binary present but non-zero exit → still reported as failure so laptop users notice a real broken install.
  - `check_python_version` remains the real gate for Python 3.10 — mise just helps dev laptops get there.
  - Renamed `test_check_mise_missing` → `test_check_mise_missing_is_soft_ok`; added `test_check_mise_broken_install_is_soft_fail`.
- **Files modified:** `src/doctor.py`, `tests/unit/test_doctor.py`
- **Commit:** `c11f6ee`

**4. [Rule 1 - Bug / Perf] Skip `setup_rvc.sh` delegation entirely when RVC is already provisioned**

- **Found during:** Task 2 (warm re-run BOOT-02)
- **Issue:** `rvc/tools/download_models.py` is NOT probe-and-skip — it unconditionally re-downloads ~35 weight files on every invocation. `scripts/setup_rvc.sh` is off-limits (BOOT-05) and `rvc/` is a vendored pinned clone, so we cannot patch either. On a healthy warm pod this turned the warm run into a ~2+ minute network round-trip for bytes already on disk, blowing past the 30s target. Technically not a violation of the original plan's truths, but functionally a bug against the "well under 30s" warm-run contract.
- **Fix:** Added a whole-delegation probe layer in `setup_pod.sh` (`_rvc_already_provisioned`) that skips the `setup_rvc.sh` call entirely when ALL of the following hold:
  1. `rvc/` cloned at the pinned commit `7ef19867780cf703841ebafb565a4e47d1ea86ff`
  2. `rvc/.venv/bin/python --version` reports Python 3.10
  3. `torch.cuda.is_available()` succeeds in `rvc/.venv`
  4. hubert_base.pt, rmvpe.pt, f0G40k.pth, f0D40k.pth, and vocals.onnx all exist above their size floors

  `--force` disables the skip (explicit opt-in to clean reinstall). Cold-run and partial-state recovery behaviour are unchanged — the probe fails and the delegation runs. Rationale is captured in the script comment: pods are ephemeral (FS vanishes on pod destroy) but stable within their lifetime, and RVC pretrained weights are frozen artifacts — we never need to "upgrade to a newer model version".
- **Files modified:** `scripts/setup_pod.sh`
- **Commit:** `62ac65b`

**5. [Rule 1 - Perf] Skip `apt-get update` + install when all prereqs already dpkg-installed**

- **Found during:** Task 2 (warm re-run BOOT-02, after deviation #4)
- **Issue:** After skipping the RVC delegation, the remaining warm-run hotspot was `apt-get update` (~15–20s of network I/O talking to 6+ mirrors) followed by a no-op `apt-get install` — the packages were all already installed. ~18s of wall time vs ~15s CPU, all I/O.
- **Fix:** Added a `_apt_prereqs_ok` helper that probes each package with `dpkg -s ... | grep '^Status: install ok installed'`. If every prereq is already installed, skip both apt calls entirely. The probe is dpkg-only (no network), runs in milliseconds, and falls through to the full `apt-get update` + install path if even one package is missing or in a broken state — so partial-state recovery is preserved.
- **Files modified:** `scripts/setup_pod.sh`
- **Commit:** `d0c8736`

---

**Total deviations:** 5 auto-fixed (2 bug fixes, 1 missing critical, 2 perf bug fixes against the warm-run contract).

**Impact on plan:** All 5 were necessary to meet the plan's own truths on a real pod. Deviation #1 was required for `doctor --training` to even pass at the end of a cold run (truth: "exits 0 non-interactively, leaving ... pretrained weights"). Deviations #4 and #5 were required to meet the warm-run truth ("under 30 seconds without mutating any venv"). Deviations #2 and #3 were unblockers discovered in the act of fixing the first. No scope creep; three files outside `scripts/setup_pod.sh` were touched (`src/doctor.py`, `tests/unit/test_doctor.py`, `.gitignore`) — all via Rule 1/Rule 2, all with matching test coverage.

## Issues Encountered

See Deviations section — all issues encountered during Task 2 pod verification were resolved automatically under Rules 1–2 and committed as the 5 follow-up commits.

## Requirement Coverage

- **BOOT-01** (cold pod bootstrap exits 0 non-interactively) — verified on real RunPod Ubuntu 22.04 + RTX 4090 pod, cold run ~25min, exit 0, all 14 doctor --training checks OK.
- **BOOT-02** (warm re-run exits 0 in well under 30s) — verified: `real 0m25.810s` on real pod after deviations #4 and #5.
- **BOOT-03** (non-interactive apt: `DEBIAN_FRONTEND=noninteractive` + `TZ=UTC`) — exported once at the top of the script after the re-exec block; every apt-get call inherits.
- **BOOT-04** (strict CUDA 12.1 probe via `nvcc --version | grep "release 12.1"`) — implemented in the CUDA layer; half-installed state triggers reinstall.
- **BOOT-05** (`scripts/setup_rvc.sh` byte-for-byte unchanged) — preserved; all fixes went into `setup_pod.sh` or `src/doctor.py`. `git diff 60cb113 HEAD -- scripts/setup_rvc.sh` is empty.
- **BOOT-06** (torch+CUDA verified at end of bootstrap) — covered transitively by the final `doctor --training` layer which runs `check_rvc_torch_cuda` across the venv boundary.
- **BOOT-07** (no `pip<24.1` pin touched in the script) — the app venv layer never upgrades pip in `.venv`, and the script never touches `rvc/.venv`'s pip; the pin is managed entirely by `setup_rvc.sh`.
- **BOOT-08** (weight file size floors) — `_check_size` layer enforces 100 MB floor on hubert_base.pt and rmvpe.pt, 30 MB floor on f0G40k.pth and f0D40k.pth.

## Threat Flags

None. The plan's threat model (T-02-01..08) was accepted or mitigated in Task 1. The deviations exposed no new trust-boundary surface — ffmpeg is installed from a pinned GitHub release URL (same class of trust as the CUDA keyring from NVIDIA), and the probe logic is additive over existing intrinsics (dpkg, stat, nvcc, torch.cuda).

## Known Stubs

None. Every layer operates on real artifacts with real probes. The hash-pinning of weight files remains deferred to V2 (T-02-08) per the plan, with byte-size floors as the agreed V1 integrity-lite guard.

## Self-Check: PASSED

- [x] `scripts/setup_pod.sh` exists (382 lines, executable, contains `_SETUP_POD_REEXEC` and `#!/usr/bin/env bash`).
- [x] Commit `60cb113` exists: `feat(01-02): add scripts/setup_pod.sh pod bootstrap`.
- [x] Commit `8366a70` exists: `fix(01-02): install ffmpeg>=5 on pod + fail fast when not root`.
- [x] Commit `597887a` exists: `fix(01-02): accept nightly ffmpeg build tags in version parse`.
- [x] Commit `c11f6ee` exists: `fix(01-02): treat missing mise as soft-OK on pods`.
- [x] Commit `62ac65b` exists: `perf(01-02): skip setup_rvc.sh delegation on already-provisioned pod`.
- [x] Commit `d0c8736` exists: `perf(01-02): skip apt update+install when all prereqs already satisfied`.
- [x] `scripts/setup_rvc.sh` unchanged byte-for-byte (BOOT-05).
- [x] Warm re-run on real pod measured at 25.810s — under the 30s target (BOOT-02).
- [x] All 14 `doctor --training` checks pass on a warm pod.

---
*Phase: 01-pod-bootstrap*
*Completed: 2026-04-09*
