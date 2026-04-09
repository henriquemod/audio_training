---
phase: 01
fixed_at: 2026-04-09
review_path: .planning/phases/01-pod-bootstrap/01-REVIEW.md
iteration: 1
findings_in_scope: 9
fixed: 8
skipped: 1
status: partial
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-04-09
**Source review:** .planning/phases/01-pod-bootstrap/01-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 9 (all Warnings; no Critical issues)
- Fixed: 8
- Skipped: 1

## Fixed Issues

### WR-01: `check_mise` does not catch `PermissionError` — violates never-raise contract

**Files modified:** `src/doctor.py`
**Commit:** 62a7e1f
**Applied fix:** Added `except (PermissionError, OSError)` branches to `check_mise`, `check_ffmpeg`, and `check_ffmpeg_filters`. Returns a soft-fail `CheckResult` with a fix_hint pointing at binary permissions, preserving the never-raise contract for all three system checks.

### WR-02: `check_hubert_base` and `check_rvc_mute_refs` can raise on permission/IO errors

**Files modified:** `src/doctor.py`, `tests/unit/test_doctor_training.py`
**Commit:** 00427f3
**Applied fix:** Wrapped `mute_dir.is_dir()`/`iterdir()` and `hubert.exists()`/`hubert.stat()` in try/except OSError blocks, mirroring the `check_disk_space_floor` pattern. Added two new tests (`test_check_rvc_mute_refs_permission_error` and `test_check_hubert_base_permission_error`) that monkeypatch `Path.iterdir` and `Path.stat` to raise `PermissionError` and verify the check returns `ok=False` with an actionable detail. The hubert test also patches `Path.exists` to `True` so the second stat call is actually exercised (since `exists()` itself swallows OSError).

### WR-04: `_check_size` uses unquoted arithmetic comparison on `stat` output that can legitimately be empty

**Files modified:** `scripts/setup_pod.sh`
**Commit:** 684b199
**Applied fix:** Added `2>/dev/null || echo 0` fallback to the `stat -c '%s'` call in `_check_size`, mirroring the sibling `_rvc_already_provisioned` probe. An empty `size` no longer crashes the arithmetic comparison under `set -e`.

### WR-05: `size=$(stat ...)` local variable leakage in `_rvc_already_provisioned`

**Files modified:** `scripts/setup_pod.sh`
**Commit:** 74eeda2
**Applied fix:** Added `entry` to the `local path size min` declaration line so the loop variable no longer leaks into the caller scope.

### WR-06: `_ffmpeg_ok` embeds `ffmpeg -version` inside an `echo` under `set -o pipefail`

**Files modified:** `scripts/setup_pod.sh`
**Commit:** c016c6a
**Applied fix:** Captured the ffmpeg version line into a `ver_line` variable (with `2>/dev/null | head -1 || true`) before the echo, at both the "already satisfied" branch (line 142) and the post-install verify branch (line 168). Eliminates the SIGPIPE/pipefail fragility and makes the echo trivial.

### WR-07: `RVC_COMMIT_PIN` is duplicated across `setup_rvc.sh` and `setup_pod.sh`

**Files modified:** `scripts/setup_pod.sh`
**Commit:** 2bcadea
**Applied fix:** Added a 6-line comment above `RVC_COMMIT_PIN=` explaining the duplication with `scripts/setup_rvc.sh`, the BOOT-05 modification ban that forces the copy, and the fail-safe degradation behavior (stale pin silently disables the skip optimization but never breaks cold install). No functional change — documentation only.

### WR-08: Nightly ffmpeg regex `N-\d+` could false-match future stable version schemes

**Files modified:** `src/doctor.py`, `tests/unit/test_doctor.py`
**Commit:** 93c162a
**Applied fix:** Tightened the nightly regex from `r"ffmpeg version N-\d+"` to `r"ffmpeg version N-\d+-g[0-9a-f]{5,}"` so it requires a git sha suffix (which BtbN always has but a hypothetical stable `N-5` release would not). Added `test_parse_ffmpeg_version_n_without_git_sha_rejected` that asserts `"ffmpeg version N-5 Copyright ..."` returns `None` (not the sentinel), locking the behavior in.

### WR-09: `check_disk_space_floor` test asserts only non-empty detail, not the actual failure mode

**Files modified:** `tests/unit/test_doctor_training.py`
**Commit:** ec2caa8
**Applied fix:** Added a substring assertion `"does not exist" in result.detail or "not found" in result.detail.lower()` to `test_check_disk_space_floor_missing_path`. A regression that swaps the FileNotFoundError and PermissionError branches would now be caught. (`test_check_disk_space_floor_permission_error` already asserts `"permission" in result.detail.lower()` — left unchanged.)

## Skipped Issues

### WR-03: BtbN ffmpeg tarball downloaded without integrity verification

**File:** `scripts/setup_pod.sh:29` and `:149-158`
**Reason:** Cannot autonomously determine a safe pinned release URL + known SHA256. The fix requires (a) choosing a specific dated BtbN `autobuild-*` release tag and (b) fetching + recording its real SHA256 — both of which need network access to github.com and a conscious pin choice with rollforward implications. Applying a placeholder SHA256 would break cold installs; leaving the URL at `latest` with an unverified sha256sum would add false security. Per the fixer instructions, skipped with explicit reason rather than patched with a guess.
**Original issue:** The script `wget`s `https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz` (mutable `latest` tag) and installs the binaries into `/usr/local/bin/ffmpeg` without checksum or signature verification. A compromised release asset or MitM on a misconfigured pod would install a trojaned ffmpeg invoked by every subsequent preprocessing run. Needs a dated release tag + SHA256 pin; V2 option is switching to a signed source build or an apt PPA.

---

_Fixed: 2026-04-09_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
