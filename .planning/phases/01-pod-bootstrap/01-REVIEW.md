---
status: issues
critical: 0
warning: 9
info: 4
depth: standard
files_reviewed_list:
  - scripts/setup_pod.sh
  - src/doctor.py
  - tests/unit/test_doctor.py
  - tests/unit/test_doctor_training.py
  - .gitignore
---

# Phase 01 Code Review

## Summary

Phase 01 landed two tightly-scoped plans: training-set doctor checks (Plan 01) and the pod bootstrap script (Plan 02). Overall quality is high — `set -euo pipefail`, probe-and-skip discipline, a clean re-exec+tee pattern, CheckResult never-raise contract, and a respectable 17-test mock suite for the new doctor functions. Verified on a real pod (warm re-run at ~26s, cold run green), with the five in-plan deviations documented transparently in the summary.

No critical issues. The notable findings are (a) a narrow `subprocess.PermissionError` leak in `check_mise` that breaks the never-raise contract for an edge case, (b) two `stat()`/`iterdir()` sites in the new training checks that are not wrapped in try/except and therefore can raise `PermissionError`/`OSError`, and (c) a handful of shell-side fragilities in `setup_pod.sh` (unquoted `stat` paths under `(( ))`, un-verified integrity of the BtbN tarball download, probe staleness coupling to a duplicated RVC commit pin). Everything else is stylistic or cosmetic. Nothing blocks phase sign-off.

## Findings

## Critical Issues

_None._

## Warnings

### WR-01: `check_mise` does not catch `PermissionError` — violates never-raise contract

**File:** `src/doctor.py:94-114`
**Issue:** `check_mise` wraps `subprocess.run(["mise", "--version"], ...)` in `try/except FileNotFoundError`. If a `mise` entry exists on `PATH` but is not executable by the current user (e.g., permissions stripped on a shared homedir, broken NFS mount, setuid mismatch), `subprocess.run` raises `PermissionError`, not `FileNotFoundError`. That escapes the catch and crashes the doctor run — violating the documented "every check function returns one instance and never raises" contract in CLAUDE.md. Same issue applies to an `OSError` subclass for weird fs states.

The problem is particularly relevant because the whole point of the soft-mise rework was to stop breaking `setup_rvc.sh`'s pre-flight on pods. A broken-permissions `mise` shim on a laptop path would now kill every doctor invocation instead of soft-failing.

**Fix:**
```python
try:
    proc = subprocess.run(
        ["mise", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
except FileNotFoundError:
    return CheckResult(
        name="mise",
        ok=True,
        detail="not installed (optional — only needed on dev laptops)",
    )
except (PermissionError, OSError) as exc:
    return CheckResult(
        name="mise",
        ok=False,
        detail=f"cannot execute mise: {exc}",
        fix_hint="Check mise binary permissions or reinstall: https://mise.jdx.dev/",
    )
```

Mirror the same pattern in `check_ffmpeg` and `check_ffmpeg_filters` for consistency (they have the same FileNotFoundError-only guard and the same latent bug).

### WR-02: `check_hubert_base` and `check_rvc_mute_refs` can raise on permission/IO errors

**File:** `src/doctor.py:494-512` and `src/doctor.py:474-491`
**Issue:** Both new training checks violate the never-raise contract on rare-but-real failure modes:

- `check_hubert_base` calls `hubert.stat().st_size` with no guard. On a filesystem with a dangling mount, broken symlink, or `EACCES` on the parent dir, `stat()` raises `PermissionError`/`OSError`. The existence check `hubert.exists()` swallows those (returns False on permission error), but the second `stat()` call on the happy path is unguarded.
- `check_rvc_mute_refs` calls `mute_dir.iterdir()` and `any(mute_dir.iterdir())`. If the directory is non-readable (`0o000`), `iterdir()` raises `PermissionError`.

Both propagate out of the check function, breaking the pre-flight table and leaving the user with a traceback instead of an actionable fix_hint. `check_disk_space_floor` does this correctly (wraps `shutil.disk_usage` in try/except for both `FileNotFoundError` and `PermissionError`) — the other two checks should mirror that.

**Fix:**
```python
def check_hubert_base() -> CheckResult:
    hubert = RVC_DIR / "assets" / "hubert" / "hubert_base.pt"
    try:
        if not hubert.exists():
            return CheckResult(name="hubert_base.pt", ok=False, ...)
        size = hubert.stat().st_size
    except OSError as exc:
        return CheckResult(
            name="hubert_base.pt",
            ok=False,
            detail=f"cannot stat {hubert}: {exc}",
            fix_hint="Check filesystem permissions; re-run ./scripts/setup_rvc.sh --force if corrupt.",
        )
    ...
```

Add matching try/except around the `mute_dir.is_dir()` + `iterdir()` block in `check_rvc_mute_refs`. Add tests mirroring `test_check_disk_space_floor_permission_error` for both functions.

### WR-03: BtbN ffmpeg tarball downloaded without integrity verification

**File:** `scripts/setup_pod.sh:29` and `:149-158`
**Issue:** The script `wget`s `https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz` and installs the extracted binaries into `/usr/local/bin/ffmpeg` and `/usr/local/bin/ffprobe`. The URL points at the `latest` release tag, which is mutable (updated nightly by BtbN's CI), and there is no checksum or signature verification. A compromised release asset or a MitM on a misconfigured pod would install a trojaned ffmpeg binary that gets invoked by every subsequent preprocessing run.

The summary notes this was accepted as "same class of trust as the CUDA keyring from NVIDIA", which is partially true — both are HTTPS downloads from trusted upstreams. But the CUDA keyring is a `.deb` that dpkg verifies against NVIDIA's signing key, while the BtbN tarball is fully unsigned and the URL is intentionally a moving target. The security posture is strictly weaker than for `cuda-keyring_1.1-1_all.deb`.

**Fix:** For V1 (pragmatic), at minimum pin to a dated release tag instead of `latest`:
```bash
FFMPEG_STATIC_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-04-09-12-30/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz"
```
and add a SHA256 check:
```bash
EXPECTED_SHA256="..."
echo "${EXPECTED_SHA256}  ${FFMPEG_TARBALL}" | sha256sum -c -
```

For V2, switch to apt's `ppa:savoury1/ffmpeg6` or build from `ffmpeg.org` signed source tarballs. The plan's T-02-08 deferral covers weight-file hash pinning; this finding extends the same deferral rationale to the ffmpeg binary itself and recommends promoting it off the V2 backlog given the higher blast radius (every future pipeline invocation runs the binary, not just a one-time download).

### WR-04: `_check_size` uses unquoted arithmetic comparison on `stat` output that can legitimately be empty

**File:** `scripts/setup_pod.sh:355-360`
**Issue:** `size=$(stat -c '%s' "$path")` has no `|| echo 0` fallback (unlike the sibling `_rvc_already_provisioned` probe at line 324). If `stat` fails for any reason after the `-f` existence check passes (e.g., a race with a concurrent cleanup tool, or a dangling symlink that existed a moment ago), `size` becomes empty and `(( size < min_bytes ))` triggers a bash arithmetic syntax error which, under `set -e`, kills the script mid-layer with no actionable error.

**Fix:** Mirror the `_rvc_already_provisioned` pattern:
```bash
size=$(stat -c '%s' "$path" 2>/dev/null || echo 0)
```

### WR-05: `size=$(stat ...)` local variable leakage in `_rvc_already_provisioned`

**File:** `scripts/setup_pod.sh:314-326`
**Issue:** `local path size min` is declared, but the loop variable `entry` is not. It leaks into the caller scope on return. Cosmetic but inconsistent with the rest of the function's hygiene.

**Fix:** Add `local entry` to the `local` declaration line.

### WR-06: `_ffmpeg_ok` embeds `ffmpeg -version` inside an `echo` under `set -o pipefail`

**File:** `scripts/setup_pod.sh:142`
**Issue:** `echo "ffmpeg already satisfies >=5.0 with required filters ($(ffmpeg -version | head -1))"` — the `$(ffmpeg -version | head -1)` runs a pipeline inside a command substitution under `set -o pipefail`. `head -1` closes the pipe after one line, ffmpeg receives SIGPIPE and exits 141. In bash, a failing command substitution does not trigger `set -e` in a simple command like `echo`, so this works today. But it is fragile: a future bash version (or different shell) could propagate the failure, and it masks any real ffmpeg error in the detail string.

**Fix:** Capture the version before the echo and sanitize:
```bash
ver_line="$(ffmpeg -version 2>/dev/null | head -1 || true)"
echo "ffmpeg already satisfies >=5.0 with required filters (${ver_line})"
```
Apply the same pattern to the matching `echo "ffmpeg OK: $(ffmpeg -version | head -1)"` on line 168.

### WR-07: `RVC_COMMIT_PIN` is duplicated across `setup_rvc.sh` and `setup_pod.sh`

**File:** `scripts/setup_pod.sh:299`
**Issue:** The pinned commit hash `7ef19867780cf703841ebafb565a4e47d1ea86ff` is duplicated in `setup_pod.sh` (for the skip probe) and in `setup_rvc.sh` (which is off-limits for modification per BOOT-05). If the pin ever changes in `setup_rvc.sh`, `setup_pod.sh`'s `_rvc_already_provisioned` probe will silently fail forever on pods provisioned with the new pin, always falling through to the (re)delegation path. That's fail-safe, not fail-dangerous — but it silently negates the warm-run perf optimization and there is no comment warning future maintainers of the coupling.

**Fix:** Add a comment next to `RVC_COMMIT_PIN=` pointing at the corresponding line in `setup_rvc.sh` and noting that stale pins degrade to "always delegate" but never break the cold-install path. Alternatively, source the pin from a shared file (`scripts/rvc-commit.txt`) that both scripts read — but that's V2 scope.

### WR-08: Nightly ffmpeg regex `N-\d+` could false-match future stable version schemes

**File:** `src/doctor.py:64`
**Issue:** `re.search(r"ffmpeg version N-\d+", output)` is intentionally loose and works today because stable releases are `X.Y.Z`. However, if FFmpeg upstream ever adopts a `N-<number>` naming convention for stable releases (unlikely but not impossible — nothing in semver forbids it), any such release would be silently mapped to the `(9999, 0, 0)` sentinel and bypass the version floor entirely.

Low because (a) the stable-release regex is tried first and would match a properly-versioned release, (b) the reality-check is that only BtbN nightly builds currently use this format. Flagging for visibility so the tradeoff is documented.

**Fix:** Tighten the nightly regex to also require a git sha component, which distinguishes BtbN's scheme from a hypothetical stable one:
```python
nightly = re.search(r"ffmpeg version N-\d+-g[0-9a-f]{5,}", output)
```
And add a test for `"ffmpeg version N-5 Copyright"` returning `None` (not the sentinel) to lock the behavior in.

### WR-09: `check_disk_space_floor` test asserts only non-empty detail, not the actual failure mode

**File:** `tests/unit/test_doctor_training.py:56-60`
**Issue:** `test_check_disk_space_floor_missing_path` asserts `result.detail` (truthy) but does not assert the string contains "does not exist" or similar. A regression that swaps the two except branches (FileNotFoundError → permission detail, PermissionError → missing detail) would not be caught by this test.

**Fix:** Add a substring assertion:
```python
assert "does not exist" in result.detail or "not found" in result.detail.lower()
```
Same for `test_check_disk_space_floor_permission_error` (currently checks `"permission"` — already good).

## Info

### IN-01: `check_mise` is silently skipped from the `--training` set, undocumented in the CLI

**File:** `src/doctor.py:581-597`
**Issue:** Per D-09, `check_mise` is deliberately excluded from the `--training` composition. This is correct behavior but not discoverable — the `--training` help text says "Run full training pre-flight set" and a user comparing output to `--system-only` would not know mise was deliberately dropped. A one-line comment inside the `elif training:` block would eliminate the "why is mise missing here?" future grep.

**Fix:** Add:
```python
elif training:
    # check_mise deliberately excluded — pods acquire Python via distro PATH, not mise (D-09).
    selected = [...]
```

### IN-02: `_make_sparse_file` assumes a sparse-file-capable filesystem

**File:** `tests/unit/test_doctor_training.py:20-26`
**Issue:** The `seek(size-1); write(b"\x00")` trick produces a true sparse file on ext4/tmpfs/xfs, but on FAT32, NTFS-without-sparse, or some CI tmpfs configurations it allocates real blocks — meaning the test would transiently write 100 MB to tmp_path. Not a correctness issue; a portability/runtime concern for future CI environments. Worth a comment so contributors don't get surprised.

**Fix:** One-line docstring clarification:
```python
def _make_sparse_file(path: Path, size: int) -> None:
    """Create a sparse file of `size` bytes. Assumes a sparse-capable FS
    (ext4/tmpfs/xfs); on FAT/NTFS the file will allocate real blocks."""
```

### IN-03: `setup_pod.sh` OS detection treats `source /etc/os-release` as unconditionally safe

**File:** `scripts/setup_pod.sh:68-69`
**Issue:** `source /etc/os-release` runs the file as shell. On a standard Ubuntu image this is fine (the file is dpkg-owned and root-written), but if the pod image is unusual and /etc/os-release is writable by non-root or has been tampered with, arbitrary code executes as root. The shellcheck `SC1091` disable is for the unknown-source warning, not the trust warning. Standard bootstrap-script tradeoff; flagged for awareness.

**Fix:** No action recommended for V1. For V2, parse the two fields explicitly without sourcing:
```bash
VERSION_ID=$(awk -F= '$1=="VERSION_ID"{gsub(/"/,"",$2); print $2}' /etc/os-release)
```

### IN-04: Minor: `for entry in \` continuation list is harder to audit than an array

**File:** `scripts/setup_pod.sh:315-320`
**Issue:** The `for entry in "a:1" "b:2" ... ; do ... done` pattern with colon-separated path:min_bytes tuples is compact but loses IDE hover support and makes it easy to typo a floor. Cosmetic.

**Fix:** Use a parallel arrays or a heredoc-driven loop:
```bash
local -a weights=(
  "$RVC_DIR/assets/hubert/hubert_base.pt" 100000000
  "$RVC_DIR/assets/rmvpe/rmvpe.pt"        100000000
  ...
)
```

## Files Reviewed

- scripts/setup_pod.sh (382 lines) — 6 findings (1 WR-high-impact, 4 WR, 2 IN)
- src/doctor.py (609 lines) — 4 findings (2 WR-high-impact, 1 WR, 1 IN)
- tests/unit/test_doctor.py (188 lines) — 0 findings
- tests/unit/test_doctor_training.py (197 lines) — 2 findings (1 WR, 1 IN)
- .gitignore (32 lines) — 0 findings

---

_Reviewed: 2026-04-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
