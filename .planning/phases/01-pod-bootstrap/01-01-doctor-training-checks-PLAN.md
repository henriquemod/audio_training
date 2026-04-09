---
phase: 01-pod-bootstrap
plan: 01
name: doctor-training-checks
type: execute
wave: 1
depends_on: []
files_modified:
  - src/doctor.py
  - tests/unit/test_doctor_training.py
requirements: [BOOT-09, BOOT-10]
autonomous: true
tags: [bootstrap, doctor, gpu, disk, testing]

must_haves:
  truths:
    - "`python src/doctor.py --training` runs and exits 0 on a healthy pod (exit 1 on any missing piece)"
    - "`check_disk_space_floor(PROJECT_ROOT, 20)` returns ok when >= 20 GiB free, not ok otherwise, without ever raising"
    - "`check_gpu_vram_floor(12)` returns ok when the largest visible GPU has >= 12 GiB VRAM, not ok otherwise, without ever raising"
    - "`check_rvc_mute_refs()` and `check_hubert_base()` exist with minimal bodies (Phase 2 extends them)"
    - "Unit tests for the four new functions pass under `.venv/bin/pytest tests/unit/test_doctor_training.py -q` without a real GPU"
  artifacts:
    - path: "src/doctor.py"
      provides: "New check functions + --training flag composition"
      contains: "def check_disk_space_floor"
    - path: "src/doctor.py"
      provides: "VRAM floor check via nvidia-smi subprocess"
      contains: "def check_gpu_vram_floor"
    - path: "src/doctor.py"
      provides: "Foreshadowed Phase 2 plumbing"
      contains: "def check_rvc_mute_refs"
    - path: "src/doctor.py"
      provides: "Foreshadowed Phase 2 plumbing"
      contains: "def check_hubert_base"
    - path: "src/doctor.py"
      provides: "--training CLI flag"
      contains: "--training"
    - path: "tests/unit/test_doctor_training.py"
      provides: "Mocked unit tests for the four new checks"
      min_lines: 80
  key_links:
    - from: "src/doctor.py::main --training path"
      to: "check_disk_space_floor, check_gpu_vram_floor, check_rvc_mute_refs, check_hubert_base"
      via: "selected checks list"
      pattern: "check_disk_space_floor.*check_gpu_vram_floor"
    - from: "tests/unit/test_doctor_training.py"
      to: "src.doctor"
      via: "from src.doctor import ..."
      pattern: "from src\\.doctor import"
---

<objective>
Add two new doctor checks (`check_disk_space_floor`, `check_gpu_vram_floor`), two foreshadowed Phase 2 plumbing checks (`check_rvc_mute_refs`, `check_hubert_base`), and a `--training` CLI flag that composes the full training pre-flight set. Cover all new Python with unit tests using the project's established `unittest.mock.patch` style.

Purpose: BOOT-09 and BOOT-10. The `scripts/setup_pod.sh` script in Plan 02 will call `python src/doctor.py --training` as its final verification layer — so this plan MUST land first and MUST export a working `--training` flag.

Output: Extended `src/doctor.py` + new `tests/unit/test_doctor_training.py`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/01-pod-bootstrap/01-CONTEXT.md
@.planning/phases/01-pod-bootstrap/01-RESEARCH.md
@src/doctor.py
@tests/unit/test_doctor.py

<interfaces>
Relevant existing symbols in `src/doctor.py` (already imported/used — reuse, do not duplicate):

```python
# Line 22-26
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RVC_DIR = PROJECT_ROOT / "rvc"
RVC_VENV_PYTHON = RVC_DIR / ".venv" / "bin" / "python"

# Line 33-41
@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    fix_hint: str = ""
Check = CheckResult  # alias for test imports

# Line 189 — reference pattern for new check_gpu_vram_floor
def check_nvidia_smi() -> CheckResult: ...

# Line 317 — REUSE as-is in --training set, do NOT rewrite
def check_rvc_torch_cuda() -> CheckResult: ...

# Line 382-424 — main() typer command to extend with --training flag
```

Existing test style from `tests/unit/test_doctor.py` (mirror exactly):

```python
from unittest.mock import patch
from src.doctor import check_ffmpeg  # import target under test

def test_check_ffmpeg_present_and_recent():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_output
        result = check_ffmpeg()
    assert result.ok is True
```

Patch target is the global `"subprocess.run"` and `"shutil.disk_usage"` / `"shutil.which"` — NOT the `src.doctor.subprocess.run` call site. `src/doctor.py` does `import subprocess` then `subprocess.run(...)` so global patching works.
</interfaces>

Absolute paths:
- `/home/henrique/Development/train_audio_model/src/doctor.py`
- `/home/henrique/Development/train_audio_model/tests/unit/test_doctor.py`
- `/home/henrique/Development/train_audio_model/tests/unit/test_doctor_training.py` (will create)
- `/home/henrique/Development/train_audio_model/.planning/phases/01-pod-bootstrap/01-RESEARCH.md` §"Concrete Snippets" has the exact function bodies to drop in.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add check_disk_space_floor, check_gpu_vram_floor, check_rvc_mute_refs, check_hubert_base to src/doctor.py</name>
  <files>src/doctor.py</files>

  <read_first>
    - src/doctor.py (full file — add new functions after line 349, before the `---------- CLI ----------` section at line 352)
    - .planning/phases/01-pod-bootstrap/01-RESEARCH.md §"Concrete Snippets" (the `check_gpu_vram_floor` and `check_disk_space_floor` function bodies to lift verbatim)
    - .planning/phases/01-pod-bootstrap/01-CONTEXT.md §Decisions D-09, D-10, D-11 (locked signatures and default floors)
  </read_first>

  <behavior>
    - `check_disk_space_floor(path, min_gb=20)` above floor → ok=True, detail contains `"GiB free"`
    - `check_disk_space_floor(path, min_gb=20)` at exactly 20 GiB free → ok=True (use `>=`, not `>`)
    - `check_disk_space_floor(path, min_gb=20)` below floor → ok=False, detail contains current free-GiB number, fix_hint contains `"20"`
    - `check_disk_space_floor` on nonexistent path → ok=False, no exception raised (catches `FileNotFoundError`)
    - `check_disk_space_floor` on PermissionError → ok=False, no exception raised
    - `check_gpu_vram_floor(12)` single GPU reporting 24564 MiB → ok=True (24.0 GiB >= 12)
    - `check_gpu_vram_floor(12)` multi-GPU with 8192+24564 → ok=True, picks MAX not min/sum
    - `check_gpu_vram_floor(12)` single GPU reporting 8192 MiB → ok=False (8.0 < 12), fix_hint contains `"12"`
    - `check_gpu_vram_floor(12)` when `shutil.which("nvidia-smi")` is None → ok=False, fix_hint mentions "nvidia-smi"
    - `check_gpu_vram_floor(12)` when nvidia-smi exits non-zero → ok=False, detail contains stderr
    - `check_gpu_vram_floor(12)` when stdout is unparseable → ok=False (ValueError handled)
    - `check_rvc_mute_refs()` when `rvc/logs/mute/` dir exists and non-empty → ok=True
    - `check_rvc_mute_refs()` when dir missing → ok=False, fix_hint points at `setup_rvc.sh`
    - `check_hubert_base()` when `rvc/assets/hubert/hubert_base.pt` exists and stat().st_size >= 100_000_000 → ok=True
    - `check_hubert_base()` when file missing → ok=False, fix_hint points at `setup_rvc.sh`
    - `check_hubert_base()` when file exists but < 100_000_000 bytes → ok=False, detail mentions size
  </behavior>

  <action>
Add the four new check functions to `src/doctor.py` in the `# ---------- Project state checks ----------` section (after `check_rvc_torch_cuda` at line 349, before `# ---------- CLI ----------` at line 352). All four follow the existing `CheckResult`-returning, never-raising pattern.

**1. `check_disk_space_floor(path: Path, min_gb: int) -> CheckResult`** — lift verbatim from 01-RESEARCH.md §"`check_disk_space_floor` body" (lines 718-747 of research). Uses `shutil.disk_usage(path).free / (1024 ** 3)` for GiB conversion. Name must be `f"disk space floor ({min_gb} GiB at {path})"`. Comparison: `free_gib < min_gb` → fail (so equality is pass). Catches `FileNotFoundError` and `PermissionError` and returns `ok=False` with a descriptive detail.

**2. `check_gpu_vram_floor(min_gb: int) -> CheckResult`** — lift verbatim from 01-RESEARCH.md §"`check_gpu_vram_floor` body" (lines 658-712 of research). Command: `["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]`. Parses lines as int MiB, picks `max()`, converts MiB→GiB via `/1024`. No torch import, no `rvc/.venv` subprocess. Guard with `shutil.which("nvidia-smi") is None` before calling subprocess. Wrap `int(...)` in try/except ValueError.

**3. `check_rvc_mute_refs() -> CheckResult`** — minimal plumbing (Phase 2 extends). Checks `RVC_DIR / "logs" / "mute"` exists and is non-empty directory:
```python
def check_rvc_mute_refs() -> CheckResult:
    """Verify RVC mute reference files exist. Phase 2 will tighten this."""
    mute_dir = RVC_DIR / "logs" / "mute"
    if not mute_dir.is_dir():
        return CheckResult(
            name="rvc mute refs",
            ok=False,
            detail=f"{mute_dir.relative_to(PROJECT_ROOT)} missing",
            fix_hint="Run ./scripts/setup_rvc.sh --force to re-clone RVC.",
        )
    if not any(mute_dir.iterdir()):
        return CheckResult(
            name="rvc mute refs",
            ok=False,
            detail=f"{mute_dir.relative_to(PROJECT_ROOT)} is empty",
            fix_hint="Run ./scripts/setup_rvc.sh --force.",
        )
    return CheckResult(name="rvc mute refs", ok=True)
```

**4. `check_hubert_base() -> CheckResult`** — existence + size floor. Floor is `100_000_000` bytes (100 MB) per 01-RESEARCH.md §"Pretrained Weight File Sizes":
```python
HUBERT_MIN_BYTES = 100_000_000

def check_hubert_base() -> CheckResult:
    """Verify hubert_base.pt exists and is not a truncated download."""
    hubert = RVC_DIR / "assets" / "hubert" / "hubert_base.pt"
    if not hubert.exists():
        return CheckResult(
            name="hubert_base.pt",
            ok=False,
            detail=f"{hubert.relative_to(PROJECT_ROOT)} missing",
            fix_hint="Run ./scripts/setup_rvc.sh (downloads pretrained weights).",
        )
    size = hubert.stat().st_size
    if size < HUBERT_MIN_BYTES:
        return CheckResult(
            name="hubert_base.pt",
            ok=False,
            detail=f"only {size} bytes (expected >= {HUBERT_MIN_BYTES})",
            fix_hint="Truncated download. Re-run ./scripts/setup_rvc.sh --force.",
        )
    return CheckResult(name="hubert_base.pt", ok=True, detail=f"{size} bytes")
```

Add `HUBERT_MIN_BYTES = 100_000_000` as a module constant near the existing `MIN_FFMPEG_VERSION` block (around line 28). Keep line length ≤ 100 (ruff E rules). Use `from __future__ import annotations` is already at top of file — do not re-add.

Do NOT modify existing functions. Do NOT upgrade pip. Do NOT import torch. Do NOT add new module-level imports beyond what is already present (`shutil`, `subprocess`, `Path` are already imported).

Implements D-09, D-10, D-11 per CONTEXT.md.
  </action>

  <verify>
    <automated>cd /home/henrique/Development/train_audio_model &amp;&amp; .venv/bin/python -c "from src.doctor import check_disk_space_floor, check_gpu_vram_floor, check_rvc_mute_refs, check_hubert_base, HUBERT_MIN_BYTES; print('imports OK'); print('HUBERT_MIN_BYTES=', HUBERT_MIN_BYTES); assert HUBERT_MIN_BYTES == 100_000_000" &amp;&amp; .venv/bin/ruff check src/doctor.py</automated>
  </verify>

  <acceptance_criteria>
    - `grep -n "def check_disk_space_floor" src/doctor.py` returns exactly one line
    - `grep -n "def check_gpu_vram_floor" src/doctor.py` returns exactly one line
    - `grep -n "def check_rvc_mute_refs" src/doctor.py` returns exactly one line
    - `grep -n "def check_hubert_base" src/doctor.py` returns exactly one line
    - `grep -n "HUBERT_MIN_BYTES = 100_000_000" src/doctor.py` returns exactly one line
    - `grep -n "import torch" src/doctor.py` returns zero matches (two-venv boundary)
    - `grep -c "shutil.disk_usage" src/doctor.py` is exactly 1
    - `grep -c "nvidia-smi" src/doctor.py` is 2 or more (existing check_nvidia_smi + new check_gpu_vram_floor)
    - `.venv/bin/ruff check src/doctor.py` exits 0
    - `.venv/bin/python -c "from src.doctor import check_disk_space_floor, check_gpu_vram_floor, check_rvc_mute_refs, check_hubert_base"` exits 0
  </acceptance_criteria>

  <done>
    All four new functions present, importable, never raise on documented edge cases, ruff clean, no torch import, match the exact signatures locked in D-10/D-11.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add --training flag to doctor.py main() and write unit tests</name>
  <files>src/doctor.py, tests/unit/test_doctor_training.py</files>

  <read_first>
    - src/doctor.py lines 352-428 (the `# ---------- CLI ----------` section and `main()` command — the composition point)
    - tests/unit/test_doctor.py (full file — the mocking style to mirror exactly, `with patch("subprocess.run")` etc.)
    - .planning/phases/01-pod-bootstrap/01-RESEARCH.md §"`--training` flag composition in `doctor.py:main`" (exact flag declaration + selected-checks list)
    - .planning/phases/01-pod-bootstrap/01-RESEARCH.md §"Unit-Test Mocking Patterns" (lines 369-455 — exact test bodies to lift for each check)
    - .planning/phases/01-pod-bootstrap/01-CONTEXT.md §D-09 (training set composition), D-15 (test requirements)
  </read_first>

  <behavior>
    - `python src/doctor.py --training` on a healthy machine runs all composed checks and exits 0
    - `python src/doctor.py --training` on a machine missing any check exits 1 and prints the failing check's fix_hint
    - `--training` is mutually coexistent with the existing `--system-only`, `--rvc-only`, `--runtime` flags (selection cascade: if `--training`, use training set; else existing logic)
    - Unit tests: 3 for disk (above/at-threshold/below), 1 for disk missing-path, 3 for VRAM (single above / multi-max-picks / below), 1 for VRAM no-driver, 1 for VRAM unparseable, 2 for mute refs (ok + missing), 2 for hubert (ok + missing/truncated)
    - All tests run without a GPU and without touching the real filesystem beyond `tmp_path`-style fixtures
    - `.venv/bin/pytest tests/unit/test_doctor_training.py -q` exits 0
  </behavior>

  <action>
**Part A — Extend `src/doctor.py:main()` with `--training` flag.**

Add a new typer Option parameter to `main()`:

```python
training: bool = typer.Option(False, "--training", help="Run full training pre-flight set"),
```

Add a new branch in the selection cascade in `main()`. Current cascade is at lines 411-418:

```python
if system_only:
    selected = system_checks
elif rvc_only:
    selected = rvc_checks
elif runtime:
    selected = runtime_checks
else:
    selected = system_checks + rvc_checks + runtime_checks
```

Insert a `--training` branch BEFORE the `else`. Per D-09 and 01-RESEARCH.md §"`--training` flag composition", the training set composes:

```python
elif training:
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
        check_rvc_mute_refs,
        check_hubert_base,
    ]
```

Note per D-09: `check_mise` is deliberately EXCLUDED from the training set (mise is a local-dev convenience, not a pod requirement). Do not add it.

The disk and VRAM checks take arguments, so they're wrapped in zero-arg lambdas to match the `check_fn()` call convention in `_run_checks` (line 367).

**Part B — Create `tests/unit/test_doctor_training.py`.**

New file. Mirror the style of `tests/unit/test_doctor.py` exactly (`from unittest.mock import patch`, module-level test functions, no fixtures unless necessary, patch globals not call sites).

Header:

```python
"""Unit tests for the training pre-flight doctor checks added in Phase 1."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

from src.doctor import (
    HUBERT_MIN_BYTES,
    check_disk_space_floor,
    check_gpu_vram_floor,
    check_hubert_base,
    check_rvc_mute_refs,
)

_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])
```

Write the following test functions. The bodies for disk + VRAM tests are lifted verbatim from 01-RESEARCH.md lines 392-451 (the §"Unit-Test Mocking Patterns" section):

1. `test_check_disk_space_floor_above` — `free=50 * 1024**3`, min_gb=20 → ok=True
2. `test_check_disk_space_floor_at_threshold` — `free=20 * 1024**3`, min_gb=20 → ok=True (>= comparison)
3. `test_check_disk_space_floor_below` — `free=5 * 1024**3`, min_gb=20 → ok=False, `"5"` in detail, `"20"` in fix_hint
4. `test_check_disk_space_floor_missing_path` — `patch("shutil.disk_usage", side_effect=FileNotFoundError)` → ok=False, does not raise
5. `test_check_gpu_vram_floor_single_gpu_above` — stdout `"24564\n"`, min_gb=12 → ok=True
6. `test_check_gpu_vram_floor_multi_gpu_picks_max` — stdout `"8192\n24564\n"`, min_gb=12 → ok=True (max is 24564)
7. `test_check_gpu_vram_floor_below` — stdout `"8192\n"`, min_gb=12 → ok=False, `"12"` in fix_hint
8. `test_check_gpu_vram_floor_no_driver` — `patch("shutil.which", return_value=None)` → ok=False, `"nvidia-smi"` in `(fix_hint + detail).lower()`
9. `test_check_gpu_vram_floor_unparseable_output` — stdout `"not a number\n"`, returncode 0 → ok=False, does not raise
10. `test_check_rvc_mute_refs_present(tmp_path, monkeypatch)` — monkeypatch `src.doctor.RVC_DIR` to tmp_path, create `tmp_path/logs/mute/dummy.wav` → ok=True
11. `test_check_rvc_mute_refs_missing(tmp_path, monkeypatch)` — monkeypatch `src.doctor.RVC_DIR` to tmp_path, no mute dir → ok=False, fix_hint mentions `setup_rvc.sh`
12. `test_check_hubert_base_present(tmp_path, monkeypatch)` — monkeypatch `src.doctor.RVC_DIR` to tmp_path, create `tmp_path/assets/hubert/hubert_base.pt` with exactly `HUBERT_MIN_BYTES` bytes (`(tmp_path/...).write_bytes(b"\\0" * HUBERT_MIN_BYTES)`) → ok=True. Note: 100 MB is large for a test fixture — use `file.seek(HUBERT_MIN_BYTES - 1); file.write(b"\\0")` sparse-file trick via `open(path, "wb")` to avoid allocating 100 MB of memory in tests.
13. `test_check_hubert_base_truncated(tmp_path, monkeypatch)` — create file with 1000 bytes → ok=False, detail mentions `"1000"`
14. `test_check_hubert_base_missing(tmp_path, monkeypatch)` — monkeypatch RVC_DIR to empty tmp_path → ok=False

For monkeypatching module-level `RVC_DIR`, use `monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)`. Do NOT patch `PROJECT_ROOT` — it's used for `relative_to()` in detail strings; override it alongside RVC_DIR if needed (set `monkeypatch.setattr("src.doctor.PROJECT_ROOT", tmp_path)` too so `relative_to` doesn't raise `ValueError`).

**Sparse file helper** for test 12 to avoid writing 100 MB:

```python
def _make_sparse_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        if size > 0:
            f.seek(size - 1)
            f.write(b"\x00")
```

Keep line length ≤ 100. All tests must be pure-Python, no GPU, no network, no real nvidia-smi binary. Patch `"subprocess.run"` and `"shutil.which"` globally (matches existing test_doctor.py style).

Implements D-15.
  </action>

  <verify>
    <automated>cd /home/henrique/Development/train_audio_model &amp;&amp; .venv/bin/ruff check src/doctor.py tests/unit/test_doctor_training.py &amp;&amp; .venv/bin/pytest tests/unit/test_doctor_training.py -x -q &amp;&amp; .venv/bin/python src/doctor.py --help 2>&amp;1 | grep -q -- "--training"</automated>
  </verify>

  <acceptance_criteria>
    - `grep -n '"--training"' src/doctor.py` returns exactly one line (the typer.Option declaration)
    - `grep -n "elif training:" src/doctor.py` returns exactly one line
    - `grep -c "check_disk_space_floor(PROJECT_ROOT, 20)" src/doctor.py` is exactly 1
    - `grep -c "check_gpu_vram_floor(12)" src/doctor.py` is exactly 1
    - `grep -n "check_mise" src/doctor.py` does NOT match inside the training set (mise deliberately excluded per D-09)
    - `.venv/bin/python src/doctor.py --help | grep -- "--training"` returns a match
    - `tests/unit/test_doctor_training.py` exists and imports all four new symbols + `HUBERT_MIN_BYTES`
    - `.venv/bin/pytest tests/unit/test_doctor_training.py -x -q` exits 0 with at least 14 tests collected
    - `.venv/bin/pytest tests/unit/test_doctor_training.py -q -k "vram"` collects at least 5 tests
    - `.venv/bin/pytest tests/unit/test_doctor_training.py -q -k "disk"` collects at least 4 tests
    - `.venv/bin/ruff check src/doctor.py tests/unit/test_doctor_training.py` exits 0
    - `grep -n "import torch" tests/unit/test_doctor_training.py` returns zero matches
    - `.venv/bin/pytest tests/unit/ -q` (full unit suite) exits 0 — no regression in existing tests
  </acceptance_criteria>

  <done>
    `--training` flag is live, composed check set matches D-09, 14+ unit tests pass without a GPU, existing tests still pass, ruff clean.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| test→doctor | Unit tests exercise check functions with mocked subprocess/shutil — no real boundary crossing |
| doctor→nvidia-smi | `check_gpu_vram_floor` shells out to the `nvidia-smi` binary (exists on pod, patched in tests) |
| doctor→filesystem | `check_disk_space_floor`, `check_hubert_base`, `check_rvc_mute_refs` read real filesystem paths |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-01-01 | Tampering | `check_hubert_base` file size check | accept | Size floor is integrity-light, not cryptographic. A 100 MB sentinel file passes. Acceptable because (a) pod is ephemeral, (b) setup_rvc.sh downloads from HTTPS HuggingFace, (c) the actual integrity story for weights is "HTTPS + trusted upstream" — upgrading to SHA256 is V2 scope. Explicitly noted in research A3. |
| T-01-02 | Information Disclosure | `check_disk_space_floor` detail strings include filesystem paths | accept | Paths are project-local (e.g., `/workspace/train_audio_model`) — no secrets. The detail is logged to `setup_pod.log` which is root-owned on the pod and ephemeral. No PII. |
| T-01-03 | Denial of Service | `check_hubert_base` test 12 writing a 100 MB sparse file | mitigate | Use `f.seek(size-1); f.write(b"\x00")` sparse-file trick in the test helper so pytest does not allocate 100 MB of RAM or disk. Filesystem reports the apparent size; no real blocks consumed. |
| T-01-04 | Elevation of Privilege | Unit tests patching `subprocess.run` globally | accept | `unittest.mock.patch("subprocess.run")` matches the existing `test_doctor.py` style. Scope is per-test context manager. Standard practice. |
| T-01-05 | Spoofing | `check_gpu_vram_floor` trusts nvidia-smi stdout | accept | nvidia-smi is a vendor-provided binary installed by the pod image's NVIDIA driver package. If it's compromised, the pod is compromised. No additional verification possible from userland. |

All threats are either mitigated in-plan or explicitly accepted with rationale — no ASVS L1 "high" severity items remain unaddressed.
</threat_model>

<verification>
1. **Autonomous (CI/local):**
   - `.venv/bin/ruff check src/doctor.py tests/unit/test_doctor_training.py` → exits 0
   - `.venv/bin/pytest tests/unit/ -q` → all tests pass (existing + new)
   - `.venv/bin/python src/doctor.py --help | grep -q -- "--training"` → match
   - `.venv/bin/python -c "from src.doctor import check_disk_space_floor, check_gpu_vram_floor, check_rvc_mute_refs, check_hubert_base, HUBERT_MIN_BYTES"` → no error

2. **Deferred to pod-side verification (Plan 02 consumes this):**
   - `python src/doctor.py --training` on a provisioned pod → exits 0
</verification>

<success_criteria>
- `src/doctor.py` gains four new check functions with exact signatures from D-10/D-11 (BOOT-09)
- `--training` CLI flag composes the full training readiness set (BOOT-09)
- `HUBERT_MIN_BYTES = 100_000_000` constant exported for Phase 2 and tests
- 14+ unit tests pass without GPU, mirror existing test_doctor.py mocking style (BOOT-10)
- No `import torch` added anywhere in `src/` (two-venv boundary preserved)
- ruff clean on both modified/new files
- Phase 2's Plan 02 `scripts/setup_pod.sh` can safely invoke `python src/doctor.py --training` as final verification
</success_criteria>

<output>
After completion, create `.planning/phases/01-pod-bootstrap/01-01-SUMMARY.md` covering:
- Which symbols were added to `src/doctor.py`
- Test count and `pytest` output snapshot
- Confirmation that `--training` flag is wired and help text shows it
- Any decisions made under "Claude's Discretion" (e.g., exact error message wording, sparse-file test helper)
- Requirement coverage: BOOT-09 ✓, BOOT-10 ✓
</output>
