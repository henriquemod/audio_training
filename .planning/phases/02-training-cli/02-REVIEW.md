---
phase: 02-training-cli
reviewed: 2026-04-09T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - src/train.py
  - src/doctor.py
  - tests/unit/test_train.py
  - tests/unit/test_train_cli.py
  - tests/unit/test_train_runner.py
  - tests/unit/test_doctor.py
findings:
  critical: 0
  warning: 4
  info: 6
  total: 10
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-04-09
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Phase 2 adds an RVC training CLI (`src/train.py`) and the supporting doctor checks
(`check_pretrained_v2_weights`, `check_training_dataset_nonempty`,
`check_rvc_mute_refs`, `check_hubert_base`, `check_disk_space_floor`,
`check_gpu_vram_floor`, `run_training_checks`). The hard architectural constraints
are honored:

- No `shell=True` and no bare-string `subprocess.run("...")` invocations.
- No `torch` / `fairseq` / `faiss` imports in `src/train.py` (asserted by
  `test_no_rvc_imports_in_train_module` and visually confirmed).
- Pure-arg-builders (`build_rvc_preprocess_cmd`, `build_rvc_extract_f0_cmd`,
  `build_rvc_extract_feature_cmd`, `build_rvc_train_cmd`) follow the
  `src/generate.py:build_rvc_subprocess_cmd` pattern and are unit-tested.
- Sentinel-probe resume is implemented per stage (`stage_1_is_done` ..
  `stage_4_is_done`) and exercised by the runner tests.
- Experiment names are validated against `^[a-zA-Z0-9_-]{1,64}$`, blocking
  path traversal.

The findings below are quality / correctness concerns. None are critical
security issues. The most significant is **WR-01**, which describes a latent
filelist bug where mute-row generation can write rows that reference files
that do not exist on disk, causing Stage 4 to crash mid-training instead of
failing fast in pre-flight.

## Warnings

### WR-01: `_write_filelist` may emit mute rows pointing at nonexistent files

**File:** `src/train.py:478-497`

**Issue:** The mute-row loop iterates `for _ in mute_files:` but emits the
**same hardcoded row** every iteration, referencing
`{mute_root}/0_gt_wavs/mute{sr_str}.wav`,
`{mute_root}/3_feature{fea_dim}/mute.npy`,
`{mute_root}/2a_f0/mute.wav.npy`, and `{mute_root}/2b-f0nsf/mute.wav.npy`.
Two problems:

1. **File-existence is never verified.** `check_rvc_mute_refs` only asserts
   that `rvc/logs/mute/` exists and is non-empty. Nothing verifies that the
   four required files (one per pipe-separated column) actually exist for the
   chosen `sr_str` and `fea_dim`. If RVC's mute tree is partially populated
   (e.g. only `0_gt_wavs/mute40k.wav` ships and `2a_f0/mute.wav.npy` does
   not), every mute row references at least one non-existent path. RVC
   `train.py` will then crash mid-Stage-4 with a FileNotFoundError instead
   of being caught by the doctor pre-flight.

2. **Duplicate rows.** When multiple mute candidates are found by
   `mute_gt.glob("*.wav")` (e.g. `mute40k.wav`, `mute48k.wav`), the loop
   writes `len(mute_files)` **identical** rows. The iteration variable `_`
   is discarded; `mute{sr_str}.wav` is hardcoded inside the loop body. This
   either duplicates training samples (skewing the distribution) or — if
   `mute_files` is `[mute48k.wav]` because no `*40k*.wav` matched and the
   fallback `mute_gt.glob("*.wav")` returned the only file present — emits
   a row that points at the non-existent `mute40k.wav`.

**Fix:** Verify each referenced file exists before emitting the row, and
emit at most one mute row (or one per *valid* entry, not per loop counter):

```python
mute_root = RVC_DIR / "logs" / "mute"
mute_paths = {
    "gt":   mute_root / "0_gt_wavs"        / f"mute{sr_str}.wav",
    "feat": mute_root / f"3_feature{fea_dim}" / "mute.npy",
    "f0":   mute_root / "2a_f0"            / "mute.wav.npy",
    "f0n":  mute_root / "2b-f0nsf"         / "mute.wav.npy",
}
required = ("gt", "feat", "f0", "f0n") if if_f0 else ("gt", "feat")
if all(mute_paths[k].exists() for k in required):
    if if_f0:
        lines.append(
            f"{mute_paths['gt']}|{mute_paths['feat']}|"
            f"{mute_paths['f0']}|{mute_paths['f0n']}|0"
        )
    else:
        lines.append(f"{mute_paths['gt']}|{mute_paths['feat']}|0")
```

Pair this with a stronger `check_rvc_mute_refs` (or a new training-mode check)
that verifies the same set of files for the chosen `sample_rate` / `version`.

---

### WR-02: `_run_stage_streamed` does not terminate the child on KeyboardInterrupt

**File:** `src/train.py:640-658`

**Issue:** The `try/finally` block closes the stdout pipe and calls
`proc.wait()`, but does not call `proc.terminate()` / `proc.kill()` if the
parent receives `SIGINT` (Ctrl+C) or `SIGTERM`. On a billing GPU pod, a
KeyboardInterrupt during Stage 4 will:

1. Raise `KeyboardInterrupt` out of the `for line in iter(...)` loop.
2. Hit the `finally` block, which calls `proc.stdout.close()` then
   `proc.wait()` — but `proc.wait()` will block indefinitely waiting for the
   already-running RVC training subprocess that nobody told to stop.
3. The user's second Ctrl+C interrupts `proc.wait()` and propagates, leaving
   the RVC subprocess as a zombie/orphan still consuming GPU memory.

Combined with the project's "resumable" guarantee, this means a normal
"interrupt and re-run" cycle can leave a stale RVC process holding the GPU,
causing the next invocation's Stage 4 to OOM.

**Fix:** Trap `BaseException` (not just `Exception`) in the runner and
terminate the child:

```python
try:
    for line in iter(proc.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
        logf.write(line)
except BaseException:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    raise
finally:
    if proc.stdout is not None:
        proc.stdout.close()
proc.wait()
```

---

### WR-03: Numeric override flags accept negative / zero values without validation

**File:** `src/train.py:887-924` (`_validate_cli_flags`) and `src/train.py:968-986` (CLI options)

**Issue:** `--epochs`, `--batch-size`, `--save-every`, `--num-procs`, and
`--gpus` are all accepted as-is by `_validate_cli_flags`. Nothing rejects
`--epochs 0`, `--epochs -1`, `--batch-size 0`, `--save-every 0`,
`--num-procs 0`, or `--num-procs -4`. Several of these will cause
non-obvious failures inside the RVC subprocess (divide-by-zero on
`epochs / save_every`, infinite loop, or silent hang) instead of being
caught at the CLI boundary as exit code 2.

The doctor checks (`check_gpu_vram_floor`, `check_disk_space_floor`) catch
**system** misconfiguration but not **flag** misconfiguration.

**Fix:** Add bounds checks in `_validate_cli_flags`:

```python
if epochs is not None and epochs < 1:
    return f"invalid --epochs {epochs}: must be >= 1"
if batch_size is not None and batch_size < 1:
    return f"invalid --batch-size {batch_size}: must be >= 1"
if save_every is not None and save_every < 1:
    return f"invalid --save-every {save_every}: must be >= 1"
if num_procs < 1:
    return f"invalid --num-procs {num_procs}: must be >= 1"
```

Pass `epochs`, `batch_size`, `save_every`, `num_procs` into
`_validate_cli_flags` (currently it only validates `experiment_name`,
`sample_rate`, `rvc_version`, `f0_method`, `preset`).

---

### WR-04: `check_pretrained_v2_weights` accepts (v1, if_f0=True) but RVC ships no f0G/f0D under `pretrained/`

**File:** `src/doctor.py:566-628`

**Issue:** `check_pretrained_v2_weights` builds the path
`assets/pretrained/f0G{sr_str}.pth` for `version="v1", if_f0=True`. Upstream
RVC only ships `f0G/f0D` under `pretrained_v2/`; the v1 `pretrained/` directory
ships only the non-f0 `G{sr}.pth` / `D{sr}.pth` pair. A user passing
`--rvc-version v1 --f0-method rmvpe` will fail this check with a
"missing: f0G40k.pth" message that doesn't reflect the real cause: that
combination simply isn't shipped.

`resolve_pretrained_paths` in `src/train.py` has the same issue and the
runtime will exit code 1 from doctor pre-flight (not the worst outcome —
fails fast — but the error message is misleading).

There is no test for `check_pretrained_v2_weights(40000, "v1", if_f0=True)`
in `test_doctor.py`; the v1 path is only tested with `if_f0=False`
(`test_check_pretrained_v1_no_f0_uses_pretrained_dir`).

**Fix:** Either reject the combination explicitly in `_validate_cli_flags`
(symmetry with the existing v1+32k rejection):

```python
if rvc_version == "v1" and f0_method != "none":
    return (
        "invalid combination: --rvc-version v1 with f0 methods is not "
        "supported by upstream RVC. Use --rvc-version v2."
    )
```

…or have `check_pretrained_v2_weights` produce a clearer `fix_hint` for the
v1+f0 case ("RVC does not ship f0 pretrained weights for v1 — use --rvc-version v2").
Add a regression test for the rejected combination either way.

## Info

### IN-01: `assert proc.stdout is not None` is stripped under `python -O`

**File:** `src/train.py:649`

**Issue:** Bare `assert` statements are removed when CPython runs with `-O`.
While `python -O` is unlikely on a training pod, defensive code should not
rely on `assert` for runtime invariants.

**Fix:** Replace with an explicit check or `typing.cast`:

```python
if proc.stdout is None:
    raise RuntimeError("Popen returned no stdout pipe")
```

---

### IN-02: `_tail_file` has an unreachable `UnicodeDecodeError` handler

**File:** `src/train.py:595-598`

**Issue:** `data.decode("utf-8", errors="replace")` cannot raise
`UnicodeDecodeError` because `errors="replace"` substitutes invalid bytes.
The `except UnicodeDecodeError: return ""` block is dead code and a test
(`test_tail_file_handles_binary_garbage`) confirms binary input is replaced
not raised.

**Fix:** Remove the dead try/except, or handle a different concrete error
that can actually occur:

```python
text = data.decode("utf-8", errors="replace")
lines = text.splitlines()
return "\n".join(lines[-n:])
```

---

### IN-03: `if_f0 = f0_method != "none"` is always True (dead branch)

**File:** `src/train.py:1012`

**Issue:** `"none"` is not a member of `VALID_F0_METHODS`, so by the time
`_validate_cli_flags` returns, `f0_method != "none"` is tautologically True.
The comment says "placeholder for future 'none'", which is fine — but the
expression silently misleads code readers, and no test exercises an `if_f0=False`
end-to-end path through `main()`.

**Fix:** Make the placeholder explicit and document the contract, or just hardcode
`if_f0 = True` until "none" is actually a valid method:

```python
# RVC's "none" pitch mode is not exposed via --f0-method yet (Phase 3+).
# All currently-valid f0_method values are pitch-aware.
if_f0 = True
```

---

### IN-04: `doctor --training` always uses `dataset/processed` regardless of caller

**File:** `src/doctor.py:783-794`

**Issue:** When the user runs `python src/doctor.py --training`, the
`default_dataset = PROJECT_ROOT / "dataset" / "processed"` is hardcoded.
On a fresh pod where the dataset lives elsewhere (e.g. `/workspace/clips`),
`check_training_dataset_nonempty` will always FAIL the table — but the
failure says "not found: .../dataset/processed", not "tell me where it is".

This is mitigated by the fact that `src/train.py` re-runs the same checks
with the user's `--dataset-dir`, but `doctor --training` is documented as
the standalone pre-flight oracle. A user calling it for sanity will get a
confusing failure.

**Fix:** Add a `--dataset-dir` flag to `doctor.main` and forward it to the
`run_training_checks` call when `--training` is set, defaulting to
`PROJECT_ROOT / "dataset" / "processed"` only when not provided.

---

### IN-05: `_validate_cli_flags` lambda wrapping pattern in `doctor.main` is fragile

**File:** `src/doctor.py:794`

**Issue:** `selected = [lambda r=r: r for r in precomputed]` wraps already-evaluated
`CheckResult`s as zero-arg callables to fit `_run_checks`'s `check_fn()` calling
convention. This works but breaks the intended contract that elements of
`selected` are *check functions* (lazily evaluated, side-effect-free at definition
time). Future maintenance that introduces logging at call time, or any caller
that introspects function names, will see opaque lambdas.

**Fix:** Either:
- Add a `_run_results(results: list[CheckResult]) -> bool` helper that takes
  pre-evaluated results, and have `--training` call it directly; or
- Have `run_training_checks` return a `list[Callable[[], CheckResult]]` so the
  composition stays uniform.

---

### IN-06: `count_dataset_inputs` swallows `PermissionError` only via `FileNotFoundError`

**File:** `src/train.py:151-158`

**Issue:** `count_dataset_inputs` catches `FileNotFoundError` and returns 0,
but if the dataset directory exists and is unreadable (`PermissionError`,
`NotADirectoryError`), the call propagates the exception out of `run_pipeline`
unhandled, becoming an uncaught traceback at the typer layer (no
context-tagged error message, no exit-code 1 mapping).

`check_training_dataset_nonempty` already catches `OSError` defensively
(doctor.py:662), so the doctor pre-flight shields most cases — but if the
permission flips between pre-flight and `run_pipeline` (race window), the
training command crashes ugly.

**Fix:** Broaden the catch:

```python
try:
    return sum(...)
except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
    return 0
```

---

_Reviewed: 2026-04-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
