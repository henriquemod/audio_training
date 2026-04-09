# Codebase Concerns

**Analysis Date:** 2026-04-09

## Tech Debt

### RVC dependency vendoring via pinned clone (not submodule)

**Issue:** RVC is cloned at a fixed commit (`7ef19867780cf703841ebafb565a4e47d1ea86ff`) and not maintained as a git submodule. This creates sync drift risk.

**Files:** `scripts/setup_rvc.sh` (lines 7-8, 54-73)

**Impact:**
- RVC upstream updates are not automatically tracked
- Pinned commit will eventually become stale (was Nov 24, 2024)
- New RVC releases with critical fixes won't be integrated without manual intervention
- Dependency security updates are developer-driven, not automatic

**Fix approach:**
- Convert to submodule with pinned commit in `.gitmodules` (easier to manage in git workflows)
- OR establish a quarterly review process to check for RVC updates
- Implement CI/CD check to alert when upstream has critical fixes

---

### Fragile RVC dependency chain with multiple pinned versions

**Issue:** RVC's unpinned transitive dependencies require manual pinning at multiple layers to prevent breakage.

**Files:** `scripts/setup_rvc.sh` (lines 85-121)

**Current workarounds:**
- `pip<24.1` pinned (line 91) due to fairseq 0.12.2's legacy PEP 440 metadata
- `gradio_client==0.2.7` pinned (line 111) to match gradio 3.34.0
- `matplotlib==3.7.3` pinned (line 121) because RVC training relies on deprecated `tostring_rgb()` (removed in 3.8+)

**Impact:**
- Each pin adds maintenance burden
- Breaking changes in pinned packages could happen: e.g., numpy 2.0 dropped support for `np.float` (RVC may use it)
- No automated testing against newer versions
- Future Python 3.11/3.12 compatibility may break all these pins

**Fix approach:**
- Periodically test with newer Python versions (3.11, 3.12) in CI
- Document why each pin exists (already done well in comments)
- Consider forking/vendoring the RVC dependency if upstream becomes unmaintained

---

### Cross-venv subprocess isolation incomplete

**Issue:** Two Python environments (`.venv` and `rvc/.venv`) have minimal integration testing. A change to RVC's CLI interface would break `generate.py`.

**Files:**
- `src/generate.py` (lines 98-145, subprocess invocation)
- `scripts/setup_rvc.sh` (RVC venv setup)

**Impact:**
- RVC subprocess command line (`build_rvc_subprocess_cmd()`) is hardcoded; if RVC's `tools/infer_cli.py` changes argument names or adds required args, `generate.py` fails silently until runtime
- No validation that RVC subprocess args are correct until actual inference runs
- Integration test (`tests/integration/test_edge_tts.py`) skips RVC subprocess—only tests Edge-TTS

**Fix approach:**
- Add smoke-test that runs actual RVC subprocess (can be skipped in CI if no GPU)
- Document RVC CLI contract in comments with version pinned
- Add pre-flight check in `doctor.py` for RVC CLI compatibility (test with `--help`)

---

## Known Bugs

### Edge-TTS Microsoft token rotation not future-proofed

**Issue:** edge-tts 7.2.8 may fail with 403 errors if Microsoft rotates API tokens. This is a known issue in the edge-tts library itself.

**Files:**
- `src/generate.py` (lines 84-95, Edge-TTS call)
- `README.md` (line 158, documented as known issue)

**Symptoms:** `Edge-TTS failed: ... 403 Forbidden` after several months of running fine.

**Workaround:** User manually upgrades edge-tts: `.venv/bin/pip install -U edge-tts`

**Fix approach:**
- Add `doctor.py` check for edge-tts version compatibility (query a test URL)
- Pin edge-tts only after testing newer versions work
- Document in code that this is a known fragile upstream dependency

---

### Possible memory leak in preprocess with large audio files

**Issue:** `_slice_with_slicer2()` reads entire audio file into memory via `sf.read()` before slicing.

**Files:** `src/preprocess.py` (lines 115, in `_slice_with_slicer2()`)

**Impact:**
- For very large raw audio files (>2 GB), could exceed available RAM
- No streaming/chunked processing
- Preprocess will silently fail if OOM (process killed by kernel)

**Current state:** README recommends 30-60 minutes of audio, which is ~100-200 MB and acceptable.

**Fix approach:**
- Add memory check in doctor.py for available RAM
- Document minimum RAM requirement in README
- OR implement streaming slicer2 for files >1GB (low priority if audio duration is bounded)

---

## Security Considerations

### Subprocess arguments validated but not escaped

**Issue:** `build_rvc_subprocess_cmd()` accepts user-controlled `model_name` which becomes a filename arg to RVC subprocess.

**Files:** `src/generate.py` (lines 98-145)

**Current protection:** Model name must already exist in `models/` (checked via `check_model_file()`), so an attacker can't create arbitrary filenames. But if a model name contains shell metacharacters (unlikely but possible), they could be passed directly to subprocess.

**Risk level:** LOW (subprocess.run() with list args never shells out, and model names are filesystem-validated)

**Fix approach:**
- Add validation that model_name matches pattern `^[a-zA-Z0-9_-]+$` (already good practice)
- Document this assumption in `build_rvc_subprocess_cmd()` docstring

---

### Environment variable handling for default model/voice

**Issue:** `DEFAULT_MODEL`, `DEFAULT_EDGE_VOICE`, `DEVICE` read from `.env` without validation.

**Files:** `src/generate.py` (lines 57-59)

**Current state:** .env is `.gitignore`-d, so not checked into git. But if .env is accidentally committed with credentials, they'd be exposed.

**Fix approach:**
- Document in README that .env must never be committed
- Use `.env.example` as template (already done)
- Consider adding pre-commit hook to prevent .env commits

---

### Vendored slicer2.py lacks upstream security tracking

**Issue:** `src/slicer2.py` is vendored from audio-slicer 1.0.1 (MIT, 2023). No automated way to know if upstream releases security fixes.

**Files:** `src/slicer2.py` (lines 1-6)

**Impact:** Pure numpy usage so unlikely to have memory-safety bugs, but if audio-slicer ever releases a critical fix, we won't know.

**Fix approach:**
- Add comment with link to upstream repo and date vendored
- Quarterly manual check of upstream for updates
- OR switch to installing audio-slicer as a dependency (adds weight)

---

## Performance Bottlenecks

### RVC inference subprocess blocks synchronously

**Issue:** `generate.py` calls RVC subprocess and waits for it synchronously (lines 315-326).

**Files:** `src/generate.py` (lines 315)

**Impact:**
- If RVC is slow (common on first run, model loading), user sees no feedback for up to 5-10 minutes
- No timeout—if RVC hangs, script hangs indefinitely
- Can't cancel without `^C` (which leaves temp files)

**Current state:** Rich console prints progress messages, but RVC subprocess stderr is silent unless `--verbose` used.

**Fix approach:**
- Add optional timeout flag: `--timeout-minutes 30`
- Stream RVC stdout/stderr in real-time instead of capturing fully
- Show elapsed time every 30 seconds during RVC inference

---

### Entire audio file held in RAM during preprocessing

**Issue:** Preprocessing loads entire normalized audio into memory before slicing (via slicer2).

**Files:** `src/preprocess.py` (lines 115-145)

**Impact:** 100+ MB files require proportional RAM. No chunked/streaming processing.

**Current state:** Fine for 30-60 min of audio (100-200 MB), but would fail on longer recordings.

**Fix approach:**
- Implement streaming window-based slicing if audio >1 GB
- Add progress indicator during slice phase (currently silent)

---

### Dense subprocess error context in generate.py

**Issue:** `generate.py` captures full RVC subprocess stderr but only shows last 20 lines on error (line 319: `_tail(proc.stderr, 20)`).

**Impact:**
- If RVC error requires context from earlier in the log, it's lost
- Hard to debug intermittent failures

**Fix approach:**
- Write full stderr to a temp log file and reference it in error message
- Preserve temp log on failure (already does this with `--keep-intermediate`)

---

## Fragile Areas

### doctor.py check registry is not extensible

**Issue:** Adding a new check requires:
1. Writing a `check_*()` function
2. Manually adding it to `system_checks`, `rvc_checks`, or `runtime_checks` list (line 392-409)
3. Updating CLI options if it's optional

**Files:** `src/doctor.py` (lines 357-424)

**Impact:** Error-prone; easy to add a check function but forget to register it. Maintainers may not notice a missing registration.

**Fix approach:**
- Use a decorator `@register_check(category="system")` to auto-register checks
- Build the check list dynamically at runtime

---

### Preprocess pipeline is all-or-nothing; no incremental mode

**Issue:** `run_preprocess()` wipes entire `dataset/processed/` and reruns all steps on every invocation (lines 168-170).

**Files:** `src/preprocess.py`

**Impact:**
- Adding 1 new audio file to `dataset/raw/` re-processes everything (slow)
- Can't pause and resume preprocessing
- If preprocessing fails on file N, all previous work is lost on next run

**Current state:** README documents this as "idempotent" but doesn't mention the cost.

**Fix approach:**
- Add `--incremental` flag to skip already-processed files
- Check if processed file exists before reconverting
- Maintain manifest of processed files (JSON) to enable resumable operations

---

### slicer2 parameters are hardcoded, not tunable

**Issue:** `Slicer()` instantiation in `_slice_with_slicer2()` (line 116) uses fixed parameters:
- `threshold=-40` (silence detection)
- `min_length=int(min_len_s * 1000)` (uses CLI param, good)
- `min_interval=300` (hardcoded)
- `hop_size=10` (hardcoded)
- `max_sil_kept=500` (hardcoded)

**Files:** `src/preprocess.py` (lines 116-123)

**Impact:** Users can't tune silence detection sensitivity. If audio has:
- Lots of breath sounds → may not slice where desired
- Soft background chatter → may be treated as silence

**Fix approach:**
- Add CLI options `--slicer-threshold`, `--slicer-min-interval`, `--slicer-max-sil` (with good defaults)
- Document what each parameter does

---

### test_preprocess_real.py skips if ffmpeg unavailable

**Issue:** Integration test is skipped silently if ffmpeg missing (lines 13-16).

**Files:** `tests/integration/test_preprocess_real.py`

**Impact:** CI could pass without ever running the integration test. No indication that tests were skipped.

**Fix approach:**
- Fail fast instead of skip: `pytest.fail("ffmpeg required for integration tests")`
- OR make ffmpeg a required CI dependency

---

## Test Coverage Gaps

### RVC subprocess integration never tested

**Issue:** No test calls actual RVC `tools/infer_cli.py`. `tests/integration/test_edge_tts.py` only tests Edge-TTS.

**Files:**
- `src/generate.py` (RVC subprocess, untested)
- `tests/integration/test_edge_tts.py` (only Edge-TTS)

**Risk:** Breaking change in RVC's CLI would not be caught until user runs `generate.py`.

**Fix approach:**
- Add `tests/integration/test_generate_rvc_subprocess.py` (marked with `@pytest.mark.gpu`)
- Mock or run actual RVC inference (use smoke-test model if available)
- At minimum, test that `build_rvc_subprocess_cmd()` produces valid args

---

### Loud norm filter effectiveness not validated

**Issue:** `build_loudnorm_args()` builds ffmpeg args but integration test doesn't verify loudness actually reaches target.

**Files:** `src/preprocess.py` (lines 80-94, loudnorm args)

**Impact:** If ffmpeg loudnorm produces different LUFS than expected, training audio quality degrades silently.

**Fix approach:**
- In integration test, use ffmpeg to measure output LUFS and assert within 1-2 of target
- Document expected loudness behavior in docstring

---

### Edge-TTS network tests are marked but never run

**Issue:** `tests/integration/test_edge_tts.py` exists but is marked `@pytest.mark.network` and excluded by default (pyproject.toml line 47).

**Files:**
- `tests/integration/test_edge_tts.py`
- `pyproject.toml` (line 47: `addopts = "-m 'not network and not gpu'"`)

**Impact:** Network-dependent code path is never tested in CI. Could break silently.

**Fix approach:**
- Run network tests in a separate CI job (or optionally via cron)
- Document in README how to run network tests locally

---

### No test for doctor.py check_nvidia_smi on non-NVIDIA systems

**Issue:** `check_nvidia_smi()` (lines 189-209) returns FAIL if nvidia-smi missing, but test can't run on CPU-only systems.

**Files:**
- `src/doctor.py` (lines 189-209)
- `tests/unit/test_doctor.py`

**Impact:** Function is untested on most developer machines. Could have bugs that only show up in production.

**Fix approach:**
- Test the function with a mocked nvidia-smi (subprocess mock)
- OR test the error path (FileNotFoundError) explicitly

---

## Scaling Limits

### Single GPU assumption

**Issue:** Code assumes CUDA device `cuda:0` (or single device specified). No multi-GPU support.

**Files:**
- `src/generate.py` (line 59: `DEFAULT_DEVICE = os.environ.get("DEVICE", "cuda:0")`)
- `src/doctor.py` (line 189: checks nvidia-smi but not multi-GPU setup)

**Impact:** Users with multiple GPUs can't distribute load. RVC subprocess runs on single GPU only.

**Current state:** README targets RTX 4090 (single premium GPU). Acceptable limitation.

**Fix approach:**
- Document as single-GPU only
- If future versions support multi-GPU, add validation in doctor.py

---

### No batch audio processing

**Issue:** `generate.py` processes one text string at a time. No batch mode to generate multiple clips from a list.

**Files:** `src/generate.py` (CLI design)

**Impact:** For producing 100 audio files, user must invoke script 100 times.

**Current state:** Not a blocker for typical use (1-5 generations per day). Low priority.

**Fix approach:**
- Add `--batch-file` option to read newline-separated prompts
- Process sequentially with progress bar

---

## Dependencies at Risk

### edge-tts has upstream maintenance concerns

**Risk:** edge-tts is a community reverse-engineering project that depends on Microsoft's undocumented API. Microsoft could change the API without notice.

**Impact:** Voice generation could fail with 403 or other errors.

**Version:** `edge-tts>=7.0,<8` (from requirements.txt)

**Current mitigation:** README documents this and suggests manual upgrade.

**Migration plan:** Switch to a stable commercial API (Google Cloud Text-to-Speech, AWS Polly) if edge-tts breaks. Would require new CLI arg.

---

### gradio_client version mismatch risk

**Risk:** `gradio_client==0.2.7` pinned to match `gradio==3.34.0`. Upgrading either breaks the pin.

**Files:** `scripts/setup_rvc.sh` (line 111)

**Impact:** RVC WebUI may not run if gradio breaks this assumption.

**Mitigation:** Already documented in setup_rvc.sh (lines 105-108).

**Fix approach:** Monitor gradio releases and test compatibility quarterly.

---

### matplotlib 3.7.3 pin for removed API

**Risk:** RVC training uses deprecated `fig.canvas.tostring_rgb()` (removed in matplotlib 3.8+).

**Files:** `scripts/setup_rvc.sh` (line 121)

**Impact:** RVC training fails if matplotlib upgrades beyond 3.7.3.

**Mitigation:** Already documented (lines 113-121).

**Fix approach:** Watch for RVC fixes upstream. If RVC gets patched, unpin matplotlib.

---

## Missing Critical Features

### No progress indication during long operations

**Issue:** RVC inference can take 5-10 minutes but provides no real-time feedback unless `--verbose` used.

**Files:** `src/generate.py` (lines 315-326)

**Impact:** User thinks the program froze.

**Fix approach:** Stream RVC subprocess stderr line-by-line in real-time (with filtering for verbose mode).

---

### No graceful cancellation of preprocessing

**Issue:** If preprocessing is interrupted (^C), temp files are left in `dataset/processed/` in a partial state.

**Files:** `src/preprocess.py` (lines 176-212)

**Impact:** Next run sees partial data and may produce unexpected results.

**Fix approach:** Use try/finally to clean temp files on interrupt. Or use atomic writes (write to temp location first, rename on success).

---

### No validation of raw audio quality before preprocessing

**Issue:** Preprocess doesn't warn if audio is too quiet, too loud, or has background noise.

**Files:** `src/preprocess.py` (full pipeline)

**Impact:** Users might preprocess bad audio without realizing. Only discover at training time (after hours of preprocessing).

**Fix approach:** Add optional `--validate` flag that analyzes raw audio and warns about:
- Peak levels (< -40 dB or > -3 dB is suspicious)
- Silence duration > 50% of file
- Constant background noise (low frequency energy)

---

## Documentation Gaps

### Hardcoded CANONICAL_SR=44100 reasoning not explained

**Issue:** `src/preprocess.py` line 36 sets `CANONICAL_SR = 44100` without comment explaining why.

**Impact:** Future maintainers might assume it's arbitrary and change it, breaking RVC compatibility.

**Fix approach:** Add comment: `# RVC trained on 44.1kHz (internal constraint). Do not change.`

---

### Slicer2 silence detection threshold is magic number

**Issue:** `_slice_with_slicer2()` uses `threshold=-40` (line 118) with no explanation.

**Files:** `src/preprocess.py` (line 118)

**Fix approach:** Add comment: `# -40 dB = ~10x quieter than -20 dB loudness target. Tuned to drop breath/background.`

---

*Concerns audit: 2026-04-09*
