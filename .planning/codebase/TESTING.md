# Testing Patterns

**Analysis Date:** 2026-04-09

## Test Framework

**Runner:**
- pytest 8.2.0
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`

**Assertion Library:**
- pytest's built-in assertions with `assert` statements
- No external assertion library required

**Run Commands:**
```bash
pytest                           # Run all tests (excluding network and gpu)
pytest -m "not network and not gpu"  # Explicit: default excludes
pytest -m network               # Run only network-marked tests
pytest -m gpu                   # Run only GPU-marked tests
pytest tests/unit/              # Run only unit tests
pytest tests/integration/       # Run only integration tests
pytest -v                       # Verbose output
pytest --tb=short              # Short traceback format
```

**Default behavior:**
- Markers: `-m 'not network and not gpu'` (configured in `addopts`)
- Tests requiring internet or GPU must be explicitly run; default run skips them
- Test paths: `["tests"]` (searches `tests/` directory)

## Test File Organization

**Location:**
- Unit tests: `tests/unit/test_*.py` (co-located with source structure implied)
- Integration tests: `tests/integration/test_*.py`
- Shared fixtures: `tests/conftest.py`

**Naming:**
- Test files: `test_<module>.py` (e.g., `test_ffmpeg_utils.py` for `src/ffmpeg_utils.py`)
- Test functions: `test_<function>_<scenario>()` (e.g., `test_run_ffmpeg_success`, `test_check_ffmpeg_present_and_recent`)
- Fixture functions: lowercase with underscores: `fake_ffmpeg`, `tmp_path`, `monkeypatch`

**Structure:**
```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_doctor.py       # Tests for src/doctor.py
│   ├── test_ffmpeg_utils.py # Tests for src/ffmpeg_utils.py
│   ├── test_generate_cli.py # Tests for src/generate.py
│   └── test_preprocess.py   # Tests for src/preprocess.py
└── integration/
    ├── test_edge_tts.py     # Network test, marked @pytest.mark.network
    └── test_preprocess_real.py  # Real ffmpeg test, skipped if ffmpeg unavailable
```

## Test Structure

**Suite organization:**
```python
"""Unit tests for src/ffmpeg_utils.py."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.ffmpeg_utils import FfmpegError, run_ffmpeg


def test_run_ffmpeg_success(tmp_path: Path, fake_ffmpeg):
    """Test successful ffmpeg invocation."""
    out = tmp_path / "out.wav"
    fake_ffmpeg.configure(exit_code=0, touch_output=True)
    run_ffmpeg(
        ["-i", "input.wav", str(out)],
        context="test stage",
        expected_output=out,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_run_ffmpeg_nonzero_exit_raises(tmp_path: Path, fake_ffmpeg):
    """Test that non-zero exit raises FfmpegError with context."""
    out = tmp_path / "out.wav"
    fake_ffmpeg.configure(exit_code=1, stderr="synthetic ffmpeg failure", touch_output=False)
    with pytest.raises(FfmpegError) as excinfo:
        run_ffmpeg(
            ["-i", "input.wav", str(out)],
            context="loudnorm stage",
            expected_output=out,
        )
    msg = str(excinfo.value)
    assert "loudnorm stage" in msg
    assert "synthetic ffmpeg failure" in msg
```

**Patterns:**
- Module docstring (always first): `"""Unit tests for src/module.py."""`
- Imports: `from __future__ import annotations`, then pytest, then module under test
- Test naming: verb + scenario: `test_<function>_<scenario>`
- Docstrings: One-liner describing what is tested
- Arrange-Act-Assert pattern: setup, call function, assert results
- Use `with pytest.raises(ExceptionType)` for exception testing
- Check exception message content: `assert "keyword" in str(excinfo.value)`

## Mocking

**Framework:** unittest.mock (stdlib)

**Patterns - Subprocess mocking:**
```python
from unittest.mock import patch

def test_check_ffmpeg_present_and_recent():
    fake_output = "ffmpeg version 6.1.1 Copyright (c) 2000-2023"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_output
        result = check_ffmpeg()
    assert result.ok is True
    assert "6.1.1" in result.detail
```

**Patterns - Side effects and exceptions:**
```python
def test_check_ffmpeg_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = check_ffmpeg()
    assert result.ok is False
    assert "apt install ffmpeg" in result.fix_hint
```

**Custom fixtures - fake_ffmpeg:**
```python
@pytest.fixture
def fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Install a fake ffmpeg binary on PATH that the test can control."""
    # Creates a shell script that mimics ffmpeg behavior
    # Usage:
    #   fake_ffmpeg.configure(exit_code=0, stderr="", touch_output=True)
    #   run_ffmpeg(...) uses the fake
```
- Location: `tests/conftest.py` (shared across all test suites)
- Controller interface: `fake_ffmpeg.configure(exit_code=int, stderr=str, touch_output=bool)`
- Implementation: Bash script written to temp directory, added to PATH via monkeypatch
- Behavior: Mimics ffmpeg by writing exit code to state file, echoing stderr, optionally creating output file

**What to mock:**
- Subprocess calls: use `patch("subprocess.run")`
- System commands: use fake binaries (like `fake_ffmpeg`)
- External services: use `patch("edge_tts.list_voices")`
- Async functions: patch the function, return async mock or use `asyncio.run()`

**What NOT to mock:**
- File system operations: use `tmp_path` fixture
- Path/pathlib operations: real paths are fine in tests
- Internal module functions: test them directly
- Exception raising: let real code raise, test error handling

## Fixtures and Factories

**Test data - fixtures (from conftest.py):**
```python
@pytest.fixture
def fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Install a fake ffmpeg binary on PATH that the test can control."""
    # ... implementation
    return controller  # exposes configure() method
```

**Built-in fixtures used:**
- `tmp_path: Path` - Temporary directory unique to test
- `monkeypatch: pytest.MonkeyPatch` - Safely modify environment/sys.path
- `capsys` - Capture stdout/stderr

**Location:**
- Fixtures live in `tests/conftest.py` (discovered automatically by pytest)
- Fixtures are reusable across all test files
- Module-specific fixtures would go in test file itself (none currently)

**Async test fixtures:**
```python
def test_edge_tts_produces_nonempty_mp3(tmp_path: Path):
    import edge_tts
    
    out = tmp_path / "out.mp3"
    
    async def go():
        communicate = edge_tts.Communicate("Hello from the test.", "en-US-GuyNeural")
        await communicate.save(str(out))
    
    asyncio.run(go())
    assert out.exists()
    assert out.stat().st_size > 1000
```
- Pattern: Define async function inside test, use `asyncio.run()`
- No async fixtures or `@pytest.mark.asyncio` used

## Coverage

**Requirements:** None enforced (no coverage target specified)

**Note:** This project prioritizes testing strategy over coverage percentage. Tests focus on:
- Error paths and exception handling
- Pipeline integration
- CLI validation
- Subprocess interaction safety

## Test Types

**Unit Tests:**
- Location: `tests/unit/`
- Scope: Test individual functions in isolation
- Mocking: Heavy use (mock subprocess, external services)
- Examples: `test_ffmpeg_utils.py`, `test_doctor.py`, `test_preprocess.py`
- Patterns:
  - `test_parse_ffmpeg_version_*`: Test parsing logic with various inputs
  - `test_check_*`: Test system/dependency checks with mocked subprocess
  - `test_build_*_args`: Test ffmpeg argument construction (pure functions)
  - `test_cli_*`: Test CLI validation and flag logic with CliRunner

**Integration Tests:**
- Location: `tests/integration/`
- Scope: Test full pipelines with real tools
- Markers: `@pytest.mark.network`, `@pytest.mark.skipif(shutil.which("ffmpeg") is None, ...)`
- Examples: `test_edge_tts.py`, `test_preprocess_real.py`
- Patterns:
  - Network tests: Actually call edge-tts API (marked network, skipped by default)
  - ffmpeg tests: Generate synthetic audio, run through preprocess, verify output
  - Conditional skip: Tests skip gracefully if required tool unavailable

**CLI Testing:**
```python
from typer.testing import CliRunner
from src.generate import app

runner = CliRunner()

def test_cli_requires_text_or_text_file():
    result = runner.invoke(app, [])
    assert result.exit_code != 0

def test_cli_text_and_text_file_mutually_exclusive(tmp_path: Path):
    text_file = tmp_path / "script.txt"
    text_file.write_text("hello")
    result = runner.invoke(app, ["hello world", "--text-file", str(text_file)])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output.lower()
```
- Tool: typer.testing.CliRunner
- Approach: Invoke CLI with arguments, check exit code and output
- Validation: Test mutually-exclusive flags, required args, empty input

## Common Patterns

**Error testing:**
```python
def test_run_ffmpeg_nonzero_exit_raises(tmp_path: Path, fake_ffmpeg):
    out = tmp_path / "out.wav"
    fake_ffmpeg.configure(exit_code=1, stderr="synthetic ffmpeg failure", touch_output=False)
    with pytest.raises(FfmpegError) as excinfo:
        run_ffmpeg(
            ["-i", "input.wav", str(out)],
            context="loudnorm stage",
            expected_output=out,
        )
    msg = str(excinfo.value)
    assert "loudnorm stage" in msg
    assert "synthetic ffmpeg failure" in msg
```
- Pattern: Configure mock to trigger error, invoke function, assert exception type and message
- Check error messages for context tags and helpful details

**Async testing:**
```python
def test_edge_tts_produces_nonempty_mp3(tmp_path: Path):
    import edge_tts
    out = tmp_path / "out.mp3"
    
    async def go():
        communicate = edge_tts.Communicate("Hello from the test.", "en-US-GuyNeural")
        await communicate.save(str(out))
    
    asyncio.run(go())
    assert out.exists()
    assert out.stat().st_size > 1000
```
- Pattern: Define async function, run with `asyncio.run()`
- No pytest-asyncio or async fixtures needed

**Monkeypatching:**
```python
def test_cli_list_voices_runs(monkeypatch):
    from src import generate
    
    async def fake_list_voices():
        return [
            {"ShortName": "en-US-GuyNeural", "Gender": "Male", "Locale": "en-US"},
            {"ShortName": "en-US-JennyNeural", "Gender": "Female", "Locale": "en-US"},
        ]
    
    monkeypatch.setattr(generate, "_list_english_voices", fake_list_voices)
    result = runner.invoke(app, ["--list-voices"])
    assert result.exit_code == 0
    assert "en-US-GuyNeural" in result.stdout
```
- Pattern: Use monkeypatch to replace module function with test version
- Import module, then monkeypatch attribute on module object

**File system testing:**
```python
def test_preprocess_produces_wav_clips(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    _synth_speech_like_wav(raw / "sample.wav")
    
    run_preprocess(
        input_dir=raw,
        output_dir=processed,
        min_len_s=3.0,
        max_len_s=15.0,
        target_lufs=-20,
    )
    
    wavs = sorted(processed.glob("*.wav"))
    assert len(wavs) >= 1
    for w in wavs:
        data, sr = sf.read(w)
        assert sr == 44100
        assert data.ndim == 1
```
- Pattern: Use `tmp_path` for file operations, verify output files exist and have correct properties
- Use soundfile.read() to verify audio properties

---

*Testing analysis: 2026-04-09*
