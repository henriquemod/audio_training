"""Unit tests for src/doctor.py dependency verification."""

from __future__ import annotations

from unittest.mock import patch

from src.doctor import (
    PRETRAINED_MIN_BYTES,
    check_ffmpeg,
    check_ffmpeg_filters,
    check_mise,
    check_pretrained_v2_weights,
    check_python_version,
    check_training_dataset_nonempty,
    parse_ffmpeg_version,
    parse_ffmpeg_version_display,
    run_training_checks,
)

# --- parse_ffmpeg_version ---


def test_parse_ffmpeg_version_standard():
    output = "ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers"
    assert parse_ffmpeg_version(output) == (6, 1, 1)


def test_parse_ffmpeg_version_short():
    output = "ffmpeg version 5.0 Copyright (c) 2000-2022"
    assert parse_ffmpeg_version(output) == (5, 0, 0)


def test_parse_ffmpeg_version_unparseable_returns_none():
    assert parse_ffmpeg_version("garbage output") is None


def test_parse_ffmpeg_version_nightly_btbn_build():
    # BtbN static builds used on training pods report a git tag, not a release.
    output = (
        "ffmpeg version N-123884-gd3d0b7a5ee-20260409 "
        "Copyright (c) 2000-2026 the FFmpeg developers"
    )
    result = parse_ffmpeg_version(output)
    assert result is not None
    # Sentinel major must be high enough to satisfy any realistic floor.
    assert result >= (5, 0, 0)
    assert result[0] >= 9999


def test_parse_ffmpeg_version_n_without_git_sha_rejected():
    # A hypothetical future stable release tagged `N-5` must NOT be mapped to
    # the nightly sentinel. Without a `-g<sha>` suffix, we should reject it.
    output = "ffmpeg version N-5 Copyright (c) 2000-2030"
    assert parse_ffmpeg_version(output) is None


# --- parse_ffmpeg_version_display ---


def test_parse_ffmpeg_version_display_stable():
    output = "ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers"
    assert parse_ffmpeg_version_display(output) == "6.1.1-3ubuntu5"


def test_parse_ffmpeg_version_display_nightly():
    output = (
        "ffmpeg version N-123884-gd3d0b7a5ee-20260409 "
        "Copyright (c) 2000-2026 the FFmpeg developers"
    )
    assert parse_ffmpeg_version_display(output) == "N-123884-gd3d0b7a5ee-20260409"


def test_parse_ffmpeg_version_display_unknown():
    assert parse_ffmpeg_version_display("no version info here") == "unknown"


# --- check_ffmpeg ---


def test_check_ffmpeg_present_and_recent():
    fake_output = "ffmpeg version 6.1.1 Copyright (c) 2000-2023"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_output
        result = check_ffmpeg()
    assert result.ok is True
    assert "6.1.1" in result.detail


def test_check_ffmpeg_too_old():
    fake_output = "ffmpeg version 4.2.0 Copyright (c) 2000-2020"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_output
        result = check_ffmpeg()
    assert result.ok is False
    assert "5.0" in result.fix_hint


def test_check_ffmpeg_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = check_ffmpeg()
    assert result.ok is False
    assert "apt install ffmpeg" in result.fix_hint


def test_check_ffmpeg_nightly_btbn_build_accepted():
    # BtbN static ffmpeg (used by scripts/setup_pod.sh) reports a git-tag version.
    # The check must accept it and surface the raw build tag in `detail` rather
    # than a misleading `9999.0.0`.
    fake_output = (
        "ffmpeg version N-123884-gd3d0b7a5ee-20260409 "
        "Copyright (c) 2000-2026 the FFmpeg developers"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_output
        result = check_ffmpeg()
    assert result.ok is True
    assert result.detail == "N-123884-gd3d0b7a5ee-20260409"


# --- check_ffmpeg_filters ---


def test_check_ffmpeg_filters_all_present():
    fake_filters = (
        " TSC afftdn            A->A       Denoise audio samples using FFT.\n"
        " ... loudnorm          A->A       EBU R128 loudness normalization\n"
        " ... silencedetect     A->A       Detect silence.\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_filters
        result = check_ffmpeg_filters()
    assert result.ok is True


def test_check_ffmpeg_filters_missing_afftdn():
    fake_filters = (
        " ... loudnorm          A->A       EBU R128 loudness normalization\n"
        " ... silencedetect     A->A       Detect silence.\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_filters
        result = check_ffmpeg_filters()
    assert result.ok is False
    assert "afftdn" in result.detail


# --- check_python_version ---


def test_check_python_version_310():
    with patch("sys.version_info", (3, 10, 14, "final", 0)):
        result = check_python_version()
    assert result.ok is True


def test_check_python_version_314_fails():
    with patch("sys.version_info", (3, 14, 3, "final", 0)):
        result = check_python_version()
    assert result.ok is False
    assert "mise" in result.fix_hint.lower()


# --- check_mise ---


def test_check_mise_present():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "2026.3.18 linux-x64"
        result = check_mise()
    assert result.ok is True


def test_check_mise_missing_is_soft_ok():
    # mise is a laptop-dev convenience; on pods it is not expected to exist.
    # A missing `mise` binary must NOT fail the system-checks pre-flight that
    # scripts/setup_rvc.sh runs, otherwise pod bootstrap aborts.
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = check_mise()
    assert result.ok is True
    assert "optional" in result.detail.lower()
    assert result.fix_hint == ""


def test_check_mise_broken_install_is_soft_fail():
    # A mise binary that exits non-zero (broken install) is still surfaced as
    # a failure so laptop users notice, but only when the binary is present.
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "mise: config error"
        mock_run.return_value.stdout = ""
        result = check_mise()
    assert result.ok is False
    assert "mise.jdx.dev" in result.fix_hint


# --- check_pretrained_v2_weights (Phase 2) ---


def test_check_pretrained_v2_weights_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    result = check_pretrained_v2_weights(40000, "v2", if_f0=True)
    assert not result.ok
    assert "f0G40k.pth" in result.detail or "f0D40k.pth" in result.detail
    assert result.fix_hint  # non-empty


def test_check_pretrained_v2_weights_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    sub = tmp_path / "assets" / "pretrained_v2"
    sub.mkdir(parents=True)
    (sub / "f0G40k.pth").write_bytes(b"\x00" * 100)
    (sub / "f0D40k.pth").write_bytes(b"\x00" * 100)
    result = check_pretrained_v2_weights(40000, "v2", if_f0=True)
    assert not result.ok
    assert "bytes" in result.detail
    assert "Truncated" in result.fix_hint or "truncated" in result.fix_hint.lower()


def _sparse(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        if size > 0:
            f.seek(size - 1)
            f.write(b"\x00")


def test_check_pretrained_v2_weights_ok(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    sub = tmp_path / "assets" / "pretrained_v2"
    sub.mkdir(parents=True)
    _sparse(sub / "f0G40k.pth", PRETRAINED_MIN_BYTES + 1)
    _sparse(sub / "f0D40k.pth", PRETRAINED_MIN_BYTES + 1)
    result = check_pretrained_v2_weights(40000, "v2", if_f0=True)
    assert result.ok, f"expected ok, got: {result.detail}"


def test_check_pretrained_v1_no_f0_uses_pretrained_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    sub = tmp_path / "assets" / "pretrained"
    sub.mkdir(parents=True)
    _sparse(sub / "G40k.pth", PRETRAINED_MIN_BYTES + 1)
    _sparse(sub / "D40k.pth", PRETRAINED_MIN_BYTES + 1)
    result = check_pretrained_v2_weights(40000, "v1", if_f0=False)
    assert result.ok


def test_check_pretrained_v2_weights_bad_sample_rate(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    result = check_pretrained_v2_weights(44100, "v2", if_f0=True)
    assert not result.ok
    assert "44100" in result.detail


# --- check_training_dataset_nonempty (Phase 2) ---


def test_check_training_dataset_nonempty_missing(tmp_path):
    result = check_training_dataset_nonempty(tmp_path / "nope")
    assert not result.ok
    assert "not found" in result.detail


def test_check_training_dataset_nonempty_not_dir(tmp_path):
    f = tmp_path / "file.wav"
    f.touch()
    result = check_training_dataset_nonempty(f)
    assert not result.ok
    assert "not a directory" in result.detail


def test_check_training_dataset_nonempty_empty(tmp_path):
    result = check_training_dataset_nonempty(tmp_path)
    assert not result.ok
    assert "no audio files" in result.detail


def test_check_training_dataset_nonempty_only_txt(tmp_path):
    (tmp_path / "readme.txt").touch()
    result = check_training_dataset_nonempty(tmp_path)
    assert not result.ok


def test_check_training_dataset_nonempty_ok(tmp_path):
    (tmp_path / "a.wav").touch()
    (tmp_path / "b.flac").touch()
    result = check_training_dataset_nonempty(tmp_path)
    assert result.ok
    assert "2 audio file" in result.detail


# --- run_training_checks composition ---


def test_run_training_checks_returns_list_of_results(tmp_path, monkeypatch):
    # Stub out every underlying check so we verify composition only.
    from src import doctor as d

    def _ok(name="x"):
        return d.CheckResult(name=name, ok=True, detail="ok")

    monkeypatch.setattr(d, "check_python_version", lambda: _ok("py"))
    monkeypatch.setattr(d, "check_ffmpeg", lambda: _ok("ffmpeg"))
    monkeypatch.setattr(d, "check_nvidia_smi", lambda: _ok("nvidia"))
    monkeypatch.setattr(d, "check_rvc_cloned", lambda: _ok("cloned"))
    monkeypatch.setattr(d, "check_rvc_venv", lambda: _ok("venv"))
    monkeypatch.setattr(d, "check_rvc_weights", lambda: _ok("weights"))
    monkeypatch.setattr(d, "check_rvc_torch_cuda", lambda: _ok("torch"))
    monkeypatch.setattr(d, "check_disk_space_floor", lambda *a, **k: _ok("disk"))
    monkeypatch.setattr(d, "check_gpu_vram_floor", lambda *a, **k: _ok("vram"))
    monkeypatch.setattr(d, "check_rvc_mute_refs", lambda: _ok("mute"))
    monkeypatch.setattr(d, "check_hubert_base", lambda: _ok("hubert"))
    monkeypatch.setattr(
        d, "check_pretrained_v2_weights", lambda *a, **k: _ok("pretrained")
    )
    monkeypatch.setattr(
        d, "check_training_dataset_nonempty", lambda *a, **k: _ok("dataset")
    )
    results = run_training_checks(
        dataset_dir=tmp_path, sample_rate=40000, version="v2", if_f0=True
    )
    assert isinstance(results, list)
    assert len(results) >= 12
    assert all(r.ok for r in results)
    names = [r.name for r in results]
    assert "pretrained" in names
    assert "dataset" in names


# --- _run_checks severity handling ---


def test_run_checks_returns_true_when_only_warnings_fail():
    """A non-ok CheckResult with severity='warning' must NOT block.

    Regression guard for the disk-space-floor soft-threshold case: an 80GB
    H100 pod with 11 GiB free should warn but still let the user run
    `src/train.py`. If _run_checks ever starts treating warnings as errors
    again, this test will catch it.
    """
    from src.doctor import CheckResult, _run_checks

    checks = [
        lambda: CheckResult(name="ok_check", ok=True, detail="fine"),
        lambda: CheckResult(
            name="soft_check",
            ok=False,
            detail="only 11.0 GiB free",
            fix_hint="consider freeing space",
            severity="warning",
        ),
    ]
    assert _run_checks(checks) is True


def test_run_checks_returns_false_when_any_error_fails():
    from src.doctor import CheckResult, _run_checks

    checks = [
        lambda: CheckResult(
            name="hard_check",
            ok=False,
            detail="missing binary",
            fix_hint="install it",
            severity="error",
        ),
        lambda: CheckResult(
            name="soft_check",
            ok=False,
            detail="tight disk",
            severity="warning",
        ),
    ]
    assert _run_checks(checks) is False


def test_run_checks_default_severity_is_error():
    """CheckResult without explicit severity must default to 'error' so
    existing checks keep their strict blocking behavior."""
    from src.doctor import CheckResult, _run_checks

    checks = [
        lambda: CheckResult(name="legacy_check", ok=False, detail="broken"),
    ]
    assert _run_checks(checks) is False
