"""CLI / main() tests for src/train.py using typer.testing.CliRunner."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from src.doctor import CheckResult
from src.train import app

runner = CliRunner(mix_stderr=False)


def _all_ok_stub(*, dataset_dir, sample_rate, version, if_f0):
    return [CheckResult(name="stub", ok=True, detail="stub")]


def _one_fail_stub(*, dataset_dir, sample_rate, version, if_f0):
    return [
        CheckResult(name="stub-ok", ok=True, detail="ok"),
        CheckResult(name="stub-fail", ok=False, detail="bad", fix_hint="fix it"),
    ]


def _make_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "a.wav").touch()
    return ds


def test_help_lists_all_flags():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    for flag in (
        "--experiment-name",
        "--dataset-dir",
        "--sample-rate",
        "--rvc-version",
        "--f0-method",
        "--preset",
        "--epochs",
        "--batch-size",
        "--save-every",
        "--num-procs",
        "--gpus",
        "--verbose",
    ):
        assert flag in out, f"missing flag: {flag}"
    assert "--resume" not in out  # D-05


def test_rejects_invalid_experiment_name(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(
        app, ["--experiment-name", "../etc", "--dataset-dir", str(ds)]
    )
    assert result.exit_code == 2
    assert "invalid --experiment-name" in result.stderr


def test_rejects_long_experiment_name(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(
        app, ["--experiment-name", "x" * 65, "--dataset-dir", str(ds)]
    )
    assert result.exit_code == 2


def test_rejects_invalid_sample_rate(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "--experiment-name", "smoke",
            "--dataset-dir", str(ds),
            "--sample-rate", "44100",
        ],
    )
    assert result.exit_code == 2
    assert "--sample-rate" in result.stderr


def test_rejects_invalid_rvc_version(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "--experiment-name", "smoke",
            "--dataset-dir", str(ds),
            "--rvc-version", "v3",
        ],
    )
    assert result.exit_code == 2


def test_rejects_invalid_f0_method_dio(tmp_path, monkeypatch):
    """dio is excluded per Open Q5 / D-04."""
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "--experiment-name", "smoke",
            "--dataset-dir", str(ds),
            "--f0-method", "dio",
        ],
    )
    assert result.exit_code == 2
    assert "--f0-method" in result.stderr


def test_rejects_invalid_preset(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "--experiment-name", "smoke",
            "--dataset-dir", str(ds),
            "--preset", "extreme",
        ],
    )
    assert result.exit_code == 2


def test_rejects_v1_with_32k_combination(tmp_path, monkeypatch):
    """Open Q4 / Risk: webui silently corrects this; we reject explicitly."""
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "--experiment-name", "smoke",
            "--dataset-dir", str(ds),
            "--rvc-version", "v1",
            "--sample-rate", "32000",
        ],
    )
    assert result.exit_code == 2
    assert "v1" in result.stderr and "32000" in result.stderr


def test_doctor_failure_exits_1(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _one_fail_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "--experiment-name", "smoke",
            "--dataset-dir", str(ds),
        ],
    )
    assert result.exit_code == 1
    # Failure name and fix_hint should appear somewhere.
    combined = (result.stdout or "") + (result.stderr or "")
    assert "stub-fail" in combined
    assert "fix it" in combined


def test_all_valid_reaches_runner(tmp_path, monkeypatch):
    """All flags valid + doctor pass -> run_pipeline is invoked and its rc is returned."""
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    calls = {}

    def fake_run_pipeline(**kwargs):
        calls.update(kwargs)
        return 0

    monkeypatch.setattr("src.train.run_pipeline", fake_run_pipeline)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "--experiment-name", "smoke",
            "--dataset-dir", str(ds),
            "--preset", "smoke",
        ],
    )
    assert result.exit_code == 0
    assert calls["experiment_name"] == "smoke"
    assert calls["hp"]["epochs"] == 1


def test_preset_override_reaches_main(tmp_path, monkeypatch):
    """Smoke verifies preset + override resolution does not crash."""
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("src.train.run_pipeline", fake_run_pipeline)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "--experiment-name", "smoke",
            "--dataset-dir", str(ds),
            "--preset", "high",
            "--epochs", "800",
        ],
    )
    assert result.exit_code == 0
    assert captured["hp"]["epochs"] == 800
    assert captured["hp"]["batch_size"] == 40  # high preset default


def test_run_pipeline_failure_exit_code_propagates(tmp_path, monkeypatch):
    """If run_pipeline returns 3, main() exits 3."""
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    monkeypatch.setattr("src.train.run_pipeline", lambda **kw: 3)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "--experiment-name", "smoke",
            "--dataset-dir", str(ds),
            "--preset", "smoke",
        ],
    )
    assert result.exit_code == 3


def test_dataset_dir_missing_reaches_doctor_exit_1(tmp_path, monkeypatch):
    """Missing dataset dir is caught by check_training_dataset_nonempty -> exit 1."""
    from src.doctor import check_training_dataset_nonempty

    def stub(*, dataset_dir, sample_rate, version, if_f0):
        return [check_training_dataset_nonempty(dataset_dir)]

    monkeypatch.setattr("src.train.run_training_checks", stub)
    result = runner.invoke(
        app,
        [
            "--experiment-name", "smoke",
            "--dataset-dir", str(tmp_path / "nope"),
        ],
    )
    assert result.exit_code == 1
