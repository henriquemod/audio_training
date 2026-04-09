"""Unit tests for src/generate.py CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from src.generate import app, build_rvc_subprocess_cmd

runner = CliRunner()


# --- build_rvc_subprocess_cmd ---


def test_build_rvc_subprocess_cmd_has_required_args(tmp_path: Path):
    cmd = build_rvc_subprocess_cmd(
        rvc_python=Path("rvc/.venv/bin/python"),
        rvc_dir=Path("rvc"),
        input_wav=tmp_path / "in.wav",
        model_name="myvoice_v1",
        index_path=Path("models/myvoice_v1.index"),
        output_wav=tmp_path / "out.wav",
        pitch=0,
        index_rate=0.7,
        filter_radius=3,
        device="cuda:0",
    )
    # Command should invoke tools/infer_cli.py with named args
    assert str(Path("rvc/.venv/bin/python")) in cmd
    assert "tools/infer_cli.py" in " ".join(cmd)
    assert "--input_path" in cmd
    assert "--model_name" in cmd
    assert "myvoice_v1.pth" in cmd or "myvoice_v1" in cmd
    assert "--index_path" in cmd
    assert "--opt_path" in cmd
    assert "--f0up_key" in cmd
    assert "--index_rate" in cmd
    assert "--device" in cmd
    assert "cuda:0" in cmd
    assert "--f0method" in cmd
    assert "rmvpe" in cmd


# --- CLI flag validation ---


def test_cli_requires_text_or_text_file():
    result = runner.invoke(app, [])
    assert result.exit_code != 0


def test_cli_text_and_text_file_mutually_exclusive(tmp_path: Path):
    text_file = tmp_path / "script.txt"
    text_file.write_text("hello")
    result = runner.invoke(app, ["hello world", "--text-file", str(text_file)])
    assert result.exit_code == 2
    # CliRunner mixes stderr into output by default
    assert "mutually exclusive" in result.output.lower() or "cannot" in result.output.lower()


def test_cli_empty_text_fails():
    result = runner.invoke(app, [""])
    assert result.exit_code == 2


def test_cli_list_voices_runs(monkeypatch):
    """--list-voices should call edge-tts list_voices and not touch RVC."""
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
    assert "en-US-JennyNeural" in result.stdout
