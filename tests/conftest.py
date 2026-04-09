"""Shared test fixtures."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest


@pytest.fixture
def fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Install a fake `ffmpeg` binary on PATH that the test can control.

    Usage:
        fake_ffmpeg.configure(exit_code=0, stderr="", touch_output=True)
        # now subprocess calls to "ffmpeg" use the fake

    The fake writes the requested exit code, echoes requested stderr, and
    optionally creates the file passed as the last positional argument
    (ffmpeg output file convention).
    """
    fake_dir = tmp_path / "fake_bin"
    fake_dir.mkdir()
    fake_script = fake_dir / "ffmpeg"
    state_file = tmp_path / "fake_ffmpeg_state"

    class Controller:
        def configure(self, exit_code: int = 0, stderr: str = "", touch_output: bool = True):
            state_file.write_text(f"{exit_code}\n{int(touch_output)}\n{stderr}")

    controller = Controller()
    controller.configure()  # default: success, touch output

    fake_script.write_text(
        f"""#!/usr/bin/env bash
state_file="{state_file}"
exit_code=$(sed -n 1p "$state_file")
touch_output=$(sed -n 2p "$state_file")
stderr_msg=$(sed -n '3,$p' "$state_file")
# last positional arg is ffmpeg's output file
last_arg="${{@: -1}}"
if [ "$touch_output" = "1" ]; then
  printf 'fake' > "$last_arg"
fi
if [ -n "$stderr_msg" ]; then
  printf '%s\\n' "$stderr_msg" >&2
fi
exit "$exit_code"
"""
    )
    fake_script.chmod(fake_script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    monkeypatch.setenv("PATH", f"{fake_dir}{os.pathsep}{os.environ['PATH']}")
    return controller
