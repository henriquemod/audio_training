"""Integration test hitting real Edge-TTS. Marked @pytest.mark.network."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytestmark = pytest.mark.network


def test_edge_tts_produces_nonempty_mp3(tmp_path: Path):
    import edge_tts

    out = tmp_path / "out.mp3"

    async def go():
        communicate = edge_tts.Communicate("Hello from the test.", "en-US-GuyNeural")
        await communicate.save(str(out))

    asyncio.run(go())
    assert out.exists()
    assert out.stat().st_size > 1000  # arbitrary sanity floor
