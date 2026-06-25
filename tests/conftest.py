"""Shared test fixtures."""

from __future__ import annotations

import math
import wave

import pytest


@pytest.fixture(scope="session")
def cmajor_wav(tmp_path_factory) -> str:
    """A short, mostly-tonal C-major chord rendered to a real WAV file."""
    sr = 22050
    dur = 4.0
    freqs = [261.63, 329.63, 392.0]  # C E G
    n = int(sr * dur)
    samples = bytearray()
    for i in range(n):
        t = i / sr
        v = sum(math.sin(2 * math.pi * f * t) for f in freqs) / len(freqs)
        samples += int(max(-1.0, min(1.0, v)) * 30000).to_bytes(2, "little", signed=True)
    path = tmp_path_factory.mktemp("audio") / "cmajor.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(samples))
    return str(path)
