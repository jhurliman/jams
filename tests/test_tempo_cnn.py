"""Unit tests for the tempo CNN dispatch + in-process inference contract.

Dispatch tests mock ``jams.analysis.tempo_cnn.analyze`` so nothing heavy runs; the
worker-protocol machinery the old subprocess path used is covered in tests/test_stems.py
(the remaining ``_Worker`` consumers). One real-inference test exercises the bundled
weights end-to-end on a synthetic click track.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest

from jams.analysis import tempo as T
from jams.analysis import tempo_cnn


def test_detect_tempo_uses_cnn_and_resolves_octave(monkeypatch, cmajor_wav):
    def fake_analyze(path):
        assert path == cmajor_wav
        return {"bpm": 87.0, "confidence": 0.9, "method": "tempo-cnn-v1"}

    monkeypatch.setattr(tempo_cnn, "analyze", fake_analyze)
    out = T.detect_tempo(cmajor_wav, genre="Drum & Bass")
    assert out["bpm"] == pytest.approx(174.0)
    assert out["bpm_raw"] == pytest.approx(87.0)
    assert out["octave_resolved"] is True
    assert out["method"] == "tempo-cnn-v1"


def test_detect_tempo_no_hint_keeps_raw(monkeypatch, cmajor_wav):
    monkeypatch.setattr(
        tempo_cnn, "analyze",
        lambda path: {"bpm": 151.0, "confidence": 0.5, "method": "tempo-cnn-v1"},
    )
    out = T.detect_tempo(cmajor_wav)
    assert out["bpm"] == pytest.approx(151.0)
    assert out["octave_resolved"] is False


@pytest.fixture(scope="module")
def click_128_wav(tmp_path_factory) -> str:
    """A 30 s click track at 128 BPM (decaying sine bursts on each beat)."""
    sr = 11025
    dur = 30.0
    beat = 60.0 / 128.0
    n = int(sr * dur)
    samples = bytearray()
    for i in range(n):
        t = i / sr
        dt = t % beat
        v = math.sin(2 * math.pi * 1000 * dt) * math.exp(-dt * 60.0)
        samples += int(max(-1.0, min(1.0, v)) * 30000).to_bytes(2, "little", signed=True)
    path = tmp_path_factory.mktemp("audio") / "click128.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(samples))
    return str(path)


def test_real_inference_128bpm_click(click_128_wav):
    # Loads the bundled weights and runs the full in-process pipeline; a clean click
    # track must land within ±1 BPM of the truth.
    out = tempo_cnn.analyze(click_128_wav)
    assert abs(out["bpm"] - 128.0) <= 1.0
    assert 0.0 < out["confidence"] <= 1.0
    assert out["method"] == "tempo-cnn-v1"


def test_empty_audio_raises(tmp_path):
    p = tmp_path / "empty.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(11025)
    with pytest.raises(ValueError, match="Empty/undecodable"):
        tempo_cnn.analyze(str(p))


def test_bundled_weights_present():
    # The wheel must carry the CNN weights; absence is a broken install.
    data = Path(T.__file__).parent.parent / "data"
    assert (data / "models" / "tempo_cnn_v1.pt").exists()


def test_old_tempocnn_graph_gone():
    # The retired TempoCNN graph must not ride along in the package.
    data = Path(T.__file__).parent.parent / "data"
    assert not (data / "models" / "deepsquare-k16-3.pb").exists()
    assert not list((data / "models").glob("*.pb"))
