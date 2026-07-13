"""Unit tests for the key CNN dispatch + in-process inference contract.

Dispatch tests mock ``jams.analysis.key_cnn.analyze`` so nothing heavy runs; the
worker-protocol machinery the old subprocess path used is covered in tests/test_stems.py
(the remaining ``_Worker`` consumers). One real-inference test exercises the bundled
weights end-to-end on the synthetic fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jams.analysis import key as K
from jams.analysis import key_cnn

_RESULT = {
    "key": "D minor",
    "tonic": "D",
    "mode": "minor",
    "confidence": 0.871,
    "probs": [0.0] * 24,
    "method": "key-cnn-v1",
}


def test_detect_key_dispatches_to_cnn(monkeypatch, cmajor_wav):
    def fake_analyze(path):
        assert path == cmajor_wav
        return dict(_RESULT)

    monkeypatch.setattr(key_cnn, "analyze", fake_analyze)
    out = K.detect_key(cmajor_wav)
    assert out == {
        "key": "D minor", "tonic": "D", "mode": "minor",
        "confidence": 0.871, "method": "key-cnn-v1",
    }


def test_cnn_error_propagates(monkeypatch, cmajor_wav):
    def boom(path):
        raise RuntimeError("key-cnn failed: weights missing")

    monkeypatch.setattr(key_cnn, "analyze", boom)
    with pytest.raises(RuntimeError, match="weights missing"):
        K.detect_key(cmajor_wav)


def test_real_inference_contract(cmajor_wav):
    # Loads the bundled weights and runs the full in-process pipeline. Asserts the
    # response contract, not the musical answer (the fixture is a bare synth chord).
    out = K.detect_key(cmajor_wav)
    tonic, mode = out["key"].split()
    assert tonic in key_cnn.NOTES and mode in ("major", "minor")
    assert out["tonic"] == tonic and out["mode"] == mode
    assert 0.0 < out["confidence"] <= 1.0
    assert out["method"] == "key-cnn-v1"


def test_empty_audio_raises(tmp_path):
    import wave

    p = tmp_path / "empty.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
    with pytest.raises(ValueError, match="Empty/undecodable"):
        key_cnn.analyze(str(p))


def test_bundled_weights_present():
    # The wheel must carry the CNN weights; absence is a broken install.
    assert (Path(K.__file__).parent.parent / "data" / "models" / "key_cnn_v1.pt").exists()
