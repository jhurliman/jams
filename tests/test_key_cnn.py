"""Unit tests for the key CNN backend dispatch + worker protocol.

These mock the subprocess round-trip so nothing heavy (uv/torch) runs; the pattern
mirrors tests/test_structure.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jams.analysis import key as K
from jams.analysis import stems as S
from jams.config import Settings, get_settings

_RESULT = {
    "key": "D minor",
    "tonic": "D",
    "mode": "minor",
    "confidence": 0.871,
    "probs": [0.0] * 24,
    "method": "key-cnn-v1",
}
_OK = json.dumps({"ok": True, "result": _RESULT}) + "\n"


@pytest.fixture(autouse=True)
def _fresh_singleton():
    K._key_cnn_singleton = None
    yield
    K._key_cnn_singleton = None


def test_worker_roundtrip(monkeypatch):
    w = S._Worker(Path("/nonexistent.py"), "key-cnn", uv_setting="key_cnn_uv")
    monkeypatch.setattr(w, "_ensure_alive", lambda: None)
    monkeypatch.setattr(w, "_round_trip", lambda req: _OK)
    out = w.analyze({"audio": "/a.wav"})
    assert out["key"] == "D minor" and out["method"] == "key-cnn-v1"


def test_worker_respawns_on_dead_pipe(monkeypatch):
    w = S._Worker(Path("/nonexistent.py"), "key-cnn", uv_setting="key_cnn_uv")
    spawned = {"n": 0}
    calls = {"n": 0}
    monkeypatch.setattr(w, "_ensure_alive", lambda: None)
    monkeypatch.setattr(w, "_spawn", lambda: spawned.__setitem__("n", spawned["n"] + 1))

    def rt(req):
        calls["n"] += 1
        return "" if calls["n"] == 1 else _OK  # first call: worker is dead

    monkeypatch.setattr(w, "_round_trip", rt)
    out = w.analyze({"audio": "/a.wav"})
    assert out["key"] == "D minor"
    assert spawned["n"] == 1  # respawned exactly once


def test_worker_uses_key_cnn_uv_setting(monkeypatch):
    monkeypatch.setattr(
        "jams.analysis.stems.get_settings",
        lambda: Settings(key_cnn_uv="/custom/uv", _env_file=None),
    )
    w = S._Worker(Path("/nonexistent.py"), "key-cnn", uv_setting="key_cnn_uv")
    captured = {}

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        raise RuntimeError("stop before real spawn")

    monkeypatch.setattr("jams.analysis.stems.subprocess.Popen", fake_popen)
    with pytest.raises(RuntimeError, match="stop before real spawn"):
        w._spawn()
    assert captured["cmd"][0] == "/custom/uv"


def test_detect_key_dispatches_to_cnn(monkeypatch, cmajor_wav):
    get_settings.cache_clear()
    monkeypatch.setenv("JAMS_KEY_BACKEND", "cnn")

    class FakeWorker:
        def analyze(self, req):
            assert req["audio"] == cmajor_wav
            return dict(_RESULT)

    monkeypatch.setattr(K, "_key_cnn_worker", lambda: FakeWorker())
    out = K.detect_key(cmajor_wav)
    assert out == {
        "key": "D minor", "tonic": "D", "mode": "minor",
        "confidence": 0.871, "method": "key-cnn-v1",
    }
    get_settings.cache_clear()


def test_detect_key_fusion_backend_still_routes_to_essentia(monkeypatch, cmajor_wav):
    get_settings.cache_clear()
    monkeypatch.setenv("JAMS_KEY_BACKEND", "fusion")
    monkeypatch.setattr(
        K, "_detect_essentia", lambda path, refine_mode: {"method": "essentia-edma"}
    )
    assert K.detect_key(cmajor_wav)["method"] == "essentia-edma"
    get_settings.cache_clear()


def test_cnn_worker_error_propagates(monkeypatch, cmajor_wav):
    get_settings.cache_clear()
    monkeypatch.setenv("JAMS_KEY_BACKEND", "cnn")

    class FakeWorker:
        def analyze(self, req):
            raise RuntimeError("key-cnn worker failed: weights missing")

    monkeypatch.setattr(K, "_key_cnn_worker", lambda: FakeWorker())
    with pytest.raises(RuntimeError, match="weights missing"):
        K.detect_key(cmajor_wav)
    get_settings.cache_clear()


def test_bundled_weights_present():
    # The wheel must carry the CNN weights; absence is a broken install.
    assert (Path(K.__file__).parent.parent / "data" / "models" / "key_cnn_v1.pt").exists()
    assert (Path(K.__file__).parent.parent / "data" / "key_cnn_worker.py").exists()
