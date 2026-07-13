"""Unit tests for the tempo CNN worker wiring + protocol.

These mock the subprocess round-trip so nothing heavy (uv/torch) runs; the pattern
mirrors tests/test_key_cnn.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jams.analysis import stems as S
from jams.analysis import tempo as T
from jams.config import Settings

_RESULT = {"bpm": 174.0, "confidence": 0.42, "method": "tempo-cnn-v1"}
_OK = json.dumps({"ok": True, "result": _RESULT}) + "\n"


@pytest.fixture(autouse=True)
def _fresh_singleton():
    T._tempo_cnn_singleton = None
    yield
    T._tempo_cnn_singleton = None


def test_worker_roundtrip(monkeypatch):
    w = S._Worker(Path("/nonexistent.py"), "tempo-cnn", uv_setting="tempo_cnn_uv")
    monkeypatch.setattr(w, "_ensure_alive", lambda: None)
    monkeypatch.setattr(w, "_round_trip", lambda req: _OK)
    out = w.analyze({"audio": "/a.wav"})
    assert out["bpm"] == 174.0 and out["method"] == "tempo-cnn-v1"


def test_worker_respawns_on_dead_pipe(monkeypatch):
    w = S._Worker(Path("/nonexistent.py"), "tempo-cnn", uv_setting="tempo_cnn_uv")
    spawned = {"n": 0}
    calls = {"n": 0}
    monkeypatch.setattr(w, "_ensure_alive", lambda: None)
    monkeypatch.setattr(w, "_spawn", lambda: spawned.__setitem__("n", spawned["n"] + 1))

    def rt(req):
        calls["n"] += 1
        return "" if calls["n"] == 1 else _OK  # first call: worker is dead

    monkeypatch.setattr(w, "_round_trip", rt)
    out = w.analyze({"audio": "/a.wav"})
    assert out["bpm"] == 174.0
    assert spawned["n"] == 1  # respawned exactly once


def test_worker_uses_tempo_cnn_uv_setting(monkeypatch):
    monkeypatch.setattr(
        "jams.analysis.stems.get_settings",
        lambda: Settings(tempo_cnn_uv="/custom/uv", _env_file=None),
    )
    w = S._Worker(Path("/nonexistent.py"), "tempo-cnn", uv_setting="tempo_cnn_uv")
    captured = {}

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        raise RuntimeError("stop before real spawn")

    monkeypatch.setattr("jams.analysis.stems.subprocess.Popen", fake_popen)
    with pytest.raises(RuntimeError, match="stop before real spawn"):
        w._spawn()
    assert captured["cmd"][0] == "/custom/uv"


def test_detect_tempo_uses_worker_and_resolves_octave(monkeypatch, cmajor_wav):
    class FakeWorker:
        def analyze(self, req):
            assert req["audio"] == cmajor_wav
            return {"bpm": 87.0, "confidence": 0.9, "method": "tempo-cnn-v1"}

    monkeypatch.setattr(T, "_tempo_cnn_worker", lambda: FakeWorker())
    out = T.detect_tempo(cmajor_wav, genre="Drum & Bass")
    assert out["bpm"] == pytest.approx(174.0)
    assert out["bpm_raw"] == pytest.approx(87.0)
    assert out["octave_resolved"] is True
    assert out["method"] == "tempo-cnn-v1"


def test_detect_tempo_no_hint_keeps_raw(monkeypatch, cmajor_wav):
    class FakeWorker:
        def analyze(self, req):
            return {"bpm": 151.0, "confidence": 0.5, "method": "tempo-cnn-v1"}

    monkeypatch.setattr(T, "_tempo_cnn_worker", lambda: FakeWorker())
    out = T.detect_tempo(cmajor_wav)
    assert out["bpm"] == pytest.approx(151.0)
    assert out["octave_resolved"] is False


def test_bundled_weights_present():
    # The wheel must carry the CNN weights; absence is a broken install.
    data = Path(T.__file__).parent.parent / "data"
    assert (data / "models" / "tempo_cnn_v1.pt").exists()
    assert (data / "tempo_cnn_worker.py").exists()


def test_old_tempocnn_graph_gone():
    # The retired TempoCNN graph must not ride along in the package.
    data = Path(T.__file__).parent.parent / "data"
    assert not (data / "models" / "deepsquare-k16-3.pb").exists()
    assert not list((data / "models").glob("*.pb"))
