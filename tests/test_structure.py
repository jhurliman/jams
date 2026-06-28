"""Unit tests for the structure backend dispatch + local worker protocol.

These mock the subprocess round-trip so nothing heavy (uv/torch/All-In-One) runs.
"""

from __future__ import annotations

import json

import pytest

from jams.analysis import structure as S
from jams.config import Settings

_RESULT = {"bpm": 174.0, "beats": [], "downbeats": [], "segments": [], "method": "x"}
_OK = json.dumps({"ok": True, "result": _RESULT}) + "\n"


def test_local_worker_roundtrip(monkeypatch):
    w = S._LocalWorker()
    monkeypatch.setattr(w, "_ensure_alive", lambda: None)
    monkeypatch.setattr(w, "_round_trip", lambda req: _OK)
    out = w.analyze("/a.wav", 174.0, "harmonix-all")
    assert out["bpm"] == 174.0 and out["method"] == "x"


def test_local_worker_respawns_on_dead_pipe(monkeypatch):
    w = S._LocalWorker()
    spawned = {"n": 0}
    calls = {"n": 0}
    monkeypatch.setattr(w, "_ensure_alive", lambda: None)
    monkeypatch.setattr(w, "_spawn", lambda: spawned.__setitem__("n", spawned["n"] + 1))

    def rt(req):
        calls["n"] += 1
        return "" if calls["n"] == 1 else _OK  # first call: worker is dead

    monkeypatch.setattr(w, "_round_trip", rt)
    out = w.analyze("/a.wav", None, "m")
    assert out["bpm"] == 174.0
    assert spawned["n"] == 1  # respawned exactly once


def test_local_worker_surfaces_worker_error(monkeypatch):
    w = S._LocalWorker()
    monkeypatch.setattr(w, "_ensure_alive", lambda: None)
    monkeypatch.setattr(
        w, "_round_trip", lambda req: json.dumps({"ok": False, "error": "boom"}) + "\n"
    )
    with pytest.raises(RuntimeError, match="boom"):
        w.analyze("/a.wav", None, "m")


def test_replicate_backend_requires_token(monkeypatch, cmajor_wav):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.setattr(
        S, "get_settings",
        lambda: Settings(structure_backend="replicate", replicate_api_token=None),
    )
    with pytest.raises(RuntimeError, match="token"):
        S.analyze_structure(cmajor_wav)


def test_local_backend_dispatches_to_worker(monkeypatch, cmajor_wav):
    monkeypatch.setattr(S, "get_settings", lambda: Settings(structure_backend="local"))
    captured = {}

    class FakeWorker:
        def analyze(self, audio, target_bpm, model):
            captured.update(audio=audio, target_bpm=target_bpm, model=model)
            return {"method": "fake"}

    monkeypatch.setattr(S, "_local_worker", lambda: FakeWorker())
    out = S.analyze_structure(cmajor_wav, target_bpm=174.0, model="harmonix-fold3")
    assert out["method"] == "fake"
    assert captured == {"audio": cmajor_wav, "target_bpm": 174.0, "model": "harmonix-fold3"}
