"""API smoke tests against the real analysis stack (Essentia or librosa fallback)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from jams.api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_upload(cmajor_wav):
    with open(cmajor_wav, "rb") as fh:
        r = client.post(
            "/v1/analyze",
            files={"file": ("cmajor.wav", fh, "audio/wav")},
            data={"tempo": "false"},  # key only — fast and deterministic
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "cmajor.wav"
    assert body["key"]["key"] == "C major"
    assert body["key"]["mode"] == "major"


def test_analyze_path(cmajor_wav):
    r = client.post("/v1/analyze/path", json={"path": cmajor_wav, "tempo": False})
    assert r.status_code == 200, r.text
    assert r.json()["key"]["tonic"] == "C"


def test_unsupported_format_rejected():
    r = client.post("/v1/analyze", files={"file": ("x.txt", b"nope", "text/plain")})
    assert r.status_code == 422


def test_missing_path_is_422():
    r = client.post("/v1/analyze/path", json={"path": "/does/not/exist.wav"})
    assert r.status_code == 422
