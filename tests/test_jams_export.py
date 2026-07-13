"""Tests for JAMS serialization (?format=jams) — structure, provenance, API wiring."""

from __future__ import annotations

from fastapi.testclient import TestClient

from jams.api.app import app
from jams.jams_export import JAMS_VERSION, to_jams

client = TestClient(app)

NATIVE = {
    "duration_sec": 120.0,
    "key": {"key": "C major", "tonic": "C", "mode": "major",
            "confidence": 0.8, "method": "key-cnn-v1"},
    "tempo": {"bpm": 174.0, "bpm_raw": 87.0, "bpm_alt": 87.0,
              "octave_resolved": True, "method": "tempo-cnn-v1"},
    "structure": {"bpm": 174.0, "beats": [0.0, 0.34, 0.69, 1.03], "downbeats": [0.0],
                  "segments": [{"start": 0.0, "end": 30.0, "label": "intro"}],
                  "method": "allin1-mps-local:harmonix-all"},
}


def test_top_level_shape():
    d = to_jams(NATIVE, filename="t.wav")
    assert set(d) == {"annotations", "file_metadata", "sandbox"}
    assert d["file_metadata"]["jams_version"] == JAMS_VERSION
    assert d["file_metadata"]["duration"] == 120.0
    assert d["file_metadata"]["title"] == "t.wav"


def test_namespaces_in_order():
    ns = [a["namespace"] for a in to_jams(NATIVE)["annotations"]]
    assert ns == ["key_mode", "tempo", "beat", "segment_open"]


def test_key_mode_value_and_provenance():
    a = next(a for a in to_jams(NATIVE)["annotations"] if a["namespace"] == "key_mode")
    obs = a["data"][0]
    assert obs["value"] == "C:major"
    assert obs["confidence"] == 0.8
    assert obs["time"] == 0.0 and obs["duration"] == 120.0
    assert a["annotation_metadata"]["annotation_tools"] == "key-cnn-v1"
    assert a["annotation_metadata"]["data_source"] == "jams MIR service"


def test_beat_positions_reconstructed():
    beat = next(a for a in to_jams(NATIVE)["annotations"] if a["namespace"] == "beat")
    assert [o["value"] for o in beat["data"]] == [1, 2, 3, 4]  # downbeat at 0.0 -> resets to 1


def test_segment_open_span():
    seg = next(a for a in to_jams(NATIVE)["annotations"] if a["namespace"] == "segment_open")
    obs = seg["data"][0]
    assert (obs["time"], obs["duration"], obs["value"]) == (0.0, 30.0, "intro")


def test_tempo_extras_in_sandbox():
    t = next(a for a in to_jams(NATIVE)["annotations"] if a["namespace"] == "tempo")
    assert t["data"][0]["value"] == 174.0
    assert t["sandbox"]["bpm_raw"] == 87.0 and t["sandbox"]["octave_resolved"] is True


def test_absent_analyses_omitted():
    d = to_jams({"duration_sec": 10.0, "key": NATIVE["key"]})
    assert [a["namespace"] for a in d["annotations"]] == ["key_mode"]


def test_api_format_jams(cmajor_wav):
    r = client.post("/v1/analyze/path?format=jams", json={"path": cmajor_wav, "tempo": False})
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) == {"annotations", "file_metadata", "sandbox"}
    km = next(a for a in d["annotations"] if a["namespace"] == "key_mode")
    assert km["data"][0]["value"].startswith("C:")


def test_api_format_native_unchanged(cmajor_wav):
    r = client.post("/v1/analyze/path", json={"path": cmajor_wav, "tempo": False})
    assert r.status_code == 200, r.text
    assert r.json()["key"]["key"] == "C major"  # native schema intact
