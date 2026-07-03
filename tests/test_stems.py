"""Unit tests for the stems orchestration, worker protocol, and pure helpers.

The two uv workers (demucs/basic-pitch + ADTOF drums) are mocked so nothing heavy runs.
The pure-python helpers in ``jams.analysis.gm`` (GM canon, quantize) and the worker's
monophonic filter are tested directly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from jams.analysis import gm
from jams.analysis import stems as S
from jams.config import Settings

_STEMS_RESULT = {
    "stems": [{"stem_type": "drums", "audio_path": "/t/drums.wav"},
              {"stem_type": "bass", "audio_path": "/t/bass.wav"}],
    "transcriptions": [
        {"stem_type": "bass", "gm_program": 33, "is_drums": False,
         "notes": [{"onset": 0.0, "offset": 0.5, "pitch": 40, "velocity": 90}],
         "method": "basic-pitch"},
    ],
    "duration_sec": 12.3,
}
_DRUM_RESULT = {"notes": [{"onset": 0.0, "offset": 0.05, "pitch": 35, "velocity": 100}]}


# --- worker protocol (_Worker) ---------------------------------------------


def test_worker_roundtrip(monkeypatch):
    w = S._Worker(S._STEMS_WORKER, "stems")
    monkeypatch.setattr(w, "_ensure_alive", lambda: None)
    monkeypatch.setattr(
        w, "_round_trip", lambda req: json.dumps({"ok": True, "result": {"x": 1}}) + "\n"
    )
    assert w.analyze({"audio": "/a.wav"})["x"] == 1


def test_worker_respawns_on_dead_pipe(monkeypatch):
    w = S._Worker(S._STEMS_WORKER, "stems")
    spawned = {"n": 0}
    calls = {"n": 0}
    monkeypatch.setattr(w, "_ensure_alive", lambda: None)
    monkeypatch.setattr(w, "_spawn", lambda: spawned.__setitem__("n", spawned["n"] + 1))

    def rt(req):
        calls["n"] += 1
        return "" if calls["n"] == 1 else json.dumps({"ok": True, "result": {"x": 2}}) + "\n"

    monkeypatch.setattr(w, "_round_trip", rt)
    assert w.analyze({"audio": "/a.wav"})["x"] == 2
    assert spawned["n"] == 1  # respawned exactly once


def test_worker_surfaces_error(monkeypatch):
    w = S._Worker(S._DRUM_WORKER, "drums")
    monkeypatch.setattr(w, "_ensure_alive", lambda: None)
    monkeypatch.setattr(
        w, "_round_trip", lambda req: json.dumps({"ok": False, "error": "boom"}) + "\n"
    )
    with pytest.raises(RuntimeError, match="boom"):
        w.analyze({"drums_wav": "/d.wav"})


# --- orchestration (analyze_stems) -----------------------------------------


class _FakeWorker:
    def __init__(self, result):
        self.result = result
        self.reqs: list[dict] = []

    def analyze(self, req):
        self.reqs.append(req)
        return self.result


def test_analyze_stems_merges_pitched_and_drums(monkeypatch, tmp_path, cmajor_wav):
    monkeypatch.setattr(S, "get_settings",
                        lambda: Settings(stems_transcriber="basic-pitch"))
    stems_w = _FakeWorker(_STEMS_RESULT)
    drum_w = _FakeWorker(_DRUM_RESULT)
    monkeypatch.setattr(S, "_stems_worker", lambda: stems_w)
    monkeypatch.setattr(S, "_drum_worker", lambda: drum_w)

    out = S.analyze_stems(cmajor_wav, out_dir=str(tmp_path), quantize=False)

    types = [t["stem_type"] for t in out["transcriptions"]]
    assert types == ["drums", "bass"]  # sorted drums-first
    drums = out["transcriptions"][0]
    assert drums["is_drums"] and drums["notes"][0]["pitch"] == gm.GM_KICK  # 35 -> canon 36
    bass = out["transcriptions"][1]
    assert bass["notes"][0]["pitch"] == 52  # 40 + BASS_OCTAVE_SHIFT applied by orchestrator
    assert out["method"] == "scnet_xl_ihf+basic-pitch+adtof"  # SCNet is the default separator
    # basic-pitch selected -> stems_worker asked to transcribe
    assert stems_w.reqs[0]["transcribe"] is True
    # MIDI files written for each stem + combined
    assert (tmp_path / "drums.mid").exists() and (tmp_path / "combined.mid").exists()
    assert set(out["midi_paths"]) == {"drums", "bass", "combined"}
    # separation mode passes audio, drums worker gets the drums stem wav
    assert stems_w.reqs[0]["audio"] == cmajor_wav
    assert drum_w.reqs[0]["drums_wav"] == "/t/drums.wav"


def test_analyze_stems_yourmt3_routing(monkeypatch, tmp_path, cmajor_wav):
    """Default transcriber: pitched stems come from the yourmt3 worker, mono-filtered."""
    monkeypatch.setattr(S, "get_settings", lambda: Settings())  # yourmt3 default
    sep_only = {"stems": _STEMS_RESULT["stems"], "transcriptions": [], "duration_sec": 12.3}
    stems_w = _FakeWorker(sep_only)
    drum_w = _FakeWorker(_DRUM_RESULT)
    ymt3 = _FakeWorker({"notes": [
        {"onset": 0.0, "offset": 1.0, "pitch": 40, "velocity": 80},
        {"onset": 0.5, "offset": 1.5, "pitch": 45, "velocity": 100},  # overlap, louder
    ]})
    monkeypatch.setattr(S, "_stems_worker", lambda: stems_w)
    monkeypatch.setattr(S, "_drum_worker", lambda: drum_w)
    monkeypatch.setattr(S, "_yourmt3_worker", lambda: ymt3)

    out = S.analyze_stems(cmajor_wav, out_dir=str(tmp_path), quantize=False)

    assert stems_w.reqs[0]["transcribe"] is False  # basic-pitch skipped
    assert [r["audio"] for r in ymt3.reqs] == ["/t/bass.wav"]  # only pitched stems present
    bass = next(t for t in out["transcriptions"] if t["stem_type"] == "bass")
    assert bass["method"] == "yourmt3"
    # mono-filter kept the louder overlapping note; orchestrator applied +12
    assert [n["pitch"] for n in bass["notes"]] == [45 + 12]
    assert out["method"] == "scnet_xl_ihf+yourmt3+adtof"


def test_analyze_stems_oracle_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "get_settings",
                        lambda: Settings(stems_transcriber="basic-pitch"))
    # oracle: only a bass stem provided -> no drums worker call, no drums transcription
    stems_res = {"stems": [{"stem_type": "bass", "audio_path": "/gt/bass.wav"}],
                 "transcriptions": _STEMS_RESULT["transcriptions"], "duration_sec": 5.0}
    stems_w = _FakeWorker(stems_res)
    drum_w = _FakeWorker(_DRUM_RESULT)
    monkeypatch.setattr(S, "_stems_worker", lambda: stems_w)
    monkeypatch.setattr(S, "_drum_worker", lambda: drum_w)

    out = S.analyze_stems(None, stems={"bass": "/gt/bass.wav"}, out_dir=str(tmp_path))
    assert stems_w.reqs[0]["stems"] == {"bass": "/gt/bass.wav"}
    assert "audio" not in stems_w.reqs[0]
    assert drum_w.reqs == []  # no drums stem -> drum worker never called
    assert out["method"] == "oracle-stems+basic-pitch"  # no adtof suffix


def test_analyze_stems_requires_path_or_stems():
    with pytest.raises(ValueError, match="path.*stems"):
        S.analyze_stems(None)


# --- pure helpers (jams.analysis.gm) ---------------------------------------


def test_canon_drum_pitch_maps_to_gm():
    assert gm.canon_drum_pitch(35) == gm.GM_KICK
    assert gm.canon_drum_pitch(37) == gm.GM_SNARE  # side-stick -> snare bucket
    assert gm.canon_drum_pitch(26) == gm.GM_OPEN_HAT
    assert gm.canon_drum_pitch(99) == 99  # unknown passes through


def test_quantize_snaps_onset_to_grid():
    beats = [0.0, 1.0, 2.0]  # 1s beats; 4 subdivisions => grid every 0.25s
    notes = [{"onset": 0.28, "offset": 0.9, "pitch": 60, "velocity": 90}]
    out = gm.quantize_notes(notes, beats, subdivisions=4)
    assert abs(out[0]["onset"] - 0.25) < 1e-9
    assert out[0]["offset"] > out[0]["onset"]


def test_quantize_noop_without_beats():
    notes = [{"onset": 0.28, "offset": 0.9, "pitch": 60, "velocity": 90}]
    assert gm.quantize_notes(notes, []) == notes


def _load_worker():
    path = Path(S.__file__).resolve().parent.parent / "data" / "stems_worker.py"
    spec = importlib.util.spec_from_file_location("stems_worker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_monophonic_filter_drops_overlaps_keeping_loudest():
    m = _load_worker()
    notes = [
        {"onset": 0.0, "offset": 1.0, "pitch": 40, "velocity": 80},
        {"onset": 0.5, "offset": 1.5, "pitch": 45, "velocity": 100},  # overlaps, louder
        {"onset": 2.0, "offset": 3.0, "pitch": 50, "velocity": 60},  # no overlap
    ]
    out = m._monophonic_filter(notes)
    assert [n["pitch"] for n in out] == [45, 50]


def test_scnet_model_name_routing():
    m = _load_worker()
    assert m._is_scnet("scnet_xl_ihf")
    assert m._is_scnet("SCNet-large")
    assert not m._is_scnet("htdemucs")
    assert not m._is_scnet("htdemucs_ft")


def test_separation_method_string(monkeypatch, tmp_path, cmajor_wav):
    monkeypatch.setattr(S, "get_settings",
                        lambda: Settings(stems_model="htdemucs", stems_transcriber="basic-pitch"))
    stems_w = _FakeWorker(_STEMS_RESULT)
    monkeypatch.setattr(S, "_stems_worker", lambda: stems_w)
    monkeypatch.setattr(S, "_drum_worker", lambda: _FakeWorker(_DRUM_RESULT))
    out = S.analyze_stems(cmajor_wav, out_dir=str(tmp_path), quantize=False)
    assert out["method"].startswith("demucs-htdemucs")  # explicit htdemucs opt-out works


def test_shift_bass_notes_caps_at_127():
    out = gm.shift_bass_notes([{"onset": 0, "offset": 1, "pitch": 40, "velocity": 90},
                               {"onset": 1, "offset": 2, "pitch": 120, "velocity": 90}])
    assert [n["pitch"] for n in out] == [52, 127]


def test_gm_monophonic_filter_keeps_loudest_overlap():
    notes = [
        {"onset": 0.0, "offset": 1.0, "pitch": 40, "velocity": 80},
        {"onset": 0.5, "offset": 1.5, "pitch": 45, "velocity": 100},
        {"onset": 2.0, "offset": 3.0, "pitch": 50, "velocity": 60},
    ]
    assert [n["pitch"] for n in gm.monophonic_filter(notes)] == [45, 50]
