"""Activation-blob resegmentation (the section-count slider backend).

Pure numpy — the shared implementation lives in ``data/structure_worker.py`` (stdlib-only
at import time) and is exercised both directly and through the jams API. No model runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from jams.analysis import structure as S
from jams.analysis.structure import resegment_structure
from jams.api.app import app

SW = S._worker_module()
LABELS = list(SW._RAVEFORM_LABELS)

DURATION = 200.0
LABEL_FPS = 5.0
# Boundary candidates at 50 s / 100 s / 150 s with descending peak strengths.
CANDIDATES = [[5000, 0.5], [10000, 0.3], [15000, 0.1]]


def _label_probs(spans, duration=DURATION, fps=LABEL_FPS, runner_up=None):
    """Rows of per-frame class probabilities: the span's label dominates at 0.9
    (optionally a fixed runner_up class everywhere at 0.5)."""
    rows = []
    for i in range(int(duration * fps)):
        t = i / fps
        row = [0.0] * len(LABELS)
        if runner_up:
            row[LABELS.index(runner_up)] = 0.5
        for start, end, label in spans:
            if start <= t < end:
                row[LABELS.index(label)] = 0.9
                break
        rows.append(row)
    return rows


def _blob(spans=((0, 50, "intro"), (50, 150, "drop"), (150, 200, "outro")),
          candidates=CANDIDATES, threshold=0.2, runner_up=None):
    return {
        "version": 1,
        "duration": DURATION,
        "frame_rate": 100.0,
        "candidates": candidates,
        "labels": LABELS,
        "label_frame_rate": LABEL_FPS,
        "label_probs": _label_probs(spans, runner_up=runner_up),
        "threshold": threshold,
    }


def test_threshold_controls_section_count():
    for thr, expected in ((0.6, 1), (0.4, 2), (0.2, 3), (0.05, 4)):
        out = resegment_structure(_blob(), threshold=thr)
        assert len(out["segments"]) == expected, thr
    # boundaries land exactly on the candidate frames
    starts = [s["start"] for s in resegment_structure(_blob(), threshold=0.05)["segments"]]
    assert starts == [0.0, 50.0, 100.0, 150.0]


def test_target_sections_picks_the_threshold():
    thresholds = []
    for k in (1, 2, 3, 4):
        out = resegment_structure(_blob(), target_sections=k)
        assert len(out["segments"]) == k
        thresholds.append(out["threshold"])
    assert thresholds == sorted(thresholds, reverse=True)
    # asking for more sections than there are candidates saturates at the max
    assert len(resegment_structure(_blob(), target_sections=10)["segments"]) == 4


def test_default_reuses_the_analysis_time_threshold():
    out = resegment_structure(_blob(threshold=0.2))
    assert len(out["segments"]) == 3
    assert out["threshold"] == 0.2


def test_labels_come_from_span_means():
    segs = resegment_structure(_blob(), threshold=0.05)["segments"]
    assert [s["label"] for s in segs] == ["intro", "drop", "drop", "outro"]


def test_positional_prior_masks_late_intros():
    # 'intro' dominant everywhere -> only the opening span may keep it.
    blob = _blob(spans=((0, 200, "intro"),), runner_up="drop")
    segs = resegment_structure(blob, threshold=0.05)["segments"]
    assert [s["label"] for s in segs] == ["intro", "drop", "drop", "drop"]


def test_positional_prior_masks_early_outros():
    blob = _blob(spans=((0, 200, "outro"),), runner_up="drop")
    segs = resegment_structure(blob, threshold=0.05)["segments"]
    # outro becomes legal from the halfway point (start frac >= 0.5)
    assert [s["label"] for s in segs] == ["drop", "drop", "outro", "outro"]


def test_short_mislabelled_opening_snaps_to_intro():
    blob = _blob(spans=((0, 10, "buildup"), (10, 200, "drop")),
                 candidates=[[1000, 0.5]])
    segs = resegment_structure(blob)["segments"]
    assert [s["label"] for s in segs] == ["intro", "drop"]


def test_beats_fill_beat_indices():
    beats = [i * 0.5 for i in range(401)]
    with_beats = resegment_structure(_blob(), threshold=0.4, beats=beats)["segments"]
    assert all(
        isinstance(s["start_beat"], int) and isinstance(s["end_beat"], int)
        for s in with_beats
    )
    assert with_beats[0]["start_beat"] == 1
    without = resegment_structure(_blob(), threshold=0.4)["segments"]
    assert without[0]["start_beat"] is None and without[0]["end_beat"] is None


def test_threshold_and_target_sections_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        resegment_structure(_blob(), threshold=0.2, target_sections=3)


def test_blob_roundtrip_matches_the_full_resolution_path():
    """resegment(blob at analysis threshold) == the model path's segments."""
    frame_rate = 100.0
    n = int(DURATION * frame_rate)
    probs = np.zeros((len(LABELS), n))
    for start, end, label in ((0, 50, "intro"), (50, 150, "drop"), (150, 200, "outro")):
        probs[LABELS.index(label), int(start * frame_rate):int(end * frame_rate)] = 0.9
    cand_frames = np.array([c[0] for c in CANDIDATES])
    cand_strengths = np.array([c[1] for c in CANDIDATES])

    thr = 0.2
    full = SW._candidate_segments(
        cand_frames, cand_strengths, thr, frame_rate, probs, frame_rate, DURATION, LABELS
    )
    full = SW._fix_boundary_labels(full, DURATION)

    blob = SW._activations_blob({
        "cand_frames": cand_frames, "cand_strengths": cand_strengths,
        "prob_functions": probs, "frame_rate": frame_rate,
        "duration": DURATION, "threshold": thr, "labels": LABELS,
    })
    assert blob["label_frame_rate"] == LABEL_FPS
    via_blob = resegment_structure(blob)["segments"]

    assert [(s["start"], s["end"], s["label"]) for s in via_blob] == [
        (pytest.approx(st), pytest.approx(en), lab) for st, en, lab in full
    ]


def test_empty_label_probs_is_rejected():
    blob = _blob()
    blob["label_probs"] = []
    with pytest.raises(ValueError, match="label_probs"):
        resegment_structure(blob)


# ----------------------------------------------------------------- API contract
client = TestClient(app)


def test_api_resegment_roundtrip():
    r = client.post("/v1/resegment", json={"activations": _blob(), "target_sections": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["segments"]) == 3
    assert body["segments"][0]["label"] in LABELS
    assert body["threshold"] > 0


def test_api_resegment_rejects_conflicting_params():
    r = client.post(
        "/v1/resegment",
        json={"activations": _blob(), "target_sections": 3, "threshold": 0.2},
    )
    assert r.status_code == 422


def test_api_resegment_rejects_empty_blob():
    blob = _blob()
    blob["label_probs"] = []
    r = client.post("/v1/resegment", json={"activations": blob})
    assert r.status_code == 422
