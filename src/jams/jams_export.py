"""Serialize a native analysis result to a JAMS document (jams.org spec, v0.3.3).

JAMS (JSON Annotated Music Specification) is the lingua franca for MIR annotations —
and the format the Harmonix Set we benchmark against ships in. Rather than enrich our
native REST schema with per-observation time/duration/confidence, we keep the native
format lean and map it here: every JAMS field is derivable from what the analyzers
already return (beat times, segment spans, key/tempo + confidence), and provenance is
synthesized from each analyzer's ``method`` string.

Namespaces emitted: ``key_mode`` (key), ``tempo`` (bpm), ``beat`` + ``segment_open``
(structure). Returns a plain dict that ``json.dumps`` cleanly and matches the structure
of a real Harmonix ``.jams`` file (validate against the official ``jams`` schema if needed;
we don't import the PyPI ``jams`` package because its import name collides with ours).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

JAMS_VERSION = "0.3.3"

try:
    _PKG_VERSION = version("jams")
except PackageNotFoundError:  # pragma: no cover - editable/uninstalled
    _PKG_VERSION = "0"


def _annotation_metadata(method: str | None) -> dict:
    """JAMS annotation_metadata with provenance pointing back at the producing method."""
    return {
        "curator": {"name": "", "email": ""},
        "annotator": {},
        "version": _PKG_VERSION,
        "corpus": "",
        "annotation_tools": method or "",
        "annotation_rules": "",
        "validation": "",
        "data_source": "jams MIR service",
    }


def _annotation(namespace: str, data: list[dict], method: str | None,
                duration: float | None, sandbox: dict | None = None) -> dict:
    return {
        "namespace": namespace,
        "annotation_metadata": _annotation_metadata(method),
        "data": data,
        "sandbox": sandbox or {},
        "time": 0.0,
        "duration": duration,
    }


def _obs(time: float, duration: float, value, confidence) -> dict:
    return {"time": float(time), "duration": float(duration), "value": value,
            "confidence": confidence}


def _beat_positions(beats: list[float], downbeats: list[float]) -> list[int]:
    """Reconstruct per-beat metric position (1 at each downbeat, incrementing otherwise)."""
    downs = sorted(downbeats)
    positions: list[int] = []
    di, pos = 0, 0
    for b in beats:
        if di < len(downs) and abs(b - downs[di]) < 1e-3:
            pos, di = 1, di + 1
        else:
            pos = pos + 1 if pos else 1
        positions.append(pos)
    return positions


def to_jams(result: dict, *, filename: str | None = None) -> dict:
    """Build a JAMS document from a native ``analyze_track`` result (+ filename).

    ``result`` may contain any of ``key``, ``tempo``, ``structure`` plus ``duration_sec``.
    """
    duration = result.get("duration_sec")
    annotations: list[dict] = []

    key = result.get("key")
    if key:
        value = f"{key['tonic']}:{key['mode']}"
        annotations.append(_annotation(
            "key_mode",
            [_obs(0.0, duration or 0.0, value, key.get("confidence"))],
            key.get("method"), duration,
        ))

    tempo = result.get("tempo")
    if tempo:
        sandbox = {k: tempo[k] for k in ("bpm_raw", "bpm_alt", "octave_resolved") if k in tempo}
        annotations.append(_annotation(
            "tempo",
            [_obs(0.0, duration or 0.0, tempo["bpm"], None)],
            tempo.get("method"), duration, sandbox,
        ))

    structure = result.get("structure")
    if structure:
        method = structure.get("method")
        beats = [float(b) for b in structure.get("beats", [])]
        downbeats = [float(d) for d in structure.get("downbeats", [])]
        if beats:
            positions = _beat_positions(beats, downbeats)
            annotations.append(_annotation(
                "beat",
                [_obs(t, 0.0, p, None) for t, p in zip(beats, positions, strict=True)],
                method, duration,
            ))
        segments = structure.get("segments", [])
        if segments:
            annotations.append(_annotation(
                "segment_open",
                [_obs(s["start"], s["end"] - s["start"], s["label"], None) for s in segments],
                method, duration,
            ))

    return {
        "annotations": annotations,
        "file_metadata": {
            "title": filename or "",
            "artist": "",
            "release": "",
            "duration": duration,
            "identifiers": {},
            "jams_version": JAMS_VERSION,
        },
        "sandbox": {},
    }
