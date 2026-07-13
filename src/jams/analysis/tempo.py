"""Tempo (BPM) detection with genre-aware octave resolution.

Method: our own 256-class tempo CNN (TP1; MIT, ~2.9 M params, weights bundled at
``data/models/tempo_cnn_v1.pt``, run in a uv worker — ``data/tempo_cnn_worker.py``).
A clean-room implementation of the Schreiber & Müller (ISMIR 2018) single-step family,
trained on Raveform + GiantSteps-Tempo v2 (minus the 42 tracks overlapping the
GiantSteps Key eval set). GiantSteps Key tempo protocol (n=458, one pre-registered
evaluation, corrected labels primary): Acc1 **0.967** — statistically non-inferior to
the previous production system (paired ΔAcc1 −0.0022, 95% CI [−0.0153, +0.0109]; see
``paper/EXPERIMENTS.md`` TP1). Method string: ``tempo-cnn-v1``.

There is deliberately no fallback tracker: earlier versions silently degraded to
RhythmExtractor2013 or librosa on any import/runtime error, which made accuracy depend
on installation accidents. A broken worker raises a clear error instead of quietly
returning worse numbers.

Trackers nail the BPM *value* but can land an octave off (half/double-time), and that
error concentrates in genres with a half-time feel. Given a ``genre`` or explicit
``bpm_range``, the result is folded into the expected octave. Critically, D&B/jungle
are conventionally FULL tempo (~174) — folding them to half-time only matches mislabeled
metadata (the bug we found while validating against GiantSteps-Tempo v2).
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

from jams.analysis.audio import validate_audio_path

logger = logging.getLogger(__name__)

_TEMPO_CNN_WORKER_PATH = Path(__file__).resolve().parent.parent / "data" / "tempo_cnn_worker.py"

# Canonical octave (lower bound of a [lo, 2*lo) window) per genre, matched
# case-insensitively as a substring of the genre string.
_GENRE_TEMPO_RANGES: dict[str, float] = {
    # Full-tempo by convention (NOT half-time):
    "drum & bass": 110.0, "drum and bass": 110.0, "dnb": 110.0, "jungle": 110.0,
    "footwork": 110.0,
    "dubstep": 96.0, "future bass": 96.0, "trap": 96.0,  # ~140
    "halftime": 70.0,
}
# Generic DJ octave for fold_default=True when genre/range is unknown.
_DEFAULT_TEMPO_OCTAVE = 84.0

_tempo_cnn_singleton = None
_tempo_cnn_lock = threading.Lock()


def _tempo_cnn_worker():
    """Resident tempo-CNN uv worker (same subprocess pattern as the stems workers)."""
    global _tempo_cnn_singleton
    if _tempo_cnn_singleton is None:
        with _tempo_cnn_lock:
            if _tempo_cnn_singleton is None:
                from jams.analysis.stems import _Worker

                _tempo_cnn_singleton = _Worker(
                    _TEMPO_CNN_WORKER_PATH, "tempo-cnn", uv_setting="tempo_cnn_uv"
                )
    return _tempo_cnn_singleton


def _genre_octave(genre: str | None) -> float | None:
    if not genre:
        return None
    g = genre.lower()
    for key, lo in _GENRE_TEMPO_RANGES.items():
        if key in g:
            return lo
    return None


def resolve_tempo_octave(
    bpm: float,
    bpm_range: tuple[float, float] | None = None,
    genre: str | None = None,
    default_octave: float | None = None,
) -> float:
    """Fold ``bpm`` (correct value, possibly wrong octave) into a target octave.

    Precedence: explicit ``bpm_range`` > ``genre`` lookup > ``default_octave``. With a
    wide range, picks the in-window candidate nearest its geometric center. Returns the
    input unchanged when nothing applies.
    """
    if bpm_range:
        lo, hi = float(bpm_range[0]), float(bpm_range[1])
    else:
        octave = _genre_octave(genre)
        if octave is None:
            octave = default_octave
        if octave is None:
            return bpm
        lo, hi = octave, octave * 2.0
    if bpm <= 0 or lo <= 0 or hi <= lo:
        return bpm
    center = math.sqrt(lo * hi)
    best, best_dist = bpm, None
    for k in range(-4, 5):
        cand = bpm * (2.0**k)
        if lo <= cand < hi:
            dist = abs(math.log2(cand / center))
            if best_dist is None or dist < best_dist:
                best, best_dist = cand, dist
    return best


def _raw_bpm(path: str) -> tuple[float, str]:
    # Tempo CNN worker only — a failure here is an environment or input defect and must
    # surface, not silently downgrade accuracy (see module docstring).
    res = _tempo_cnn_worker().analyze({"audio": path})
    return float(res["bpm"]), res["method"]


def detect_tempo(
    path: str,
    *,
    bpm_range: tuple[float, float] | None = None,
    genre: str | None = None,
    fold_default: bool = False,
) -> dict:
    """Detect global tempo (BPM), optionally octave-resolved by ``genre``/``bpm_range``.

    Returns bpm, bpm_raw, bpm_alt, octave_resolved, method.
    """
    validate_audio_path(path)
    raw_bpm, method = _raw_bpm(path)
    default_octave = _DEFAULT_TEMPO_OCTAVE if fold_default else None
    bpm = resolve_tempo_octave(
        raw_bpm, bpm_range=bpm_range, genre=genre, default_octave=default_octave
    )
    return {
        "bpm": round(bpm, 2),
        "bpm_raw": round(raw_bpm, 2),
        "bpm_alt": round(bpm * 2 if bpm < 100 else bpm / 2, 2),
        "octave_resolved": abs(bpm - raw_bpm) > 0.01,
        "method": method,
    }
