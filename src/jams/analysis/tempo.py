"""Tempo (BPM) detection with genre-aware octave resolution.

Method cascade (best first, each falling through on failure):
  1. TempoCNN ``deepsquare`` (pretrained, bundled)  — Acc1 0.92 on GiantSteps
  2. Essentia ``RhythmExtractor2013`` (multifeature) — Acc1 0.93
  3. librosa beat-tracker                            — Acc1 0.83

Trackers nail the BPM *value* but can land an octave off (half/double-time), and that
error concentrates in genres with a half-time feel. Given a ``genre`` or explicit
``bpm_range``, the result is folded into the expected octave. Critically, D&B/jungle
are conventionally FULL tempo (~174) — folding them to half-time only matches mislabeled
metadata (the bug we found while validating against GiantSteps-Tempo v2).

Against corrected labels this reaches global Acc1 0.965 (D&B 0.79, Dubstep 0.87).
"""

from __future__ import annotations

import logging
import math
import threading
from functools import lru_cache
from pathlib import Path

from jams.analysis.audio import load_mono, validate_audio_path

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "deepsquare-k16-3.pb"

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

# Essentia algorithm state isn't thread-safe; serialize use of the cached instances.
_essentia_lock = threading.Lock()


@lru_cache(maxsize=1)
def _tempocnn():
    import essentia
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    import essentia.standard as es

    if _MODEL_PATH.exists() and hasattr(es, "TempoCNN"):
        return es.TempoCNN(graphFilename=str(_MODEL_PATH))
    return None


@lru_cache(maxsize=1)
def _rhythm2013():
    import essentia
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    import essentia.standard as es

    return es.RhythmExtractor2013(method="multifeature")


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
    # 1. TempoCNN
    try:
        with _essentia_lock:
            model = _tempocnn()
            if model is not None:
                audio = load_mono(path, 11025)
                return float(model(audio)[0]), "tempocnn-deepsquare"
    except Exception as exc:
        logger.warning("TempoCNN unavailable (%s); trying RhythmExtractor2013", exc)

    # 2. RhythmExtractor2013
    try:
        with _essentia_lock:
            audio = load_mono(path, 44100)
            return float(_rhythm2013()(audio)[0]), "rhythm2013"
    except Exception as exc:
        logger.warning("RhythmExtractor2013 unavailable (%s); using librosa fallback", exc)

    # 3. librosa
    import librosa
    import numpy as np

    y = load_mono(path, 22050)
    bpm = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=22050)[0])[0])
    return bpm, "librosa"


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
    bpm = resolve_tempo_octave(raw_bpm, bpm_range=bpm_range, genre=genre, default_octave=default_octave)
    return {
        "bpm": round(bpm, 2),
        "bpm_raw": round(raw_bpm, 2),
        "bpm_alt": round(bpm * 2 if bpm < 100 else bpm / 2, 2),
        "octave_resolved": abs(bpm - raw_bpm) > 0.01,
        "method": method,
    }
