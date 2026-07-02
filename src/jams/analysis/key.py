"""Musical-key detection.

Tonic: Essentia ``KeyExtractor`` with the EDM-tuned ``edma`` profile (SOTA on
GiantSteps Key). Mode (major/minor): edma's call, refined by a small logistic
classifier trained on chroma cues — the *third* (minor 3rd vs major 3rd above the
tonic) plus the 6th/7th and a bass-register third. Template correlation dilutes the
diagnostic third among the shared scale tones; the classifier targets it directly and
only overrides edma when confident, lifting MIREX 0.759->0.801 (5-fold CV) while
keeping major-key recall.

essentia-tensorflow is a **hard requirement** (wheels for macOS arm64 + Linux x86_64,
CPython 3.14). There is deliberately no fallback detector: earlier versions silently
degraded to librosa Krumhansl-Schmuckler (MIREX 0.801 -> ~0.61) on any import/runtime
error, making accuracy depend on installation accidents. A broken install now raises.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path

from jams.analysis.audio import load_mono, validate_audio_path

logger = logging.getLogger(__name__)

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B", "Fb": "E", "E#": "F", "B#": "C",
}
_MODE_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "mode_model.json"


def _normalize(tonic: str, scale: str) -> tuple[str, str]:
    tonic = FLAT_TO_SHARP.get(tonic, tonic)
    mode = "minor" if "min" in scale.lower() else "major"
    return tonic, mode


@lru_cache(maxsize=1)
def _mode_model() -> dict:
    # Bundled with the package; absence is a broken install, not a soft condition.
    try:
        return json.loads(_MODE_MODEL_PATH.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"Bundled mode model missing/corrupt at {_MODE_MODEL_PATH} — broken install? "
            "Pass detect_key(..., refine_mode=False) only to intentionally skip refinement."
        ) from exc


def _mode_features(path: str, tonic_idx: int) -> list[float]:
    """Chroma cues for the mode decision, anchored at ``tonic_idx`` (pitch class).

    Must match the training feature extraction exactly (librosa harmonic chroma at
    22.05 kHz; treble + bass-register chroma, L2-normalized).
    """
    import librosa
    import numpy as np

    # Use librosa.load to match the training feature extraction byte-for-byte.
    y, _ = librosa.load(path, sr=22050, mono=True)
    yh = librosa.effects.harmonic(y)
    c = np.sum(librosa.feature.chroma_cqt(y=yh, sr=22050), axis=1)
    c = c / (np.linalg.norm(c) + 1e-9)
    b = np.sum(
        librosa.feature.chroma_cqt(y=yh, sr=22050, fmin=librosa.note_to_hz("C2"), n_octaves=3),
        axis=1,
    )
    b = b / (np.linalg.norm(b) + 1e-9)
    t = tonic_idx

    def g(a, iv):
        return float(a[(t + iv) % 12])

    # [m3-M3, bass m3-M3, m6-M6, m7-M7, tonic, fifth, m3, M3]
    return [g(c, 3) - g(c, 4), g(b, 3) - g(b, 4), g(c, 8) - g(c, 9), g(c, 10) - g(c, 11),
            g(c, 0), g(c, 7), g(c, 3), g(c, 4)]


def _refine_mode(path: str, tonic_idx: int, edma_mode: str) -> str:
    """Override edma's mode only when the classifier is confident; else keep edma's."""
    model = _mode_model()
    x = _mode_features(path, tonic_idx)
    z = model["intercept"]
    for xi, mean, scale, coef in zip(
        x, model["mean"], model["scale"], model["coef"], strict=True
    ):
        z += coef * ((xi - mean) / scale)
    p_minor = 1.0 / (1.0 + math.exp(-z))
    thr = model["threshold"]
    if p_minor >= thr:
        return "minor"
    if p_minor <= 1.0 - thr:
        return "major"
    return edma_mode


def _detect_essentia(path: str, refine_mode: bool) -> dict:
    try:
        import essentia
        essentia.log.infoActive = False
        essentia.log.warningActive = False
        import essentia.standard as es
    except ImportError as exc:
        raise RuntimeError(
            "essentia-tensorflow is required for key detection (no fallback by design). "
            "It ships wheels for macOS arm64 and Linux x86_64 on CPython 3.14 — check that "
            "`uv sync` ran on Python 3.14 (.python-version)."
        ) from exc

    audio = load_mono(path, 44100)
    tonic, scale, strength = es.KeyExtractor(profileType="edma")(audio)
    tonic, mode = _normalize(tonic, scale)
    method = "essentia-edma"
    if refine_mode:
        refined = _refine_mode(path, NOTES.index(tonic), mode)
        if refined != mode:
            mode = refined
            method = "essentia-edma+modeclf"
    return {
        "key": f"{tonic} {mode}",
        "tonic": tonic,
        "mode": mode,
        "confidence": round(float(strength), 3),
        "method": method,
    }


def detect_key(path: str, *, refine_mode: bool = True) -> dict:
    """Detect the musical key. Returns key, tonic, mode, confidence, method.

    ``refine_mode`` runs the learned major/minor refinement (adds a librosa chroma
    pass, ~1-2 s); set False to skip it for speed. Failures raise — there is no
    silent fallback detector (see module docstring).
    """
    validate_audio_path(path)
    return _detect_essentia(path, refine_mode)
