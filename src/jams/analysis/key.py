"""Musical-key detection.

Tonic: Essentia ``KeyExtractor`` with the EDM-tuned ``edma`` profile (SOTA on
GiantSteps Key). Mode (major/minor): edma's call, refined by a small logistic
classifier trained on chroma cues — the *third* (minor 3rd vs major 3rd above the
tonic) plus the 6th/7th and a bass-register third. Template correlation dilutes the
diagnostic third among the shared scale tones; the classifier targets it directly and
only overrides edma when confident, lifting MIREX 0.759->0.801 (5-fold CV) while
keeping major-key recall. Falls back to librosa Krumhansl-Schmuckler without Essentia.
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
def _mode_model() -> dict | None:
    try:
        return json.loads(_MODE_MODEL_PATH.read_text())
    except Exception as exc:  # pragma: no cover
        logger.debug("mode model unavailable: %s", exc)
        return None


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
    if not model:
        return edma_mode
    try:
        x = _mode_features(path, tonic_idx)
    except Exception as exc:
        logger.debug("mode features failed (%s); keeping edma mode", exc)
        return edma_mode
    z = model["intercept"]
    for xi, mean, scale, coef in zip(x, model["mean"], model["scale"], model["coef"]):
        z += coef * ((xi - mean) / scale)
    p_minor = 1.0 / (1.0 + math.exp(-z))
    thr = model["threshold"]
    if p_minor >= thr:
        return "minor"
    if p_minor <= 1.0 - thr:
        return "major"
    return edma_mode


def _detect_essentia(path: str, refine_mode: bool) -> dict:
    import essentia
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    import essentia.standard as es

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


def _detect_librosa(path: str) -> dict:
    import librosa
    import numpy as np

    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major /= np.linalg.norm(major)
    minor /= np.linalg.norm(minor)

    y = load_mono(path, 22050)
    chroma = np.sum(librosa.feature.chroma_cqt(y=y, sr=22050), axis=1)
    chroma = chroma / np.linalg.norm(chroma)

    best = (-2.0, "C", "major")
    for i in range(12):
        rotated = np.roll(chroma, -i)
        for prof, mode in ((major, "major"), (minor, "minor")):
            corr = float(np.corrcoef(rotated, prof)[0, 1])
            if corr > best[0]:
                best = (corr, NOTES[i], mode)
    corr, tonic, mode = best
    return {"key": f"{tonic} {mode}", "tonic": tonic, "mode": mode,
            "confidence": round(corr, 3), "method": "librosa-krumhansl"}


def detect_key(path: str, *, refine_mode: bool = True) -> dict:
    """Detect the musical key. Returns key, tonic, mode, confidence, method.

    ``refine_mode`` runs the learned major/minor refinement (adds a librosa chroma
    pass, ~1-2 s); set False to skip it for speed.
    """
    validate_audio_path(path)
    try:
        return _detect_essentia(path, refine_mode)
    except Exception as exc:
        logger.warning("Essentia key detection unavailable (%s); using librosa fallback", exc)
        return _detect_librosa(path)
