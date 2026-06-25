"""Musical-key detection.

Primary: Essentia ``KeyExtractor`` with the EDM-tuned ``edma`` profile — the SOTA
choice for electronic/DJ material (MIREX 0.759 / exact 0.688 on GiantSteps Key).
Fallback: librosa chroma + Krumhansl-Schmuckler (0.614 / 0.529) when Essentia is
unavailable.
"""

from __future__ import annotations

import logging

from jams.analysis.audio import load_mono, validate_audio_path

logger = logging.getLogger(__name__)

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B", "Fb": "E", "E#": "F", "B#": "C",
}


def _normalize(tonic: str, scale: str) -> tuple[str, str]:
    tonic = FLAT_TO_SHARP.get(tonic, tonic)
    mode = "minor" if "min" in scale.lower() else "major"
    return tonic, mode


def _detect_essentia(path: str) -> dict:
    import essentia
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    import essentia.standard as es

    audio = load_mono(path, 44100)
    tonic, scale, strength = es.KeyExtractor(profileType="edma")(audio)
    tonic, mode = _normalize(tonic, scale)
    return {
        "key": f"{tonic} {mode}",
        "tonic": tonic,
        "mode": mode,
        "confidence": round(float(strength), 3),
        "method": "essentia-edma",
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
    return {
        "key": f"{tonic} {mode}",
        "tonic": tonic,
        "mode": mode,
        "confidence": round(corr, 3),
        "method": "librosa-krumhansl",
    }


def detect_key(path: str) -> dict:
    """Detect the musical key. Returns key, tonic, mode, confidence, method."""
    validate_audio_path(path)
    try:
        return _detect_essentia(path)
    except Exception as exc:
        logger.warning("Essentia key detection unavailable (%s); using librosa fallback", exc)
        return _detect_librosa(path)
