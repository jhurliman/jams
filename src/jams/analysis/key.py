"""Musical-key detection.

Default pipeline (**cnn**): our own 24-class key CNN (K10; MIT, ~0.1 M params, weights
bundled at ``data/models/key_cnn_v1.pt``, run in a uv worker —
``data/key_cnn_worker.py``). Trained on the public Beatport corpus underlying
GiantSteps-MTG-Keys with pitch-shift augmentation; GiantSteps Key honest protocol, one
pre-registered evaluation: weighted MIREX **0.832** / exact **0.780** — statistically
indistinguishable from the strongest published system (madmom's CNN, whose weights are
CC BY-NC-SA) with no non-commercial restriction. Method string: ``key-cnn-v1``.
Select with ``JAMS_KEY_BACKEND`` ("cnn" | "fusion").

The previous **fusion** pipeline remains available: Essentia ``KeyExtractor`` (EDM-tuned
``edma`` profile) provides the tonic and a first mode estimate; Deezer's **S-KEY**
(self-supervised, MIT, run in a uv worker — ``data/skey_worker.py``) provides an
independent 24-class key posterior. Two small logistic heads fuse them:

  1. *mode head* — refines major/minor from chroma cues (the diagnostic third, 6th/7th,
     bass-register third) + edma confidence + S-KEY posterior features anchored at edma's
     tonic. Overrides edma's mode only when confident.
  2. *rerank head* — decides per-track whether to keep the refined edma key or switch to
     S-KEY's key outright (their errors decorrelate: edma is exact-hit-strong, S-KEY is
     near-miss-strong).

Both heads were trained **only on GiantSteps-MTG-Keys** (the training split) and evaluated
once on GiantSteps Key: weighted MIREX **0.812** / exact **0.757** — above the honest
published SOTA (~0.76 weighted) and above every single component. The method string for
this pipeline is ``essentia-edma+skey-fusion`` (the whole decision is fusion-informed,
whichever branch wins). An earlier mode model was trained on the test set itself; it
remains only behind ``JAMS_KEY_FUSION=0`` as the legacy path (``essentia-edma+modeclf``)
and its numbers must not be quoted against the literature.

essentia-tensorflow is a **hard requirement** (wheels for macOS arm64 + Linux x86_64,
CPython 3.14), and with fusion enabled the S-KEY worker is too. There is deliberately no
fallback detector and no silent degradation: any failure raises (accuracy must not depend
on installation accidents).
"""

from __future__ import annotations

import json
import logging
import math
import threading
from functools import lru_cache
from pathlib import Path

from jams.analysis.audio import load_mono, validate_audio_path
from jams.config import get_settings

logger = logging.getLogger(__name__)

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B", "Fb": "E", "E#": "F", "B#": "C",
}
_MODE_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "mode_model.json"
_KEY_FUSION_PATH = Path(__file__).resolve().parent.parent / "data" / "key_fusion.json"
_SKEY_WORKER_PATH = Path(__file__).resolve().parent.parent / "data" / "skey_worker.py"
_KEY_CNN_WORKER_PATH = Path(__file__).resolve().parent.parent / "data" / "key_cnn_worker.py"

# S-KEY's 24-class posterior ordering (majors 0-11, minors 12-23) — must match the
# key_map in deezer/skey and the ordering baked into key_fusion.json at export time.
_SKEY_ORDER = (
    [(n, "major") for n in ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#")]
    + [(n, "minor") for n in ("B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#")]
)
_SKEY_IDX = {k: i for i, k in enumerate(_SKEY_ORDER)}


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


@lru_cache(maxsize=1)
def _key_fusion_model() -> dict:
    # Bundled with the package; absence is a broken install, not a soft condition.
    try:
        return json.loads(_KEY_FUSION_PATH.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"Bundled key-fusion model missing/corrupt at {_KEY_FUSION_PATH} — broken "
            "install? Set JAMS_KEY_FUSION=0 only to intentionally use the legacy path."
        ) from exc


_skey_singleton = None
_key_cnn_singleton = None
_skey_singleton_lock = threading.Lock()


def _skey_worker():
    """Resident S-KEY uv worker (same subprocess pattern as the stems workers)."""
    global _skey_singleton
    if _skey_singleton is None:
        with _skey_singleton_lock:
            if _skey_singleton is None:
                from jams.analysis.stems import _Worker

                _skey_singleton = _Worker(_SKEY_WORKER_PATH, "skey")
    return _skey_singleton


def _key_cnn_worker():
    """Resident key-CNN uv worker (same subprocess pattern as the stems workers)."""
    global _key_cnn_singleton
    if _key_cnn_singleton is None:
        with _skey_singleton_lock:
            if _key_cnn_singleton is None:
                from jams.analysis.stems import _Worker

                _key_cnn_singleton = _Worker(
                    _KEY_CNN_WORKER_PATH, "key-cnn", uv_setting="key_cnn_uv"
                )
    return _key_cnn_singleton


def _detect_cnn(path: str) -> dict:
    """Run the K10 key CNN worker; failures raise (no fallback by design)."""
    res = _key_cnn_worker().analyze({"audio": path})
    return {
        "key": res["key"],
        "tonic": res["tonic"],
        "mode": res["mode"],
        "confidence": round(float(res["confidence"]), 3),
        "method": res["method"],
    }


def _parse_skey_key(key: str) -> tuple[str, str]:
    """Normalize an S-KEY key string ("Bb minor", "D Major") to (tonic, mode)."""
    parts = key.split()
    if len(parts) != 2:
        raise ValueError(f"Unparseable S-KEY key: {key!r}")
    return _normalize(parts[0], parts[1])


def _skey_feats(posterior: list[float], tonic_idx: int, edma_mode: str) -> list[float]:
    """S-KEY posterior features anchored at edma's tonic.

    Order is load-bearing — it must match the training extraction in the eval scripts:
    [P(tonic,minor), P(tonic,major), diff, P(relative-minor), P(relative-major),
     P(fifth-up, edma mode), P(fourth-up, edma mode), max, entropy].
    """
    def p(t: int, m: str) -> float:
        return float(posterior[_SKEY_IDX[(NOTES[t % 12], m)]])

    ent = -sum(x * math.log(x + 1e-12) for x in posterior)
    return [
        p(tonic_idx, "minor"), p(tonic_idx, "major"),
        p(tonic_idx, "minor") - p(tonic_idx, "major"),
        p((tonic_idx + 9) % 12, "minor"), p((tonic_idx + 3) % 12, "major"),
        p(tonic_idx + 7, edma_mode), p(tonic_idx + 5, edma_mode),
        float(max(posterior)), float(ent),
    ]


def _logistic(model: dict, x: list[float]) -> float:
    z = model["intercept"]
    for xi, mean, scale, coef in zip(x, model["mean"], model["scale"], model["coef"],
                                     strict=True):
        z += coef * ((xi - mean) / scale)
    return 1.0 / (1.0 + math.exp(-z))


def _fuse(path: str, tonic: str, edma_mode: str, conf: float) -> tuple[str, str]:
    """Run the fusion heads; return (final tonic, final mode)."""
    fusion = _key_fusion_model()
    tonic_idx = NOTES.index(tonic)
    cues = _mode_features(path, tonic_idx)
    skey = _skey_worker().analyze({"audio": path})
    sfeat = _skey_feats(skey["posterior"], tonic_idx, edma_mode)

    # Head 1: mode refinement (keeps edma's tonic).
    p_minor = _logistic(fusion["mode"], cues + [conf] + sfeat)
    thr = fusion["mode"]["threshold"]
    mode = edma_mode
    if p_minor >= thr:
        mode = "minor"
    elif p_minor <= 1.0 - thr:
        mode = "major"

    # Head 2: keep the refined edma key, or switch to S-KEY's key outright.
    sk_tonic, sk_mode = _parse_skey_key(skey["skey_key"])
    agree_full = 1.0 if (sk_tonic, sk_mode) == (tonic, mode) else 0.0
    agree_tonic = 1.0 if sk_tonic == tonic else 0.0
    x2 = cues + [conf, p_minor, abs(p_minor - 0.5)] + sfeat + [agree_full, agree_tonic]
    p_switch = _logistic(fusion["rerank"], x2)
    if p_switch >= fusion["rerank"]["threshold"]:
        return sk_tonic, sk_mode
    return tonic, mode


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
        if get_settings().key_fusion:
            tonic, mode = _fuse(path, tonic, mode, float(strength))
            method = "essentia-edma+skey-fusion"
        else:
            # Legacy path (JAMS_KEY_FUSION=0): mode_model.json was trained on GiantSteps
            # Key itself — usable, but its accuracy must not be quoted vs the literature.
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

    Backend is selected by ``JAMS_KEY_BACKEND``: "cnn" (default) runs the bundled K10
    CNN in its uv worker (``refine_mode`` has no effect — there is no refinement
    stage); "fusion" runs the edma + S-KEY pipeline, where ``refine_mode=False`` skips
    the learned refinement for speed (plain edma). Failures raise — there is no silent
    fallback detector or silent backend downgrade (see module docstring).
    """
    validate_audio_path(path)
    if get_settings().key_backend == "cnn":
        return _detect_cnn(path)
    return _detect_essentia(path, refine_mode)
