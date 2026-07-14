"""Musical-key detection.

The pipeline is our own 24-class key CNN (K10; MIT, ~0.1 M params, weights bundled at
``data/models/key_cnn_v1.pt``, run in-process — ``jams.analysis.key_cnn``). Trained
on the public Beatport corpus underlying GiantSteps-MTG-Keys with pitch-shift
augmentation; GiantSteps Key honest protocol, one pre-registered evaluation: weighted
MIREX **0.832** / exact **0.780** — statistically indistinguishable from the strongest
published system (madmom's CNN, whose weights are CC BY-NC-SA) with no non-commercial
restriction. Method string: ``key-cnn-v1``.

There is deliberately no fallback detector and no silent degradation: any failure raises
(accuracy must not depend on installation accidents).

The retired edma + S-KEY fusion system (weighted MIREX 0.812 / exact 0.757) no longer
runs here; its trained heads remain at ``data/key_fusion.json`` / ``data/mode_model.json``
and the pure helpers below so ``eval/stats_significance.py`` can replay the archived
baseline from banked features.
"""

from __future__ import annotations

import logging
import math

from jams.analysis.audio import validate_audio_path

logger = logging.getLogger(__name__)

# NOTE: NOTES / FLAT_TO_SHARP / _normalize have no production callers left — the CNN
# path (key_cnn.py) keeps its own note table. They are retained for the eval-replay
# helpers below; eval/stats_significance.py imports NOTES directly.
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B", "Fb": "E", "E#": "F", "B#": "C",
}


def _normalize(tonic: str, scale: str) -> tuple[str, str]:
    tonic = FLAT_TO_SHARP.get(tonic, tonic)
    mode = "minor" if "min" in scale.lower() else "major"
    return tonic, mode


def _detect_cnn(path: str) -> dict:
    """Run the K10 key CNN in-process; failures raise (no fallback by design)."""
    from jams.analysis import key_cnn

    res = key_cnn.analyze(path)
    return {
        "key": res["key"],
        "tonic": res["tonic"],
        "mode": res["mode"],
        "confidence": round(float(res["confidence"]), 3),
        "method": res["method"],
    }


def detect_key(path: str) -> dict:
    """Detect the musical key. Returns key, tonic, mode, confidence, method.

    Runs the bundled K10 CNN in-process. Failures raise — there is no silent
    fallback detector (see module docstring).
    """
    validate_audio_path(path)
    return _detect_cnn(path)


# --- EVAL-REPLAY ONLY — retired edma + S-KEY fusion helpers -------------------------
# Nothing below runs on the production path (that is detect_key -> key_cnn above).
# These pure functions exist so eval/stats_significance.py can replay the retired
# fusion heads (data/key_fusion.json) from banked per-track features and regenerate
# the paper's K4/K6 baseline CIs (paper/EXPERIMENTS.md ledger; the paper claims every
# CI regenerates from committed code). Do not delete while the paper reports those
# rows. Guarded by tests/test_key_replay_helpers.py; see also src/jams/data/README.md.

# S-KEY's 24-class posterior ordering (majors 0-11, minors 12-23) — must match the
# key_map in deezer/skey and the ordering baked into key_fusion.json at export time.
_SKEY_ORDER = (
    [(n, "major") for n in ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#")]
    + [(n, "minor") for n in ("B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#")]
)
_SKEY_IDX = {k: i for i, k in enumerate(_SKEY_ORDER)}


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
