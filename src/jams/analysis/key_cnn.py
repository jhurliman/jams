"""In-process inference for the K10 key CNN (24-class; MIT).

This code previously lived in ``data/key_cnn_worker.py``, a uv worker subprocess that
existed solely because the old essentia-pinned core env (CPython 3.14) had no torch
wheels. With the core on 3.13 the model runs in-process: same weights, same feature
geometry, same inference math — moved, not modified.

Feature geometry must match ``eval/train_key_cnn.py`` exactly (the weights were trained
on features computed with these constants). Requests are serialized by a module lock,
preserving the worker's one-at-a-time semantics. Failures raise — no fallback by design.
"""
from __future__ import annotations

import threading
from pathlib import Path

SR = 22050
HOP = 4096
BINS_PER_OCT = 24
N_OCT = 8
PAD_SEMI = 4
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_WEIGHTS = Path(__file__).resolve().parent.parent / "data" / "models" / "key_cnn_v1.pt"
_MODEL = None  # resident across requests
_LOCK = threading.Lock()


def _build_model():
    import torch.nn as nn

    def block(cin: int, cout: int) -> list:
        return [
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ELU(),
            nn.MaxPool2d(2),
        ]

    # Head pools over TIME ONLY: key identity lives in absolute frequency position.
    return nn.Sequential(
        *block(1, 16), *block(16, 32), *block(32, 64),
        nn.Dropout2d(0.2),
        nn.AdaptiveAvgPool2d((24, 1)),
        nn.Flatten(),
        nn.Linear(64 * 24, 24),
    )


def _load():
    global _MODEL
    if _MODEL is None:
        import torch

        if not _WEIGHTS.exists():
            raise RuntimeError(
                f"Bundled key CNN weights missing at {_WEIGHTS} — broken install? "
                "(no fallback by design)"
            )
        model = _build_model()
        model.load_state_dict(torch.load(_WEIGHTS, map_location="cpu"))
        model.eval()
        _MODEL = model
    return _MODEL


def analyze(audio: str) -> dict:
    import librosa
    import numpy as np
    import torch

    with _LOCK:
        model = _load()
        y, _ = librosa.load(audio, sr=SR, mono=True)
        if y.size == 0:
            raise ValueError(f"Empty/undecodable audio: {audio}")
        n_bins = (N_OCT * 12 + 2 * PAD_SEMI) * (BINS_PER_OCT // 12)
        fmin = librosa.note_to_hz("C1") * 2 ** (-PAD_SEMI / 12)
        C = np.log1p(np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=fmin,
                                        n_bins=n_bins, bins_per_octave=BINS_PER_OCT)))
        per = BINS_PER_OCT // 12
        X = C[PAD_SEMI * per: PAD_SEMI * per + N_OCT * BINS_PER_OCT].astype(np.float32)
        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(X)[None, None]), dim=-1)[0]
    cls = int(probs.argmax())
    tonic = NOTES[cls // 2]
    mode = "minor" if cls % 2 else "major"
    return {
        "key": f"{tonic} {mode}",
        "tonic": tonic,
        "mode": mode,
        "confidence": round(float(probs.max()), 3),
        "probs": [round(float(p), 6) for p in probs],
        "method": "key-cnn-v1",
    }
