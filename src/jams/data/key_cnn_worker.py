#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.8.*",
#   "librosa>=0.10",
#   "numpy>=1.26,<2.3",
# ]
# ///
"""Key-detection CNN worker for jams — our own 24-class model (K10; MIT).

A small convolutional network (~0.1 M params) over a log-CQT, trained on the public
Beatport corpus underlying GiantSteps-MTG-Keys with ±4-semitone pitch-shift
augmentation (see ``eval/train_key_cnn.py`` and ``paper/EXPERIMENTS.md`` K10).
GiantSteps Key (n=567, honest protocol, one pre-registered evaluation): weighted MIREX
**0.832** / exact **0.780** — statistically indistinguishable from the strongest
published system (madmom's CNN, CC BY-NC-SA) with no non-commercial restriction.

Kept in its own uv env because torch has no CPython 3.14 wheels (jams is pinned to
3.14 for essentia-tensorflow). Same resident-worker JSONL pattern as the other workers.
CPU inference is well under a second per track after the CQT.

Modes:
  single-shot:  key_cnn_worker.py --audio FILE          -> prints one JSON object
  serve (JSONL): key_cnn_worker.py --serve
     request:  {"audio": "track.wav"}
     response: {"ok": true, "result": {"key": "D minor", "tonic": "D", "mode": "minor",
                "confidence": 0.87, "probs": [24 floats], "method": "key-cnn-v1"}}
               | {"ok": false, "error": "..."}
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

# Feature geometry — must match eval/train_key_cnn.py exactly (the weights were
# trained on features computed with these constants).
SR = 22050
HOP = 4096
BINS_PER_OCT = 24
N_OCT = 8
PAD_SEMI = 4
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_WEIGHTS = Path(__file__).resolve().parent / "models" / "key_cnn_v1.pt"
_MODEL = None  # resident across requests


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


def _serve() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            # Keep protocol stdout clean: model/lib prints go to stderr.
            with contextlib.redirect_stdout(sys.stderr):
                res = analyze(req["audio"])
            out = {"ok": True, "result": res}
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true", help="JSONL stdin/stdout server mode")
    ap.add_argument("--audio", help="Audio file to analyze (single-shot)")
    args = ap.parse_args()
    if args.serve:
        _serve()
        return
    if not args.audio:
        ap.error("provide --audio FILE or --serve")
    with contextlib.redirect_stdout(sys.stderr):  # model-load prints must not precede JSON
        res = analyze(args.audio)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
