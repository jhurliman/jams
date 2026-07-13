#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.8.*",
#   "librosa>=0.10",
#   "numpy>=1.26,<2.3",
# ]
# ///
"""Tempo-estimation CNN worker for jams — our own 256-class model (TP1; MIT).

A clean-room implementation of the Schreiber & Müller (ISMIR 2018) single-step tempo
CNN family (~2.9 M params), trained on Raveform + GiantSteps-Tempo v2 (minus the 42
tracks overlapping the GiantSteps Key eval set). GiantSteps Key tempo protocol
(n=458, one pre-registered evaluation, corrected labels primary): Acc1 **0.967** —
statistically non-inferior to the previous production system (paired ΔAcc1 −0.0022,
95% CI [−0.0153, +0.0109]; see ``paper/EXPERIMENTS.md`` TP1).

Inference must reproduce the trainer exactly — ``eval/train_tempo_cnn.py`` (branch
``paper-arxiv-draft``) is the source of truth for the architecture, feature geometry,
window normalization, and sliding-window aggregation copied below. The gate was run
with this exact pipeline (librosa decode included), so nothing here may drift from it.

Kept in its own uv env because torch has no CPython 3.14 wheels. Same resident-worker
JSONL pattern as the other workers.

Modes:
  single-shot:  tempo_cnn_worker.py --audio FILE          -> prints one JSON object
  serve (JSONL): tempo_cnn_worker.py --serve
     request:  {"audio": "track.wav"}
     response: {"ok": true, "result": {"bpm": 174.0, "confidence": 0.42,
                "method": "tempo-cnn-v1"}}
               | {"ok": false, "error": "..."}
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

# Feature geometry — must match eval/train_tempo_cnn.py exactly (the weights were
# trained on features computed with these constants).
SR = 11025
N_FFT = 1024
HOP = 512
N_MELS = 40
WIN_FRAMES = 256          # ~11.9 s
BPM_MIN = 30              # 256 classes, 1-BPM bins, 30-285 BPM

_WEIGHTS = Path(__file__).resolve().parent / "models" / "tempo_cnn_v1.pt"
_MODEL = None  # resident across requests


def _build_model():
    # Copied from eval/train_tempo_cnn.py build_model() — do not modify independently.
    import torch
    import torch.nn as nn

    class MFMod(nn.Module):
        """Multi-filter module: freq avg-pool, BN, six parallel temporal convs
        (24 filters each, ELU), concat, 1x1 bottleneck to 36 (ELU)."""

        KERNELS = (32, 64, 96, 128, 192, 256)

        def __init__(self, cin, pool):
            super().__init__()
            self.pool = nn.AvgPool2d((pool, 1))
            self.bn = nn.BatchNorm2d(cin)
            self.branches = nn.ModuleList(
                nn.Conv2d(cin, 24, (1, k), padding=(0, k // 2)) for k in self.KERNELS)
            self.bottleneck = nn.Conv2d(24 * len(self.KERNELS), 36, 1)
            self.act = nn.ELU()

        def forward(self, x):
            x = self.bn(self.pool(x))
            t = x.shape[-1]  # even kernels pad to T+1: trim back to input length
            outs = [self.act(b(x))[..., :t] for b in self.branches]
            return self.act(self.bottleneck(torch.cat(outs, dim=1)))

    def short_filter(cin):
        return [nn.BatchNorm2d(cin), nn.Conv2d(cin, 16, (1, 5), padding=(0, 2)),
                nn.ELU()]

    # Freq axis: 40 -> 8 -> 4 -> 2 -> 1; time axis stays WIN_FRAMES and is
    # flattened intact into the dense back-end (36 * 256 = 9216).
    return nn.Sequential(
        *short_filter(1), *short_filter(16), *short_filter(16),
        MFMod(16, 5), MFMod(36, 2), MFMod(36, 2), MFMod(36, 2),
        nn.Flatten(),
        nn.BatchNorm1d(36 * WIN_FRAMES),
        nn.Dropout(0.5),
        nn.Linear(36 * WIN_FRAMES, 64), nn.ELU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 64), nn.ELU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 256),
    )


def _window_norm(w):
    """Reconstruct magnitude mel from log1p(power) and rescale the window to [0,1].

    Copied from eval/train_tempo_cnn.py window_norm() — do not modify independently.
    """
    import numpy as np

    w = np.sqrt(np.expm1(w))
    lo, hi = float(w.min()), float(w.max())
    return (w - lo) / (hi - lo + 1e-8)


def _load():
    global _MODEL
    if _MODEL is None:
        import torch

        if not _WEIGHTS.exists():
            raise RuntimeError(
                f"Bundled tempo CNN weights missing at {_WEIGHTS} — broken install? "
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
    X = np.log1p(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)).astype(np.float32)

    # Sliding windows, averaged softmax — mirrors predict_track() in the trainer.
    T = X.shape[1]
    wins = []
    for s in range(0, max(1, T - WIN_FRAMES + 1), WIN_FRAMES // 2):
        w = X[:, s:s + WIN_FRAMES]
        if w.shape[1] < WIN_FRAMES:
            w = np.pad(w, ((0, 0), (0, WIN_FRAMES - w.shape[1])))
        wins.append(_window_norm(w))
    batch = torch.from_numpy(np.stack(wins)[:, None])
    with torch.no_grad():
        probs = torch.softmax(model(batch), -1).mean(0)
    return {
        "bpm": float(int(probs.argmax()) + BPM_MIN),
        "confidence": round(float(probs.max()), 3),
        "method": "tempo-cnn-v1",
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
