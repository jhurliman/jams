#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.8.*",
#   "librosa>=0.10",
#   "numpy>=1.26,<2.3",
# ]
# ///
"""Drum-transcription worker for jams — our own 5-class drum CRNN (D1; MIT).

Transcribes a drum stem to General MIDI percussion notes with our own CRNN
(~1.5 M params, bundled at ``models/drum_cnn_v1.pt``), trained on E-GMD +
Slakh2100-redux train splits (oracle and separator-processed drum stems). It
replaced the previous ADTOF-pytorch port after passing the pre-registered D1
one-shot gate with superiority on both arms (paper/EXPERIMENTS.md §D1):
Slakh-test oracle macro onset-F **0.767** vs ADTOF 0.638, E-GMD test **0.818**
vs 0.645. Unlike ADTOF it also predicts per-hit velocity (a real dynamics
head, not post-hoc RMS), and the weights are MIT like the rest of jams — no
subprocess-isolated licensing required anymore.

Inference must reproduce the trainer exactly — ``eval/train_drum_cnn.py``
(in-repo; blob 2d7f9e9d, the exact code that trained the shipped checkpoint)
is the source of truth for the architecture, feature geometry, chunked
sliding inference, and peak-picking copied below. The gate was run with this
exact pipeline (librosa decode included), so nothing here may drift from it.

The model emits 5 drum classes at the canonical GM representatives used across
jams (``jams.analysis.gm``) — 36 kick, 38 snare, 42 hi-hat, 47 tom, 49 cymbal —
with velocity from the model's velocity head (0-127, floored at 1 for the MIDI
note-on contract). Kept in its own uv env (torch has no CPython 3.14 wheels)
so the pipeline pieces stay independently replaceable; same resident-worker
JSONL pattern as the other workers.

Modes:
  single-shot:  drum_worker.py --drums-wav FILE           -> prints one JSON object
  serve (JSONL): drum_worker.py --serve
     request:  {"drums_wav": "drums.wav"}
     response: {"ok": true, "result": {"notes": [{"onset","offset","pitch","velocity"}, ...]}}
               | {"ok": false, "error": "..."}
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

# Feature geometry — must match eval/train_drum_cnn.py exactly (the weights were
# trained on features computed with these constants).
SR = 22050
N_FFT = 1024
HOP = 220                 # ~9.98 ms -> ~100.2 frames/s ("10 ms hop")
N_MELS = 96
FPS = SR / HOP

# 5-class GM vocabulary (class idx 0..4) — the representatives jams.analysis.gm
# reduces onto (reduce_drum_pitch_5): kick, snare, hi-hat, tom, cymbal.
CLASSES = [36, 38, 42, 47, 49]

# Per-class onset thresholds, grid-searched on validation at train time
# (val_report.json next to the D1 checkpoint; class order as CLASSES above).
THRESHOLDS = np.array([0.9, 0.85, 0.7, 0.9, 0.75])

# Drums have onsets, not offsets: fixed short notes (DAW-friendly one-shot hits).
NOTE_LEN = 0.05

_WEIGHTS = Path(__file__).resolve().parent / "models" / "drum_cnn_v1.pt"
_MODEL = None  # resident across requests
_DEVICE = "cpu"


def _build_model():
    # Copied from eval/train_drum_cnn.py build_model() — do not modify independently.
    # CRNN: 3 conv blocks (32/64/96, two 3x3 convs each + BN + ReLU, freq-pool 2)
    # -> freq-flatten -> 2-layer BiGRU 128 -> 5-class onset logits + velocity head.
    import torch.nn as nn

    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d((2, 1))]

    class CRNN(nn.Module):
        def __init__(self, n_mels=N_MELS, hidden=128):
            super().__init__()
            self.conv = nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 96))
            feat = 96 * (n_mels // 8)                      # 96 mels -> 12 bands
            self.rnn = nn.GRU(feat, hidden, num_layers=2, batch_first=True,
                              bidirectional=True)
            self.onset = nn.Linear(2 * hidden, len(CLASSES))
            self.velocity = nn.Linear(2 * hidden, len(CLASSES))

        def forward(self, x):            # x: (B, 1, mels, T)
            h = self.conv(x)             # (B, 96, mels/8, T)
            h = h.permute(0, 3, 1, 2).flatten(2)   # (B, T, 96*mels/8)
            h, _ = self.rnn(h)
            return self.onset(h), self.velocity(h)  # (B, T, 5) logits, (B, T, 5)

    return CRNN()


def _local_max_mask(p: np.ndarray) -> np.ndarray:
    """Vectorized: True where p[f] is the max of its ±2-frame window."""
    # Copied from eval/train_drum_cnn.py — do not modify independently.
    m = p.copy()
    for shift in (-2, -1, 1, 2):
        s = np.roll(p, shift)
        if shift > 0:
            s[:shift] = -1.0
        else:
            s[shift:] = -1.0
        m = np.maximum(m, s)
    return p >= m - 1e-9


def _pick_events(prob: np.ndarray, vel: np.ndarray, thresholds: np.ndarray
                 ) -> list[list[float]]:
    """Per-class peak-picking: threshold + local max over ±2 frames + 50 ms min gap.
    Returns [time_s, class_idx, velocity 0..1] sorted by time."""
    # Copied from eval/train_drum_cnn.py — do not modify independently.
    events: list[list[float]] = []
    min_gap = int(round(0.05 * FPS))
    for c in range(prob.shape[1]):
        p = prob[:, c]
        peaks = np.where(_local_max_mask(p) & (p >= thresholds[c]))[0]
        last = -10_000
        for f in peaks:
            if f - last < min_gap:
                continue
            last = f
            events.append([f / FPS, c, float(np.clip(vel[f, c], 0.0, 1.0))])
    events.sort(key=lambda e: e[0])
    return events


def _predict_probs(model, X: np.ndarray, dev, chunk: int = 8000, overlap: int = 200):
    """Full-track frame probabilities + velocities, chunked with center-stitching."""
    # Copied from eval/train_drum_cnn.py — do not modify independently.
    import torch

    T = X.shape[1]
    on = np.zeros((T, len(CLASSES)), dtype=np.float32)
    vl = np.zeros((T, len(CLASSES)), dtype=np.float32)
    s = 0
    with torch.no_grad():
        while s < T:
            e = min(T, s + chunk)
            xw = torch.from_numpy(X[:, s:e].astype(np.float32))[None, None].to(dev)
            lo_t, ve_t = model(xw)
            po = torch.sigmoid(lo_t)[0].cpu().numpy()
            pv = ve_t[0].cpu().numpy()
            a = s + (overlap if s > 0 else 0)
            on[a:e] = po[a - s:e - s]
            vl[a:e] = pv[a - s:e - s]
            if e == T:
                break
            s = e - 2 * overlap
    return on, vl


def _load():
    global _MODEL, _DEVICE
    if _MODEL is None:
        import torch

        if not _WEIGHTS.exists():
            raise RuntimeError(
                f"Bundled drum CNN weights missing at {_WEIGHTS} — broken install? "
                "(no fallback by design)"
            )
        _DEVICE = ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
        model = _build_model()
        model.load_state_dict(torch.load(_WEIGHTS, map_location="cpu", weights_only=True))
        model.to(_DEVICE).eval()
        _MODEL = model
    return _MODEL, _DEVICE


def transcribe_drums(wav: str) -> list[dict]:
    """Transcribe a drum stem to GM percussion notes via the D1 drum CRNN."""
    import librosa

    model, dev = _load()
    y, _ = librosa.load(wav, sr=SR, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty/undecodable audio: {wav}")
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS)
    X = np.log1p(m).astype(np.float32)
    prob, vel = _predict_probs(model, X, dev)
    # Same event mapping as the D1 gate inference (predict_drums.py): time rounded
    # to 0.1 ms, class -> GM representative, velocity head -> 0..127. The only
    # note-level additions are the fixed NOTE_LEN offset and the velocity floor
    # of 1 (MIDI velocity 0 means note-off; the API contract is 1..127).
    notes = []
    for t, c, v in _pick_events(prob, vel, THRESHOLDS):
        onset = round(t, 4)
        notes.append({
            "onset": onset,
            "offset": round(onset + NOTE_LEN, 4),
            "pitch": CLASSES[int(c)],
            "velocity": max(1, int(round(v * 127))),
        })
    return notes


def _serve() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            # Keep protocol stdout clean: model/lib prints go to stderr.
            with contextlib.redirect_stdout(sys.stderr):
                notes = transcribe_drums(req["drums_wav"])
            out = {"ok": True, "result": {"notes": notes}}
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--drums-wav")
    args = ap.parse_args()
    if args.serve:
        _serve()
        return
    if not args.drums_wav:
        ap.error("provide --drums-wav FILE or --serve")
    with contextlib.redirect_stdout(sys.stderr):  # model-load prints must not precede the JSON
        notes = transcribe_drums(args.drums_wav)
    print(json.dumps({"notes": notes}, indent=2))


if __name__ == "__main__":
    main()
