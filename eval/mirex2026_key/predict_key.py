#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "librosa>=0.10",
#   "numpy>=1.26,<2.3",
#   "torch==2.8.*",
# ]
# ///
"""MIREX 2026 Audio Key Detection submission — JAMS key CNN (K10, width 1.0).

Usage (MIREX per-file contract):

    predict_key.py %input %output

where %input is a path to an audio file (the MIREX format is 44.1 kHz 16-bit
mono WAV; any librosa-decodable file works) and %output is the path to write
the estimated key as tab-delimited ASCII, e.g.:

    C\tmajor

One line, tonic TAB mode, trailing newline. Tonic in {C, C#, D, D#, E, F, F#,
G, G#, A, A#, B}; mode in {major, minor}.

Inference path is byte-for-byte the training featurization from
eval/train_key_cnn.py: input audio is resampled to 22050 Hz mono, a log1p
magnitude CQT is computed (24 bins/octave, hop 4096, 8 octaves + 4-semitone
augmentation pad from a base of C1), the 4-semitone pad is cropped off, and
the 192 x T patch gets a single forward pass through the CNN. The 24-way
argmax (tonic*2 + minor) is the reported key. CPU only; no randomness.

Weights: key_cnn_v1.pt in this directory (~450 KB, bundled). It is the K10
`final.pt` (trained on mirdata beatport_key, 5-fold CV model selection, final
model trained on the full corpus at the CV-selected budget).
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- feature geometry: MUST match eval/train_key_cnn.py exactly -------------
SR = 22050
HOP = 4096
BINS_PER_OCT = 24
N_OCT = 8
PAD_SEMI = 4
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

WEIGHTS = Path(__file__).resolve().parent / "key_cnn_v1.pt"


def build_model():
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


def predict(audio_path: str) -> str:
    import librosa
    import numpy as np
    import torch

    torch.set_num_threads(1)  # single-threaded per MIREX README declaration

    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty or undecodable audio: {audio_path}")

    n_bins = (N_OCT * 12 + 2 * PAD_SEMI) * (BINS_PER_OCT // 12)
    fmin = librosa.note_to_hz("C1") * 2 ** (-PAD_SEMI / 12)
    C = np.log1p(np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=fmin,
                                    n_bins=n_bins, bins_per_octave=BINS_PER_OCT)))
    per = BINS_PER_OCT // 12
    X = C[PAD_SEMI * per: PAD_SEMI * per + N_OCT * BINS_PER_OCT].astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(X)[None, None])
    cls = int(logits.argmax())
    tonic = NOTES[cls // 2]
    mode = "minor" if cls % 2 else "major"
    return f"{tonic}\t{mode}"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <input.wav> <output.txt>")
    result = predict(sys.argv[1])
    with open(sys.argv[2], "w", newline="\n") as f:
        f.write(result + "\n")


if __name__ == "__main__":
    main()
