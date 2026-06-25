#!/usr/bin/env -S uv run --extra eval
"""Benchmark candidate key/tempo methods against GiantSteps Key ground truth.

Runs several algorithms over the manifest's audio and reports, per method:
  - key  : MIREX weighted score + exact accuracy
  - tempo: Acc1 (within tol) + Acc2 (octave-tolerant)

so we can pick the best for the production service. Audio is loaded once per
track (44.1 kHz mono) and shared across methods.

    uv run eval/benchmark_methods.py --limit 80     # fast ranking subset
    uv run eval/benchmark_methods.py                # full dataset
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH = {n: i for i, n in enumerate(NOTES)}
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
                 "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}


def norm_key(tonic: str, scale: str) -> str | None:
    tonic = FLAT_TO_SHARP.get(tonic, tonic)
    if tonic not in PITCH:
        return None
    scale = "minor" if "min" in scale.lower() else "major"
    return f"{tonic} {scale}"


def parse_key(key: str):
    if not key:
        return None
    p = key.split()
    if len(p) != 2 or p[0] not in PITCH:
        return None
    return PITCH[p[0]], p[1]


def mirex_score(ref: str, pred: str) -> float:
    r, p = parse_key(ref), parse_key(pred)
    if r is None or p is None:
        return 0.0
    rp, rm = r
    pp, pm = p
    iv = (pp - rp) % 12
    if iv == 0 and pm == rm:
        return 1.0
    if pm == rm and iv in (7, 5):
        return 0.5
    if rm == "major" and pm == "minor" and iv == 9:
        return 0.3
    if rm == "minor" and pm == "major" and iv == 3:
        return 0.3
    if iv == 0 and pm != rm:
        return 0.2
    return 0.0


def tempo_acc(ref: float, pred: float, tol: float = 0.04):
    if not ref or not pred:
        return None
    a1 = abs(pred - ref) <= tol * ref
    a2 = a1 or any(abs(pred * f - ref) <= tol * ref for f in (0.5, 2.0, 1 / 3, 3.0))
    return a1, a2


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--manifest", type=Path, default=here / "data" / "manifest.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import essentia
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    import essentia.standard as es
    import numpy as np
    from tqdm import tqdm

    recs = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("audio_exists")]
    if args.limit:
        recs = recs[: args.limit]

    # --- key methods: name -> profileType for essentia KeyExtractor ---
    key_profiles = ["edma", "krumhansl", "temperley", "bgate", "shaath"]
    key_extractors = {p: es.KeyExtractor(profileType=p) for p in key_profiles}

    # --- tempo methods ---
    rhythm2013 = es.RhythmExtractor2013(method="multifeature")
    percival = es.PercivalBpmEstimator()

    key_w = defaultdict(list)
    key_exact = defaultdict(list)
    tempo_hits = defaultdict(lambda: [0, 0, 0])  # method -> [n, acc1, acc2]
    per_track = []

    for r in tqdm(recs, desc="bench", unit="trk"):
        path = r["audio_path"]
        try:
            audio = es.MonoLoader(filename=path, sampleRate=44100)()
        except Exception as e:
            per_track.append({"track_id": r["track_id"], "error": f"load: {e}"})
            continue
        ref_key = r["ref_key"]
        ref_tempo = r.get("ref_tempo")
        row = {"track_id": r["track_id"], "ref_key": ref_key, "ref_tempo": ref_tempo}

        for name, ke in key_extractors.items():
            try:
                tonic, scale, _ = ke(audio)
                pred = norm_key(tonic, scale)
                w = mirex_score(ref_key, pred) if pred else 0.0
            except Exception:
                pred, w = None, 0.0
            key_w[name].append(w)
            key_exact[name].append(1.0 if w == 1.0 else 0.0)
            row[f"key.{name}"] = pred

        # tempo
        try:
            bpm13 = float(rhythm2013(audio)[0])
        except Exception:
            bpm13 = 0.0
        try:
            bpmp = float(percival(audio))
        except Exception:
            bpmp = 0.0
        row["tempo.rhythm2013"] = round(bpm13, 2)
        row["tempo.percival"] = round(bpmp, 2)
        for name, bpm in (("rhythm2013", bpm13), ("percival", bpmp)):
            acc = tempo_acc(ref_tempo, bpm)
            if acc is not None:
                h = tempo_hits[name]
                h[0] += 1
                h[1] += acc[0]
                h[2] += acc[1]
        per_track.append(row)

    n = len(recs)
    print(f"\n==== KEY  (n={n}, MIREX weighted / exact) ====")
    ranked = sorted(key_profiles, key=lambda p: -sum(key_w[p]) / max(1, len(key_w[p])))
    for p in ranked:
        m = sum(key_w[p]) / len(key_w[p])
        e = sum(key_exact[p]) / len(key_exact[p])
        print(f"  essentia:{p:10}  MIREX {m:.4f}   exact {e:.4f}")
    print("  [baseline librosa K-S]  MIREX 0.6138   exact 0.5291")

    print(f"\n==== TEMPO  (Acc1 ±4% / Acc2 octave) ====")
    for name in ("rhythm2013", "percival"):
        h = tempo_hits[name]
        if h[0]:
            print(f"  essentia:{name:12}  n={h[0]}  Acc1 {h[1]/h[0]:.4f}   Acc2 {h[2]/h[0]:.4f}")
    print("  [baseline librosa]      Acc1 0.8297   Acc2 0.8690")

    if args.out:
        args.out.write_text(json.dumps(per_track, indent=2))
        print(f"\nper-track -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
