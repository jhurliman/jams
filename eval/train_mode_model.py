#!/usr/bin/env -S uv run --extra eval
"""Train the major/minor refinement model shipped at src/jams/data/mode_model.json.

edma gets the tonic right ~90% of the time but over-calls *major* on minor EDM tracks
(template correlation dilutes the diagnostic third). This trains a small logistic
classifier on chroma cues anchored at edma's tonic — the minor-3rd vs major-3rd,
the 6th/7th, and a bass-register third — and uses it to override edma's mode only when
confident. 5-fold cross-validated, so the reported number is an honest generalization
estimate (the shipped model is then refit on all data).

    uv run --extra eval eval/train_mode_model.py            # uses cached chroma if present
    uv run --extra eval eval/train_mode_model.py --reextract

Reproduces: MIREX 0.759 -> ~0.801, exact 0.688 -> ~0.743 (threshold 0.90).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH = {n: i for i, n in enumerate(NOTES)}
HERE = Path(__file__).resolve().parent


def parse(k):
    p = (k or "").split()
    return (PITCH[p[0]], p[1]) if len(p) == 2 and p[0] in PITCH else None


def mirex(refkey, tonic_idx, mode):
    r = parse(refkey)
    if r is None:
        return 0.0
    iv = (tonic_idx - r[0]) % 12
    if iv == 0 and mode == r[1]:
        return 1.0
    if iv in (7, 5) and mode == r[1]:
        return 0.5
    if r[1] == "major" and mode == "minor" and iv == 9:
        return 0.3
    if r[1] == "minor" and mode == "major" and iv == 3:
        return 0.3
    if iv == 0 and mode != r[1]:
        return 0.2
    return 0.0


def _features_at(c, b, t):
    def g(a, iv):
        return float(a[(t + iv) % 12])
    return [g(c, 3) - g(c, 4), g(b, 3) - g(b, 4), g(c, 8) - g(c, 9), g(c, 10) - g(c, 11),
            g(c, 0), g(c, 7), g(c, 3), g(c, 4)]


def build_cache(reextract: bool) -> list[dict]:
    cache_path = HERE / "data" / "chroma_cache.json"
    if cache_path.is_file() and not reextract:
        return json.loads(cache_path.read_text())
    import librosa
    from tqdm import tqdm

    res = json.load(open(HERE / "data" / "results_sota.json"))
    man = {json.loads(line)["track_id"]: json.loads(line)
           for line in open(HERE / "data" / "manifest.jsonl") if line.strip()}
    out = []
    for r in tqdm(res, desc="chroma"):
        pk = parse(r.get("pred_key"))
        m = man.get(r["track_id"], {})
        ap = m.get("audio_path")
        if pk is None or not ap or not Path(ap).is_file():
            continue
        y, _ = librosa.load(ap, sr=22050, mono=True)
        yh = librosa.effects.harmonic(y)
        c = np.sum(librosa.feature.chroma_cqt(y=yh, sr=22050), axis=1)
        c = (c / (np.linalg.norm(c) + 1e-9)).tolist()
        b = np.sum(librosa.feature.chroma_cqt(y=yh, sr=22050, fmin=librosa.note_to_hz("C2"), n_octaves=3), axis=1)
        b = (b / (np.linalg.norm(b) + 1e-9)).tolist()
        out.append({"track_id": r["track_id"], "ref_key": r.get("ref_key"),
                    "edma_tonic": pk[0], "edma_mode": pk[1], "chroma": c, "chroma_bass": b})
    cache_path.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reextract", action="store_true", help="Recompute chroma from audio")
    ap.add_argument("--threshold", type=float, default=0.90, help="Override-confidence threshold")
    ap.add_argument("--out", type=Path, default=HERE.parent / "src" / "jams" / "data" / "mode_model.json")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    D = build_cache(args.reextract)
    X = np.array([_features_at(np.array(d["chroma"]), np.array(d["chroma_bass"]), d["edma_tonic"]) for d in D])
    y = np.array([1 if parse(d["ref_key"]) and parse(d["ref_key"])[1] == "minor" else 0 for d in D])
    refs = [d["ref_key"] for d in D]
    tonics = [d["edma_tonic"] for d in D]
    edma = [d["edma_mode"] for d in D]

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    thr = args.threshold
    modes = ["minor" if proba[i] >= thr else ("major" if proba[i] <= 1 - thr else edma[i]) for i in range(len(D))]
    w = [mirex(refs[i], tonics[i], modes[i]) for i in range(len(D))]
    base = [mirex(refs[i], tonics[i], edma[i]) for i in range(len(D))]
    cv_mirex = sum(w) / len(w)
    cv_exact = sum(1.0 for x in w if x == 1.0) / len(w)
    print(f"edma baseline : MIREX {sum(base)/len(base):.4f}  exact {sum(1.0 for x in base if x==1.0)/len(base):.4f}")
    print(f"+ mode clf (CV): MIREX {cv_mirex:.4f}  exact {cv_exact:.4f}  (threshold {thr})")

    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    sc = final.named_steps["standardscaler"]
    lr = final.named_steps["logisticregression"]
    params = {
        "_doc": "Mode refinement for detect_key. Features at edma's tonic from librosa "
                "harmonic chroma (treble+bass). minor if sigmoid(coef.z+b)>=threshold, "
                "major if <=1-threshold, else keep edma.",
        "feature_names": ["m3-M3", "bass_m3-M3", "m6-M6", "m7-M7", "tonic", "fifth", "m3", "M3"],
        "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "threshold": thr, "cv_mirex": round(cv_mirex, 4), "cv_exact": round(cv_exact, 4),
        "trained_on": f"GiantSteps Key ({len(D)})",
    }
    args.out.write_text(json.dumps(params, indent=2))
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
