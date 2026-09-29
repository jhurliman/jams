# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mir_eval==0.8.2"]
# ///
"""Recompute the GiantSteps Key table of the MIREX 2026 extended abstract (and the paper's
Appendix B key numbers) from the committed per-track predictions.

Input: paper/evidence/key/gskey_predictions.jsonl — one row per GiantSteps Key excerpt
with the reference label (from the public GiantSteps annotations), the K10 prediction
(single-shot evaluation of the locked model, 2026-07-12), and madmom's default
CNNKeyRecognitionProcessor prediction on the same audio. Usable excerpts are those whose
reference parses as a major/minor key (567 of 604; 600 had retrievable audio).

Scores: the project's legacy weighted score (fifth credit in both directions, as in the
2018 madmom paper) and mir_eval 0.8.2's weighted score (ascending fifth only); exact
accuracy; paired K10 - madmom differences with 10,000-resample track bootstrap, seed 0.

Usage: uv run paper/key_scores.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mir_eval
import numpy as np

HERE = Path(__file__).resolve().parent
PRED = HERE / "evidence" / "key" / "gskey_predictions.jsonl"
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}


def parse(k):
    p = (k or "").replace("Major", "major").replace("Minor", "minor").split()
    if len(p) != 2:
        return None
    t = FLAT.get(p[0], p[0])
    return (NOTES.index(t), p[1]) if t in NOTES and p[1] in ("major", "minor") else None


def norm(k):
    p = parse(k)
    return f"{NOTES[p[0]]} {p[1]}" if p else None


def legacy(ref, est):
    """Symmetric-fifth weighted score (identical to eval/stats_significance.mirex)."""
    r, e = parse(ref), parse(est)
    if r is None or e is None:
        return 0.0
    iv = (e[0] - r[0]) % 12
    if iv == 0 and e[1] == r[1]:
        return 1.0
    if e[1] == r[1] and iv in (7, 5):
        return 0.5
    if r[1] == "major" and e[1] == "minor" and iv == 9:
        return 0.3
    if r[1] == "minor" and e[1] == "major" and iv == 3:
        return 0.3
    if iv == 0:
        return 0.2
    return 0.0


def main() -> int:
    rows = [json.loads(line) for line in PRED.read_text().splitlines() if line.strip()]
    ids = [r["track_id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit("duplicate track ids in predictions file")
    usable = [r for r in rows if parse(r["ref_key"]) and r["k10_key"] and r["madmom_key"]]
    if len(usable) != 567:
        sys.exit(f"expected 567 usable excerpts, found {len(usable)}")
    out = {"n": len(usable)}
    per = {}
    for name in ("k10", "madmom"):
        sym = np.array([legacy(r["ref_key"], r[f"{name}_key"]) for r in usable])
        asc = np.array(
            [
                mir_eval.key.weighted_score(norm(r["ref_key"]), norm(r[f"{name}_key"]))
                for r in usable
            ]
        )
        exact = np.array(
            [1.0 if norm(r["ref_key"]) == norm(r[f"{name}_key"]) else 0.0 for r in usable]
        )
        per[name] = {"sym": sym, "asc": asc}
        rng = np.random.default_rng(0)
        bs = [sym[rng.integers(0, len(sym), len(sym))].mean() for _ in range(10_000)]
        out[name] = {
            "weighted_symmetric_fifth": round(float(sym.mean()), 4),
            "weighted_symmetric_ci": [
                round(float(np.percentile(bs, 2.5)), 4),
                round(float(np.percentile(bs, 97.5)), 4),
            ],
            "weighted_ascending_fifth": round(float(asc.mean()), 4),
            "exact": round(float(exact.mean()), 4),
        }
    for conv in ("sym", "asc"):
        d = per["k10"][conv] - per["madmom"][conv]
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)]
        out[f"paired_k10_minus_madmom_{conv}"] = {
            "mean": round(float(d.mean()), 4),
            "ci": [
                round(float(np.percentile(bs, 2.5)), 4),
                round(float(np.percentile(bs, 97.5)), 4),
            ],
            "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum()),
            "ties": int((d == 0).sum()),
        }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
