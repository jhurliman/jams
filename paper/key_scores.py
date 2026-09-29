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
accuracy; paired K10 - madmom differences with 10,000-resample track bootstrap, seed 0;
counts per credit category; descriptive per-genre means; and a reference-label
sensitivity (the same predictions rescored on the excerpts whose raw GiantSteps
annotation, `ref_key_raw`, is a single key, and on the plain major/minor ones).

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


def category(ref, est):
    """Which symmetric-fifth credit bucket a prediction falls in."""
    r, e = parse(ref), parse(est)
    iv = (e[0] - r[0]) % 12
    if iv == 0 and e[1] == r[1]:
        return "exact"
    if e[1] == r[1] and iv in (7, 5):
        return "fifth"
    if (r[1], e[1], iv) in (("major", "minor", 9), ("minor", "major", 3)):
        return "relative"
    if iv == 0:
        return "parallel"
    return "other"


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
    # Error categories of the symmetric-fifth scorer (counts, one per track).
    out["error_categories"] = {
        name: {
            c: int(sum(category(r["ref_key"], r[f"{name}_key"]) == c for r in usable))
            for c in ("exact", "fifth", "relative", "parallel", "other")
        }
        for name in ("k10", "madmom")
    }
    # Descriptive per-genre means (symmetric fifth) for K10. Genre is the GiantSteps
    # Beatport genre tag; every excerpt has at most one, and 100 of the 567 have none.
    genre = {}
    for r in usable:
        g = (r.get("genre") or [None])[0] or "(no genre tag)"
        genre.setdefault(g, []).append(legacy(r["ref_key"], r["k10_key"]))
    out["k10_by_genre"] = {
        g: {"n": len(v), "weighted_symmetric_fifth": round(float(np.mean(v)), 4)}
        for g, v in sorted(genre.items(), key=lambda kv: -len(kv[1]))
    }
    for conv in ("sym", "asc", "exact"):
        if conv == "exact":
            d = np.array(
                [
                    float(norm(r["ref_key"]) == norm(r["k10_key"]))
                    - float(norm(r["ref_key"]) == norm(r["madmom_key"]))
                    for r in usable
                ]
            )
        else:
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
    # Reference-label sensitivity. The acquisition script normalized every GiantSteps Key
    # annotation to one major/minor key: modal qualifiers were collapsed by mode family and
    # two-key annotations ("C major | C minor phrygian") were collapsed by the same rule
    # (tonic of the first key; minor if any listed key is minor-family). The rows below
    # rescore the same predictions on the subsets whose reference is a single annotated key.
    raw = {r["track_id"]: (r.get("ref_key_raw") or "") for r in usable}
    single = [r for r in usable if "|" not in raw[r["track_id"]]]
    plain = [
        r
        for r in single
        if raw[r["track_id"]].lower().split()[1:]
        in (["major"], ["minor"], ["major", "ionian"], ["minor", "aeolian"])
    ]
    out["reference_label_categories"] = {
        "two_key_annotation_collapsed": len(usable) - len(single),
        "single_key_modal_qualifier_collapsed": len(single) - len(plain),
        "single_key_plain_major_minor_or_ionian_aeolian": len(plain),
    }
    # Rows whose normalized reference equals neither key listed in a two-key annotation
    # (15 excerpts); excluding only those is the narrowest correction.
    listed = [r for r in usable if r.get("ref_matches_listed_key", True)]
    out["reference_label_categories"]["reference_matches_no_listed_key"] = len(usable) - len(listed)
    for name, sub in (
        ("excluding_neither_key_subset", listed),
        ("single_key_subset", single),
        ("plain_major_minor_subset", plain),
    ):
        s = {"n": len(sub)}
        for sysname in ("k10", "madmom"):
            s[sysname] = {
                "weighted_symmetric_fifth": round(
                    float(np.mean([legacy(r["ref_key"], r[f"{sysname}_key"]) for r in sub])), 4
                ),
                "exact": round(
                    float(np.mean([norm(r["ref_key"]) == norm(r[f"{sysname}_key"]) for r in sub])),
                    4,
                ),
            }
        d = np.array(
            [
                legacy(r["ref_key"], r["k10_key"]) - legacy(r["ref_key"], r["madmom_key"])
                for r in sub
            ]
        )
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)]
        s["paired_k10_minus_madmom_sym"] = {
            "mean": round(float(d.mean()), 4),
            "ci": [
                round(float(np.percentile(bs, 2.5)), 4),
                round(float(np.percentile(bs, 97.5)), 4),
            ],
        }
        out[name] = s
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
