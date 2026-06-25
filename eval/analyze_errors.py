#!/usr/bin/env -S uv run --extra eval
"""Deep error analysis of an eval run, oriented to the electronic/DJ domain.

Joins per-track predictions (results JSON from evaluate.py) with genres from the
manifest and reports where the service loses accuracy and *why*, so we know what
to fix next:

  KEY
    - error taxonomy (exact / fifth / relative / parallel / other)
    - mode-confusion direction (major->minor vs minor->major) + per-mode recall
    - top confused ref->pred key pairs
    - per-genre MIREX / exact (weak genres surface here)
  TEMPO
    - octave-error classes (half / double / third / triple / other)
    - signed bias + error distribution
    - per-genre Acc1
    - the hardest tracks (key AND tempo both wrong)

    uv run eval/analyze_errors.py                          # uses data/results_sota.json
    uv run eval/analyze_errors.py --results data/results.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH = {n: i for i, n in enumerate(NOTES)}


def parse_key(k):
    if not k:
        return None
    p = k.split()
    return (PITCH[p[0]], p[1]) if len(p) == 2 and p[0] in PITCH else None


def relation(ref, pred):
    r, p = parse_key(ref), parse_key(pred)
    if r is None or p is None:
        return "unparsed"
    iv = (p[0] - r[0]) % 12
    if iv == 0 and p[1] == r[1]:
        return "exact"
    if p[1] == r[1] and iv in (7, 5):
        return "fifth"
    if (r[1] == "major" and p[1] == "minor" and iv == 9) or (r[1] == "minor" and p[1] == "major" and iv == 3):
        return "relative"
    if iv == 0 and p[1] != r[1]:
        return "parallel"
    return "other"


def tempo_class(ref, pred, tol=0.04):
    if not ref or not pred:
        return "n/a"
    if abs(pred - ref) <= tol * ref:
        return "correct"
    for f, name in ((0.5, "half"), (2.0, "double"), (1 / 3, "third"), (3.0, "triple"),
                    (2 / 3, "two-thirds"), (1.5, "3:2")):
        if abs(pred * f - ref) <= tol * ref:
            return name
    return "other"


def bar(frac, width=24):
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def pct(a, b):
    return f"{100*a/b:5.1f}%" if b else "  n/a"


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results", type=Path, default=here / "data" / "results_sota.json")
    ap.add_argument("--manifest", type=Path, default=here / "data" / "manifest.jsonl")
    ap.add_argument("--min-genre", type=int, default=8, help="Min tracks to report a genre")
    args = ap.parse_args()

    res = json.loads(args.results.read_text())
    genre = {}
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g = (r.get("genres") or ["?"])
        genre[r["track_id"]] = (g[0] if g else "?")

    n = len(res)
    print(f"\n{'='*64}\n  ERROR ANALYSIS  —  {n} tracks  ({args.results.name})\n{'='*64}")

    # ---------- KEY ----------
    rel_counts = Counter()
    mode_conf = Counter()           # 'maj->min' / 'min->maj'
    by_refmode = defaultdict(lambda: [0, 0])   # refmode -> [n, exact]
    confused = Counter()            # (ref,pred) for non-exact
    g_key = defaultdict(lambda: [0, 0.0, 0])   # genre -> [n, mirex_sum, exact]
    weights = {"exact": 1.0, "fifth": 0.5, "relative": 0.3, "parallel": 0.2, "other": 0.0, "unparsed": 0.0}

    for row in res:
        ref, pred = row.get("ref_key"), row.get("pred_key")
        rel = relation(ref, pred)
        rel_counts[rel] += 1
        rp, pp = parse_key(ref), parse_key(pred)
        if rp:
            by_refmode[rp[1]][0] += 1
            if rel == "exact":
                by_refmode[rp[1]][1] += 1
        if rel == "parallel" and rp and pp:
            mode_conf["maj->min" if rp[1] == "major" else "min->maj"] += 1
        if rel != "exact" and ref and pred:
            confused[(ref, pred)] += 1
        g = genre.get(row["track_id"], "?")
        gk = g_key[g]
        gk[0] += 1
        gk[1] += weights[rel]
        gk[2] += 1 if rel == "exact" else 0

    print("\n── KEY error taxonomy ───────────────────────────────")
    for k in ("exact", "fifth", "relative", "parallel", "other"):
        c = rel_counts[k]
        print(f"  {k:9} {bar(c/n)} {c:4} {pct(c,n)}")
    mirex = sum(weights[k] * v for k, v in rel_counts.items()) / n
    print(f"  MIREX {mirex:.4f}   exact {rel_counts['exact']/n:.4f}")

    print("\n── Mode handling (EDM skews minor) ──────────────────")
    for m in ("major", "minor"):
        nn, ex = by_refmode[m]
        print(f"  ref {m:5}: {nn:4} tracks, exact recall {pct(ex,nn)}")
    if mode_conf:
        print(f"  parallel mode flips: {dict(mode_conf)}  (which direction we wrongly flip)")

    print("\n── Top confused ref → pred (non-exact) ──────────────")
    for (ref, pred), c in confused.most_common(10):
        print(f"  {c:3}x  {ref:9} → {pred:9}  [{relation(ref,pred)}]")

    print("\n── KEY by genre (weak genres first) ─────────────────")
    rows = [(g, v[0], v[1] / v[0], v[2] / v[0]) for g, v in g_key.items() if v[0] >= args.min_genre]
    for g, c, mx, ex in sorted(rows, key=lambda x: x[2]):
        print(f"  {g:22} n={c:3}  MIREX {mx:.3f}  exact {ex:.3f}")

    # ---------- TEMPO ----------
    tcls = Counter()
    errs = []
    g_tempo = defaultdict(lambda: [0, 0])
    has_tempo = [r for r in res if r.get("ref_tempo") and r.get("pred_tempo")]
    for row in has_tempo:
        ref, pred = float(row["ref_tempo"]), float(row["pred_tempo"])
        c = tempo_class(ref, pred)
        tcls[c] += 1
        if c == "correct":
            errs.append((pred - ref) / ref)
        g = genre.get(row["track_id"], "?")
        gt = g_tempo[g]
        gt[0] += 1
        gt[1] += 1 if c == "correct" else 0

    if has_tempo:
        m = len(has_tempo)
        print(f"\n── TEMPO octave-error classes — {m} tracks ──────────")
        for k, c in tcls.most_common():
            print(f"  {k:10} {bar(c/m)} {c:4} {pct(c,m)}")
        acc1 = tcls["correct"] / m
        acc2 = (tcls["correct"] + tcls["half"] + tcls["double"] + tcls["third"]
                + tcls["triple"] + tcls["two-thirds"] + tcls["3:2"]) / m
        print(f"  Acc1 {acc1:.4f}   Acc2 {acc2:.4f}")
        if errs:
            mean = sum(errs) / len(errs)
            print(f"  within-tol signed bias: {mean*100:+.2f}%  (n={len(errs)})")

        print("\n── TEMPO Acc1 by genre (weak first) ─────────────────")
        rows = [(g, v[0], v[1] / v[0]) for g, v in g_tempo.items() if v[0] >= args.min_genre]
        for g, c, a in sorted(rows, key=lambda x: x[2]):
            print(f"  {g:22} n={c:3}  Acc1 {a:.3f}")

    # ---------- HARDEST ----------
    print("\n── Hardest tracks (key non-exact AND tempo wrong) ───")
    hard = []
    for row in res:
        if relation(row.get("ref_key"), row.get("pred_key")) == "exact":
            continue
        if row.get("ref_tempo") and row.get("pred_tempo"):
            if tempo_class(float(row["ref_tempo"]), float(row["pred_tempo"])) != "correct":
                hard.append(row)
    print(f"  {len(hard)} tracks miss on both. Sample:")
    for row in hard[:8]:
        print(f"    {genre.get(row['track_id'],'?'):18} key {row.get('ref_key')}→{row.get('pred_key')}  "
              f"tempo {row.get('ref_tempo')}→{row.get('pred_tempo')}")


if __name__ == "__main__":
    main()
