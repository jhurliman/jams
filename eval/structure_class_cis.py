#!/usr/bin/env -S uv run --extra eval
"""Paired per-class bootstrap CIs for structure gate arms (ST-v3/ST-v4 tables).

Compares two gate arm outputs (raw JSONL prediction files from the gate driver) on
their common tracks: per-class label quality as *per-track* GT-duration coverage
(mean coverage over that track's GT segments of the class; tracks lacking the class
are excluded pairwise), plus aggregate metrics read from the corresponding scored
JSONs. Track-level resampling (tracks are the independent units), 10k resamples,
seed 0, 95% percentile CIs — the numbers in paper/EXPERIMENTS.md ST-v3/ST-v4
per-class tables and paper/arxiv Fig. "structure trade".

Usage:
  uv run --extra eval eval/structure_class_cis.py \
      --arm gate_st4.jsonl --stock gate_stock.jsonl \
      --arm-scored gate_st4_scored.json --stock-scored gate_stock_scored.json \
      --fold 2 [--manifest eval/data/raveform/manifest.jsonl] [--out cis.json]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

CLASSES = (
    "buildup", "cooldown", "drop", "intro", "breakdown",
    "outro", "end", "bridge", "altintro", "altoutro",
)
AGGREGATES = ("pairwise_f", "beat_f", "bound_f_0.5")
N_BOOT = 10_000


def load_preds(path: Path) -> tuple[dict, list[str]]:
    """Return (predictions by track id, ids of rows that carry an ``error``)."""
    out, errors = {}, []
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if r.get("error"):
                errors.append(r["track_id"])
            else:
                out[r["track_id"]] = r
    return out, errors


def track_class_cov(pred: dict, ref_int, ref_lab, want: str) -> float | None:
    segs = pred.get("segments") or []
    covs = []
    for (a, b), lab in zip(ref_int, ref_lab):
        if lab != want or b <= a:
            continue
        cov = sum(
            max(0.0, min(b, e["end"]) - max(a, e["start"]))
            for e in segs
            if e["label"] == lab
        )
        covs.append(cov / (b - a))
    return st.mean(covs) if covs else None


def boot_ci(deltas: list[float], rng: random.Random) -> dict:
    boots = sorted(st.mean(rng.choices(deltas, k=len(deltas))) for _ in range(N_BOOT))
    return {
        "n": len(deltas),
        "delta": round(st.mean(deltas), 4),
        "ci": [round(boots[249], 4), round(boots[9749], 4)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, type=Path, help="challenger arm JSONL")
    ap.add_argument("--stock", required=True, type=Path, help="stock arm JSONL")
    ap.add_argument("--arm-scored", required=True, type=Path)
    ap.add_argument("--stock-scored", required=True, type=Path)
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent / "data/raveform/manifest.jsonl",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="intersect the arms even if they do not both cover the eligible fold (default: fail)",
    )
    args = ap.parse_args()

    import evaluate_structure as es  # noqa: PLC0415 — sibling module, heavy imports

    rows = [json.loads(x) for x in open(args.manifest) if x.strip()]
    rows = {
        r["track_id"]: r
        for r in rows
        if r.get("fold") == args.fold and r.get("audio_exists")
    }

    arm, arm_errors = load_preds(args.arm)
    stock, stock_errors = load_preds(args.stock)
    eligible = set(rows)
    problems = []
    if arm_errors:
        problems.append(f"{len(arm_errors)} error rows in --arm: {sorted(arm_errors)[:10]}")
    if stock_errors:
        problems.append(f"{len(stock_errors)} error rows in --stock: {sorted(stock_errors)[:10]}")
    for name, ids in (("--arm", set(arm)), ("--stock", set(stock))):
        if ids != eligible:
            problems.append(
                f"{name} tracks differ from the eligible fold-{args.fold} manifest: "
                f"missing {sorted(eligible - ids)[:10]} extra {sorted(ids - eligible)[:10]}"
            )
    if problems and not args.allow_partial:
        raise SystemExit(
            "Both gate arms must cover exactly the eligible held-out tracks with no error rows "
            "(the ST-v3/ST-v4 ledger entries are 165-track comparisons). Problems:\n  - "
            + "\n  - ".join(problems)
            + "\nRe-run with --allow-partial to intersect anyway (the output then describes a subset)."
        )
    common = sorted(set(arm) & set(stock) & eligible)
    if problems:
        print("WARNING: partial comparison; " + "; ".join(problems))
    print(f"paired tracks: {len(common)} of {len(eligible)} eligible")

    refs = {}
    for tid in common:
        _, _, ref_int, ref_lab = es.load_refs(rows[tid])
        refs[tid] = (ref_int, ref_lab)

    rng = random.Random(0)
    results = {}
    for name in CLASSES:
        pairs = []
        for tid in common:
            ref_int, ref_lab = refs[tid]
            a = track_class_cov(arm[tid], ref_int, ref_lab, name)
            b = track_class_cov(stock[tid], ref_int, ref_lab, name)
            if a is not None and b is not None:
                pairs.append(a - b)
        if pairs:
            results[name] = boot_ci(pairs, rng)

    pt_arm = {t["track_id"]: t for t in json.load(open(args.arm_scored))["per_track"]}
    pt_stock = {t["track_id"]: t for t in json.load(open(args.stock_scored))["per_track"]}
    for m in AGGREGATES:
        deltas = [
            pt_arm[t][m] - pt_stock[t][m]
            for t in common
            if pt_arm[t][m] is not None and pt_stock[t][m] is not None
        ]
        results[m] = boot_ci(deltas, rng)

    print(json.dumps(results, indent=1))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
