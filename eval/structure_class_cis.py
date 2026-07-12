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


def load_preds(path: Path) -> dict:
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                out[r["track_id"]] = r
    return out


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
    args = ap.parse_args()

    import evaluate_structure as es  # noqa: PLC0415 — sibling module, heavy imports

    rows = [json.loads(x) for x in open(args.manifest) if x.strip()]
    rows = {
        r["track_id"]: r
        for r in rows
        if r.get("fold") == args.fold and r.get("audio_exists")
    }

    arm = load_preds(args.arm)
    stock = load_preds(args.stock)
    common = sorted(set(arm) & set(stock) & set(rows))
    print(f"paired tracks: {len(common)}")

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
