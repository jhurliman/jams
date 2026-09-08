#!/usr/bin/env -S uv run --extra eval
"""Paired per-class bootstrap CIs for structure gate arms (ST-v3/ST-v4 tables).

Compares two gate arm outputs (raw JSONL prediction files from the gate driver), which
must both cover exactly the eligible held-out fold: per-class label quality as
*per-track* GT-duration coverage (mean coverage over that track's GT segments of the
class; tracks lacking the class are excluded pairwise), plus aggregate metrics
recomputed per track from the raw predictions with ``evaluate_structure.score_track``.
Scored JSONs (``--arm-scored``/``--stock-scored``) are optional cross-checks: their
track sets must equal the arm's and every per-track metric must match the
recomputation, otherwise the run fails. Track-level resampling (tracks are the
independent units), 10k resamples,
seed 0, 95% percentile CIs — the numbers in paper/EXPERIMENTS.md ST-v3/ST-v4
per-class tables and paper/arxiv Fig. "structure trade".

Usage:
  uv run --extra eval eval/structure_class_cis.py \
      --arm gate_st4.jsonl --stock gate_stock.jsonl \
      --fold 2 [--arm-scored gate_st4_scored.json] [--stock-scored gate_stock_scored.json] \
      [--manifest eval/data/raveform/manifest.jsonl] [--out cis.json]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

CLASSES = (
    "buildup",
    "cooldown",
    "drop",
    "intro",
    "breakdown",
    "outro",
    "end",
    "bridge",
    "altintro",
    "altoutro",
)
AGGREGATES = ("pairwise_f", "beat_f", "bound_f_0.5")  # paired deltas reported
# Every per-track field a scored artifact carries; all are cross-checked against the
# recomputation, not only the three reported aggregates.
SCORED_METRICS = ("beat_f", "downbeat_f", "bound_f_0.5", "bound_f_3.0", "pairwise_f", "v_measure")
N_BOOT = 10_000


def load_preds(path: Path) -> tuple[dict, list[str]]:
    """Return (predictions by track id, ids of rows that carry an ``error``).

    A track id appearing in more than one successful row makes the arm ambiguous (the
    result would depend on row order), so that is an error rather than last-wins.
    """
    out, errors = {}, []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("error"):
                    errors.append(r["track_id"])
                elif r["track_id"] in out:
                    raise SystemExit(f"{path}: duplicate prediction rows for {r['track_id']}")
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
            max(0.0, min(b, e["end"]) - max(a, e["start"])) for e in segs if e["label"] == lab
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
    ap.add_argument(
        "--arm-scored", type=Path, default=None, help="optional scored JSON to cross-check"
    )
    ap.add_argument(
        "--stock-scored", type=Path, default=None, help="optional scored JSON to cross-check"
    )
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "paper/evidence/structure/raveform_eval_manifest.jsonl",
        help="frozen evaluation manifest (fold membership + reference paths); default: the "
        "committed folds-1/2 manifest, so the held-out sets do not depend on current audio",
    )
    ap.add_argument(
        "--data-home",
        type=Path,
        default=Path(__file__).parent / "data/raveform",
        help="Raveform annotation root for relative beats_csv paths "
        "(populate with `eval/acquire_raveform.py --no-audio`)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="intersect the arms even if they do not both cover the eligible fold (default: fail)",
    )
    args = ap.parse_args()

    import evaluate_structure as es  # noqa: PLC0415 — sibling module, heavy imports

    with open(args.manifest) as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    rows = [r for r in rows if "track_id" in r]  # the frozen manifest starts with a header row
    # Eligibility is the frozen flag ("eligible": audio retrievable at gate time), falling back
    # to a live manifest's audio_exists; it must not depend on what is downloadable today.
    rows = {
        r["track_id"]: r
        for r in rows
        if r.get("fold") == args.fold and r.get("eligible", r.get("audio_exists"))
    }
    missing_refs = []
    for r in rows.values():
        csv = Path(r["beats_csv"])
        if not csv.is_absolute():
            csv = args.data_home / csv
            r["beats_csv"] = str(csv)
        if not csv.exists():
            missing_refs.append(r["track_id"])
    if missing_refs:
        raise SystemExit(
            f"{len(missing_refs)} of {len(rows)} eligible fold-{args.fold} tracks lack their "
            f"Raveform beat CSV under {args.data_home} (e.g. {missing_refs[:3]}); run "
            "`uv run --extra eval eval/acquire_raveform.py --no-audio` (annotations only)."
        )

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
            + "\nRe-run with --allow-partial to intersect anyway "
            + "(the output then describes a subset)."
        )
    common = sorted(set(arm) & set(stock) & eligible)
    if problems:
        print("WARNING: partial comparison; " + "; ".join(problems))
    print(f"paired tracks: {len(common)} of {len(eligible)} eligible")

    refs = {}
    full_refs = {}
    for tid in common:
        ref_beats, ref_down, ref_int, ref_lab = es.load_refs(rows[tid])
        full_refs[tid] = (ref_beats, ref_down, ref_int, ref_lab)
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

    # Aggregate metrics recomputed from the raw predictions (the same scorer the gate used).
    pt_arm = {tid: es.score_track(*full_refs[tid], arm[tid]) for tid in common}
    pt_stock = {tid: es.score_track(*full_refs[tid], stock[tid]) for tid in common}
    for label, scored_path, computed in (
        ("--arm-scored", args.arm_scored, pt_arm),
        ("--stock-scored", args.stock_scored, pt_stock),
    ):
        if scored_path is None:
            continue
        with open(scored_path) as fh:
            scored_rows = json.load(fh)["per_track"]
        ids = [r["track_id"] for r in scored_rows]
        if len(ids) != len(set(ids)) or set(ids) != set(common):
            raise SystemExit(
                f"{label}: scored track set (n={len(ids)}, {len(set(ids))} unique) does not "
                f"equal the paired arm set (n={len(common)})"
            )
        for r in scored_rows:
            for m in SCORED_METRICS:
                a, b = r.get(m), computed[r["track_id"]].get(m)
                for v in (a, b):
                    bad_type = isinstance(v, bool) or not isinstance(v, (int, float))
                    if v is not None and (bad_type or not math.isfinite(v)):
                        raise SystemExit(f"{label}: {r['track_id']} {m} non-finite value {v!r}")
                if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
                    raise SystemExit(
                        f"{label}: {r['track_id']} {m} scored={a} recomputed={b}; the scored "
                        "artifact does not correspond to the supplied raw predictions"
                    )
        print(
            f"{label}: {len(ids)} tracks match the recomputation on all "
            f"{len(SCORED_METRICS)} metrics"
        )
    for m in AGGREGATES:
        # Every paired track must carry a finite value from both arms; a None (no beats or
        # segments, or intervals the scorer rejected) would silently shrink the support
        # while the coverage checks above still report the full fold.
        missing = [
            t
            for t in common
            for v in (pt_arm[t][m], pt_stock[t][m])
            if v is None
            or isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
        ]
        if missing and not args.allow_partial:
            raise SystemExit(
                f"{m}: {len(set(missing))} paired tracks lack a finite score in one arm "
                f"(e.g. {sorted(set(missing))[:5]}); re-run with --allow-partial to drop them"
            )
        deltas = [pt_arm[t][m] - pt_stock[t][m] for t in common if t not in set(missing)]
        results[m] = boot_ci(deltas, rng)
        results[m]["dropped"] = len(set(missing))

    print(json.dumps(results, indent=1))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
