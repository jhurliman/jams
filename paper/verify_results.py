#!/usr/bin/env python3
"""Validate the reported snapshot and recompute means/CIs from the per-track score archives.

With a stored_scores_recomputed snapshot the archives default to paper/evidence/ and are
always checked: every mean, interval, SHA-256, and paired contrast recorded in the snapshot
must match the recomputation. --data-dir points at another root; it requires all four
primary archives and rejects incomplete or mismatched pairs. It never runs model inference or
proves that stored F1 values were correctly scored from MIDI. Requires numpy for CIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"F1 must be numeric, got {value!r}")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"F1 outside finite [0, 1]: {value!r}")
    return float(value)


def read_scores(path: Path, group: str = "other") -> dict[str, float]:
    """Accept the ledger's nested evaluator format and flat YourMT3+ rescore format.

    None denotes an ineligible reference, never a model failure converted to zero.
    Pair support is checked separately. Unknown schemas fail instead of being guessed.
    """
    doc = json.loads(path.read_text())
    rows = doc.get("per_track")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected nonempty per_track list")
    if doc.get("failed", 0) != 0:
        raise ValueError(f"{path}: reports pipeline failures: {doc['failed']}")
    result, seen = {}, set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("track_id"), (str, int)):
            raise ValueError(f"{path}: missing/invalid track_id")
        tid = str(row["track_id"])
        if "stems" in row:
            identity = (tid, "nested")
            if not isinstance(row["stems"], dict):
                raise ValueError(f"{path}: invalid stems for {tid}")
            values = row["stems"].get(group, {})
        elif "stem" in row and "note_f" in row:
            identity = (tid, str(row["stem"]))
            values = row if row["stem"] == group else {}
        else:
            raise ValueError(f"{path}: unknown row schema for {tid}")
        if identity in seen:
            raise ValueError(f"{path}: duplicate row {identity}")
        seen.add(identity)
        value = values.get("note_f")
        if value is not None:
            if tid in result:
                raise ValueError(f"{path}: duplicate score for {tid}/{group}")
            result[tid] = score(value)
    if not result:
        raise ValueError(f"{path}: no eligible {group} scores")
    return result


def paired_values(a: dict[str, float], b: dict[str, float], n: int):
    if set(a) != set(b):
        raise ValueError(f"Unmatched IDs: A-only={sorted(set(a)-set(b))[:10]}, "
                         f"B-only={sorted(set(b)-set(a))[:10]}")
    if len(a) != n:
        raise ValueError(f"Expected {n} eligible pairs, found {len(a)}")
    ids = sorted(a)
    return ids, [a[t] for t in ids], [b[t] for t in ids]


def bootstrap(values, resamples: int = 10_000, seed: int = 0):
    import numpy as np
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1 or not len(vals) or not np.isfinite(vals).all():
        raise ValueError("Bootstrap requires a nonempty finite vector")
    idx = np.random.default_rng(seed).integers(0, len(vals), size=(resamples, len(vals)))
    means = vals[idx].mean(axis=1)
    return {"mean": float(vals.mean()), "ci": [float(x) for x in np.percentile(means, [2.5, 97.5])]}


STATUSES = ("reported_not_recomputed", "stored_scores_recomputed")


def check_snapshot(data: dict) -> dict:
    if data["status"] not in STATUSES:
        raise ValueError("Snapshot provenance changed; review before use")
    if data["status"] == "stored_scores_recomputed":
        ver = data.get("verification") or {}
        for key in ("report", "archives_dir", "checksums", "command", "scope"):
            if not ver.get(key):
                raise ValueError(f"Recomputed snapshot lacks verification.{key}")
        for r in data["transcription"]:
            if r.get("ci") is None or not r.get("sha256"):
                raise ValueError(f"Recomputed snapshot row {r['id']} lacks ci/sha256")
    rows = {r["id"]: r for r in data["transcription"]}
    if set(rows) != {"T1", "T2b", "S4", "T10"} or len(data["transcription"]) != 4:
        raise ValueError("Expected four unique primary configurations")
    for r in rows.values():
        score(r["mean"])
        if r["n"] != 151:
            raise ValueError("Unexpected accompaniment support")
        if r["ci"] is not None:
            lo, hi = map(score, r["ci"])
            if not lo <= r["mean"] <= hi:
                raise ValueError(f"Invalid interval for {r['id']}")
    delta = rows["T2b"]["mean"] - rows["T1"]["mean"]
    if not math.isclose(delta, data["reference_paired_delta"]["mean"], abs_tol=1e-8):
        raise ValueError("Inconsistent reference-input contrast")
    # Falsifies the old exclusion-only explanation under its own assumptions.
    subset_bound = 604 * 0.746 / 567
    if not subset_bound < 0.8328:
        raise ValueError("Unexpected subset-bound calculation")
    return {"status": "summary_consistent_archives_not_checked",
            "snapshot_status": data["status"],
            "reference_delta": round(delta, 4),
            "end_to_end_delta": round(rows["T10"]["mean"] - rows["S4"]["mean"], 4),
            "old_key_subset_bound": subset_bound}


def check_archives(data: dict, root: Path) -> dict:
    rows = data["transcription"]
    missing = [str(root / r["archive"]) for r in rows if not (root / r["archive"]).is_file()]
    if missing:
        raise ValueError("Required archives missing; no reproduction claim is possible:\n" + "\n".join(missing))
    loaded = {r["id"]: read_scores(root / r["archive"]) for r in rows}
    report = {"status": "stored_scores_recomputed", "archives": {}, "contrasts": {},
              "limitations": ["Stored scores were not rescored from MIDI or reproduced by inference.",
                 "CIs use sorted track IDs; finite-bootstrap endpoints may differ from historical row order.",
                 "Matching filenames alone does not establish correct dataset/checkpoint identity."]}
    base = loaded["T1"]
    for r in rows:
        ids, _, vals = paired_values(base, loaded[r["id"]], r["n"])
        stats = bootstrap(vals, **{k: data["bootstrap"][k] for k in ("resamples", "seed")})
        if abs(stats["mean"] - r["mean"]) > 0.00005001:
            raise ValueError(f"{r['id']}: recomputed {stats['mean']:.8f} disagrees with reported {r['mean']}")
        digest = hashlib.sha256((root / r["archive"]).read_bytes()).hexdigest()
        if r.get("sha256") and r["sha256"] != digest:
            raise ValueError(f"{r['id']}: archive SHA-256 {digest} differs from snapshot {r['sha256']}")
        if r.get("ci") is not None and r.get("ci_provenance") == "recomputed_from_stored_scores":
            if [round(x, 4) for x in stats["ci"]] != [round(x, 4) for x in r["ci"]]:
                raise ValueError(f"{r['id']}: recomputed CI {stats['ci']} disagrees with snapshot {r['ci']}")
        report["archives"][r["id"]] = {"path": r["archive"], "n": len(ids), **stats,
            "sha256": digest, "track_ids": ids}
    recorded = {c["comparison"]: c for c in data.get("paired_contrasts", [])}
    for a, b in (("T2b", "T1"), ("T10", "S4"), ("S4", "T1"), ("T10", "T2b")):
        _, va, vb = paired_values(loaded[a], loaded[b], 151)
        diffs = [x - y for x, y in zip(va, vb)]
        stats = bootstrap(diffs)
        stats["win_fraction"] = sum(d > 0 for d in diffs) / len(diffs)
        stats["loss_fraction"] = sum(d < 0 for d in diffs) / len(diffs)
        report["contrasts"][f"{a} - {b}"] = stats
        rec = recorded.get(f"{a} - {b}")
        if rec is not None:
            if (round(stats["mean"], 4) != rec["mean"]
                    or [round(x, 4) for x in stats["ci"]] != rec["ci"]
                    or round(stats["win_fraction"], 4) != rec["win_fraction"]):
                raise ValueError(f"Contrast {a} - {b}: recomputed {stats} disagrees with snapshot {rec}")
    wins = sum(loaded["T2b"][t] > base[t] for t in base) / len(base)
    if wins != data["reference_paired_delta"]["win_fraction"]:
        raise ValueError("Reference-input win fraction disagrees with snapshot")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        help="Root containing results_aws/ archives (default: paper/evidence when the "
                             "snapshot status is stored_scores_recomputed); requires all primary archives")
    parser.add_argument("--output", type=Path, help="Write a report only after all requested checks succeed")
    args = parser.parse_args()
    try:
        data = json.loads((HERE / "results_snapshot.json").read_text())
        result = check_snapshot(data)
        data_dir = args.data_dir
        if data_dir is None and data["status"] == "stored_scores_recomputed":
            data_dir = HERE / "evidence"
        if data_dir:
            result = check_archives(data, data_dir)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
