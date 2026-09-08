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


def read_scores(path: Path, group: str = "other", key: str = "note_f") -> dict[str, float]:
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
        value = values.get(key)
        if value is not None:
            if tid in result:
                raise ValueError(f"{path}: duplicate score for {tid}/{group}")
            result[tid] = score(value)
    if not result:
        raise ValueError(f"{path}: no eligible {group} scores")
    return result


def paired_values(a: dict[str, float], b: dict[str, float], n: int):
    if set(a) != set(b):
        raise ValueError(
            f"Unmatched IDs: A-only={sorted(set(a) - set(b))[:10]}, "
            f"B-only={sorted(set(b) - set(a))[:10]}"
        )
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
    check_secondary_sections(data, rows)
    # Falsifies the old exclusion-only explanation under its own assumptions.
    subset_bound = 604 * 0.746 / 567
    if not subset_bound < 0.8328:
        raise ValueError("Unexpected subset-bound calculation")
    return {
        "status": "summary_consistent_archives_not_checked",
        "snapshot_status": data["status"],
        "reference_delta": round(delta, 4),
        "end_to_end_delta": round(rows["T10"]["mean"] - rows["S4"]["mean"], 4),
        "old_key_subset_bound": subset_bound,
    }


def _interval(
    name: str, mean: float, ci, lo_bound: float | None = None, hi_bound: float | None = None
) -> None:
    lo, hi = ci
    for v in (mean, lo, hi):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name}: non-numeric or non-finite value {v!r}")
        if lo_bound is not None and v < lo_bound or hi_bound is not None and v > hi_bound:
            raise ValueError(f"{name}: value {v} outside [{lo_bound}, {hi_bound}]")
    if not lo <= mean <= hi:
        raise ValueError(f"{name}: mean {mean} outside interval [{lo}, {hi}]")


def check_secondary_sections(data: dict, rows: dict) -> None:
    """Every numeric section make_figures.py or the manuscript consumes, not only the four rows."""
    sep = {r["id"]: r for r in data["separation"]}
    if set(sep) != {"S1", "S4"} or len(data["separation"]) != 2:
        raise ValueError("Expected separation rows S1 and S4")
    for r in sep.values():
        for g in ("drums", "bass", "other"):
            v = r[f"{g}_si_sdr"]
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not -10 <= v <= 40
            ):
                raise ValueError(f"separation {r['id']} {g}_si_sdr implausible: {v!r}")
            score(r[f"{g}_f1"])
        if r["nominal_tracks"] != 151 or (r["bass_n"] is not None and not 0 < r["bass_n"] <= 151):
            raise ValueError(f"separation {r['id']}: unexpected support")
    if not math.isclose(sep["S4"]["other_f1"], rows["S4"]["mean"], abs_tol=5e-5):
        raise ValueError("separation S4 other_f1 disagrees with transcription row S4")
    b = data["bass_reference"]
    for k in ("basic_pitch", "yourmt3"):
        score(b[k])
    if b["n"] != 143 or round(b["yourmt3"] - b["basic_pitch"], 4) != round(b["paired_delta"], 4):
        raise ValueError("bass_reference support or paired delta inconsistent")
    _interval("bass_reference", b["paired_delta"], b["paired_ci"], -1, 1)
    rp = data["reference_paired_delta"]
    _interval("reference_paired_delta", rp["mean"], rp["ci"], -1, 1)
    if not 0 <= rp["win_fraction"] <= 1:
        raise ValueError("reference_paired_delta win_fraction outside [0, 1]")
    for c in data.get("paired_contrasts", []):
        a, _, bb = c["comparison"].partition(" - ")
        if a not in rows or bb not in rows or c["n"] != 151:
            raise ValueError(f"paired contrast {c['comparison']}: unknown rows or support")
        if abs((rows[a]["mean"] - rows[bb]["mean"]) - c["mean"]) > 1.5e-4:
            raise ValueError(f"paired contrast {c['comparison']}: mean inconsistent with row means")
        _interval(c["comparison"], c["mean"], c["ci"], -1, 1)
        if (
            not 0 <= c["win_fraction"] <= 1
            or not 0 <= c["loss_fraction"] <= 1
            or c["win_fraction"] + c["loss_fraction"] > 1 + 1e-9
        ):
            raise ValueError(f"paired contrast {c['comparison']}: win/loss fractions invalid")
    spd = data.get("separator_paired_delta")
    if spd:
        for key, n_expected in (("drums_onset_f1", 151), ("other_f1", 151), ("bass_f1", 143)):
            c = spd[key]
            if c["n"] != n_expected:
                raise ValueError(f"separator_paired_delta {key}: support {c['n']} != {n_expected}")
            _interval(f"separator_paired_delta {key}", c["mean"], c["ci"], -1, 1)
        # Row means and paired means are rounded independently: allow one unit in the 4th place.
        for key, g in (
            ("other_f1", "other_f1"),
            ("drums_onset_f1", "drums_f1"),
            ("bass_f1", "bass_f1"),
        ):
            if abs((sep["S4"][g] - sep["S1"][g]) - spd[key]["mean"]) > 1.5e-4:
                raise ValueError(f"separator_paired_delta {key} inconsistent with separation rows")


def check_archives(data: dict, root: Path) -> dict:
    rows = data["transcription"]
    sep_rows = {r["id"]: r for r in data["separation"]}
    needed = [r["archive"] for r in rows] + [r["archive"] for r in sep_rows.values()]
    needed.append(sep_rows["S4"]["si_sdr_archive"])
    missing = [str(root / a) for a in dict.fromkeys(needed) if not (root / a).is_file()]
    if missing:
        raise ValueError(
            "Required archives missing; no reproduction claim is possible:\n" + "\n".join(missing)
        )
    loaded = {r["id"]: read_scores(root / r["archive"]) for r in rows}
    report = {
        "status": "stored_scores_recomputed",
        "archives": {},
        "contrasts": {},
        "limitations": [
            "Stored scores were not rescored from MIDI or reproduced by inference.",
            "CIs use sorted track IDs; finite-bootstrap endpoints may differ from historical row order.",
            "Matching filenames alone does not establish correct dataset/checkpoint identity.",
        ],
    }
    base = loaded["T1"]
    for r in rows:
        ids, _, vals = paired_values(base, loaded[r["id"]], r["n"])
        stats = bootstrap(vals, **{k: data["bootstrap"][k] for k in ("resamples", "seed")})
        if abs(stats["mean"] - r["mean"]) > 0.00005001:
            raise ValueError(
                f"{r['id']}: recomputed {stats['mean']:.8f} disagrees with reported {r['mean']}"
            )
        digest = hashlib.sha256((root / r["archive"]).read_bytes()).hexdigest()
        if r.get("sha256") and r["sha256"] != digest:
            raise ValueError(
                f"{r['id']}: archive SHA-256 {digest} differs from snapshot {r['sha256']}"
            )
        if r.get("ci") is not None and r.get("ci_provenance") == "recomputed_from_stored_scores":
            if [round(x, 4) for x in stats["ci"]] != [round(x, 4) for x in r["ci"]]:
                raise ValueError(
                    f"{r['id']}: recomputed CI {stats['ci']} disagrees with snapshot {r['ci']}"
                )
        report["archives"][r["id"]] = {
            "path": r["archive"],
            "n": len(ids),
            **stats,
            "sha256": digest,
            "track_ids": ids,
        }
    recorded = {c["comparison"]: c for c in data.get("paired_contrasts", [])}
    for a, b in (("T2b", "T1"), ("T10", "S4"), ("S4", "T1"), ("T10", "T2b")):
        _, va, vb = paired_values(loaded[a], loaded[b], 151)
        diffs = [x - y for x, y in zip(va, vb)]
        stats = bootstrap(diffs, **{k: data["bootstrap"][k] for k in ("resamples", "seed")})
        stats["win_fraction"] = sum(d > 0 for d in diffs) / len(diffs)
        stats["loss_fraction"] = sum(d < 0 for d in diffs) / len(diffs)
        report["contrasts"][f"{a} - {b}"] = stats
        rec = recorded.get(f"{a} - {b}")
        if rec is not None:
            if (
                round(stats["mean"], 4) != rec["mean"]
                or [round(x, 4) for x in stats["ci"]] != rec["ci"]
                or round(stats["win_fraction"], 4) != rec["win_fraction"]
                or round(stats["loss_fraction"], 4) != rec["loss_fraction"]
            ):
                raise ValueError(
                    f"Contrast {a} - {b}: recomputed {stats} disagrees with snapshot {rec}"
                )
    # The separately published reference contrast must equal the recomputed T2b - T1 and
    # its duplicate paired_contrasts record.
    rp = data["reference_paired_delta"]
    ref = report["contrasts"]["T2b - T1"]
    if (
        round(ref["mean"], 4) != rp["mean"]
        or [round(x, 4) for x in ref["ci"]] != rp["ci"]
        or round(ref["win_fraction"], 4) != rp["win_fraction"]
    ):
        raise ValueError(f"reference_paired_delta {rp} disagrees with recomputed T2b - T1 {ref}")
    dup = recorded.get("T2b - T1")
    if dup is not None and (dup["mean"], dup["ci"], dup["win_fraction"]) != (
        rp["mean"],
        rp["ci"],
        rp["win_fraction"],
    ):
        raise ValueError("reference_paired_delta and paired_contrasts['T2b - T1'] disagree")
    boot_kw = {k: data["bootstrap"][k] for k in ("resamples", "seed")}
    report["bass_reference"] = check_bass_reference(data, root, boot_kw)
    report["separator"] = check_separator(data, root, boot_kw)
    return report


def _close4(a: float, b: float) -> bool:
    return round(a, 4) == round(b, 4)


def check_bass_reference(data: dict, root: Path, boot_kw: dict) -> dict:
    """Recompute the 143-pair bass comparison (T1 vs T2b, bass group) from the archives."""
    rows = {r["id"]: r for r in data["transcription"]}
    b = data["bass_reference"]
    t1 = read_scores(root / rows["T1"]["archive"], group="bass")
    t2 = read_scores(root / rows["T2b"]["archive"], group="bass")
    ids, v1, v2 = paired_values(t1, t2, b["n"])
    mean1, mean2 = sum(v1) / len(v1), sum(v2) / len(v2)
    if abs(mean1 - b["basic_pitch"]) > 0.00005001 or abs(mean2 - b["yourmt3"]) > 0.00005001:
        raise ValueError(f"bass_reference means {mean1:.6f}/{mean2:.6f} disagree with snapshot")
    diffs = [y - x for x, y in zip(v1, v2, strict=True)]
    stats = bootstrap(diffs, **boot_kw)
    if (
        not _close4(stats["mean"], b["paired_delta"])
        or [round(x, 4) for x in stats["ci"]] != b["paired_ci"]
    ):
        raise ValueError(f"bass_reference paired delta {stats} disagrees with snapshot {b}")
    return {
        "n": len(ids),
        "basic_pitch_mean": mean1,
        "yourmt3_mean": mean2,
        **stats,
        "win_fraction": sum(d > 0 for d in diffs) / len(diffs),
        "track_ids": ids,
    }


def check_separator(data: dict, root: Path, boot_kw: dict) -> dict:
    """Recompute the S1/S4 separator table and paired deltas from the per-track archives."""
    sep = {r["id"]: r for r in data["separation"]}
    out: dict = {"rows": {}, "paired": {}}
    per_track: dict[str, dict[str, dict[str, float]]] = {}
    for sid, r in sep.items():
        path = root / r["archive"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        scores = {
            "drums": read_scores(path, group="drums", key="drum_onset_f"),
            "bass": read_scores(path, group="bass"),
            "other": read_scores(path, group="other"),
        }
        per_track[sid] = scores
        row_report = {"path": r["archive"], "sha256": digest, "f1": {}, "si_sdr": {}}
        for g, key in (("drums", "drums_f1"), ("bass", "bass_f1"), ("other", "other_f1")):
            n_expected = 151 if g != "bass" else r["bass_n"]
            if len(scores[g]) != n_expected:
                raise ValueError(f"separation {sid} {g}: support {len(scores[g])} != {n_expected}")
            mean = sum(scores[g].values()) / len(scores[g])
            if abs(mean - r[key]) > 0.00005001:
                raise ValueError(
                    f"separation {sid} {key}: recomputed {mean:.6f} vs snapshot {r[key]}"
                )
            row_report["f1"][g] = mean
        doc = json.loads(path.read_text())
        has_sdr = any(isinstance(t.get("sdr"), dict) for t in doc["per_track"])
        if has_sdr:
            for g in ("drums", "bass", "other"):
                vals = [
                    t["sdr"][g]
                    for t in doc["per_track"]
                    if isinstance(t.get("sdr"), dict) and t["sdr"].get(g) is not None
                ]
                mean = sum(vals) / len(vals)
                if abs(mean - r[f"{g}_si_sdr"]) > 0.0051:
                    raise ValueError(
                        f"separation {sid} {g}_si_sdr: recomputed {mean:.4f} vs snapshot"
                    )
                row_report["si_sdr"][g] = {"mean": mean, "n": len(vals), "source": "per_track"}
        else:
            agg_path = root / r["si_sdr_archive"]
            agg = json.loads(agg_path.read_text())["si_sdr"]
            for g in ("drums", "bass", "other"):
                if abs(agg[g] - r[f"{g}_si_sdr"]) > 0.0051:
                    raise ValueError(
                        f"separation {sid} {g}_si_sdr: aggregate archive {agg[g]} vs snapshot"
                    )
                row_report["si_sdr"][g] = {
                    "mean": agg[g],
                    "n": None,
                    "source": "aggregate_only_archive",
                }
            row_report["si_sdr_archive_sha256"] = hashlib.sha256(agg_path.read_bytes()).hexdigest()
        out["rows"][sid] = row_report
    spd = data.get("separator_paired_delta") or {}
    for key, g, n in (
        ("drums_onset_f1", "drums", 151),
        ("other_f1", "other", 151),
        ("bass_f1", "bass", 143),
    ):
        _, v1, v4 = paired_values(per_track["S1"][g], per_track["S4"][g], n)
        diffs = [b - a for a, b in zip(v1, v4, strict=True)]
        stats = bootstrap(diffs, **boot_kw)
        stats["win_fraction"] = sum(d > 0 for d in diffs) / len(diffs)
        stats["loss_fraction"] = sum(d < 0 for d in diffs) / len(diffs)
        out["paired"][key] = stats
        rec = spd.get(key)
        if rec is not None and (
            not _close4(stats["mean"], rec["mean"])
            or [round(x, 4) for x in stats["ci"]] != rec["ci"]
            or not _close4(stats["win_fraction"], rec["win_fraction"])
            or not _close4(stats["loss_fraction"], rec["loss_fraction"])
        ):
            raise ValueError(
                f"separator_paired_delta {key}: recomputed {stats} disagrees with snapshot {rec}"
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Root containing results_aws/ archives (default: paper/evidence when the "
        "snapshot status is stored_scores_recomputed); requires all primary archives",
    )
    parser.add_argument(
        "--output", type=Path, help="Write a report only after all requested checks succeed"
    )
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
