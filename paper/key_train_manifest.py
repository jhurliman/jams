# /// script
# requires-python = ">=3.10"
# dependencies = ["mirdata>=0.3.9"]
# ///
"""Reconstruct the K10 training-set membership from the public beatport_key annotations
and cross-check it against the out-of-fold record of the actual run.

The training script (eval/train_key_cnn.py, `acquire`) keeps a beatport_key track when the
FIRST listed key annotation parses as "<tonic> <major|minor>" (flats mapped to sharps) and
the audio file exists; mirdata returns two-key annotations ("E minor | E major") as a list,
and the first entry is used as the label. This script applies that rule to the annotations
alone (no audio needed), which is exact because the run's log records no audio drops
(usable 1363 / 1486; features stage completed with no failures).

Cross-check: paper/evidence/key/k10_oof_train.jsonl is the out-of-fold prediction file of
the run (one row per retained track with its mirdata id, Beatport id, fold, and reference
class). Membership, fold assignment (seed-0 shuffle of the sorted ids dealt round-robin into
5 folds), and labels must all agree, or the script exits nonzero.

Usage: uv run paper/key_train_manifest.py [--data-home DIR]
Writes paper/evidence/key/k10_train_labels.jsonl (retained tracks) and
paper/evidence/key/k10_train_drops.json (dropped tracks with reasons).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import mirdata

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "paper" / "evidence" / "key" / "k10_oof_train.jsonl"
LABELS = ROOT / "paper" / "evidence" / "key" / "k10_train_labels.jsonl"
DROPS = ROOT / "paper" / "evidence" / "key" / "k10_train_drops.json"
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}
SEED = 0


def parse_key(k: str | None):
    """Same rule as eval/train_key_cnn.py: exactly '<tonic> <major|minor>'."""
    p = (k or "").replace("Major", "major").replace("Minor", "minor").split()
    if len(p) != 2:
        return None
    t = FLAT.get(p[0], p[0])
    if t not in NOTES or p[1] not in ("major", "minor"):
        return None
    return NOTES.index(t) * 2 + (1 if p[1] == "minor" else 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-home", type=Path, default=Path.home() / "mir_datasets" / "beatport_key")
    a = ap.parse_args()
    bp = mirdata.initialize("beatport_key", data_home=str(a.data_home))
    bp.download(partial_download=["keys", "metadata"], cleanup=True)
    oof = {}
    for line in OOF.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            oof[r["bp_tid"]] = r
    kept, drops = {}, []
    for tid in bp.track_ids:
        t = bp.track(tid)
        keys = t.key if isinstance(t.key, list) else [t.key]
        keys = [k.strip() for k in keys]
        cls = parse_key(keys[0])
        if cls is None:
            reason = "first listed key is not a single major/minor key"
            if len(keys) > 1:
                reason = "two-key annotation whose first key is not a single major/minor key"
            drops.append({"bp_tid": tid, "annotation": keys, "reason": reason})
            continue
        kept[tid] = {
            "bp_tid": tid,
            "beatport_id": (re.match(r"(\d+)", Path(t.audio_path).name) or [None, None])[1]
            if t.audio_path
            else None,
            "annotation": keys,
            "label_used": f"{NOTES[cls // 2]} {'minor' if cls % 2 else 'major'}",
            "label_cls": cls,
            "two_key_annotation": len(keys) > 1,
        }
    tids = sorted(kept)
    rng = random.Random(SEED)
    rng.shuffle(tids)
    for i in range(5):
        for tid in tids[i::5]:
            kept[tid]["fold"] = i
    problems = []
    if set(kept) != set(oof):
        problems.append(f"membership differs: {len(set(kept) ^ set(oof))} ids")
    for tid, r in kept.items():
        o = oof.get(tid)
        if o is None:
            continue
        if o["fold"] != r["fold"]:
            problems.append(f"{tid}: fold {r['fold']} vs run {o['fold']}")
        if o["ref_cls"] != r["label_cls"]:
            problems.append(f"{tid}: label {r['label_cls']} vs run {o['ref_cls']}")
        if r["beatport_id"] and str(o["track_id"]) != r["beatport_id"]:
            problems.append(f"{tid}: beatport id {r['beatport_id']} vs run {o['track_id']}")
    summary = {
        "index_tracks": len(bp.track_ids),
        "retained": len(kept),
        "retained_single_key": sum(not r["two_key_annotation"] for r in kept.values()),
        "retained_two_key_first_used": sum(r["two_key_annotation"] for r in kept.values()),
        "dropped": len(drops),
        "drop_reasons": dict(Counter(d["reason"] for d in drops)),
        "dropped_first_key_values": dict(Counter(d["annotation"][0] for d in drops).most_common()),
        "fold_sizes": [sum(r["fold"] == i for r in kept.values()) for i in range(5)],
        "cross_check_problems": problems,
    }
    LABELS.write_text("\n".join(json.dumps(kept[t]) for t in sorted(kept, key=int)) + "\n")
    DROPS.write_text(json.dumps({"summary": summary, "drops": drops}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
