#!/usr/bin/env -S uv run --extra eval
"""Acquire GiantSteps-MTG-Keys — the TRAINING split for key detection.

GiantSteps **Key** (600 tracks, `acquire_dataset.py`) is our key benchmark = TEST set.
GiantSteps-**MTG**-Keys (Faraldo et al.; 1,486 Beatport previews, of which ~1,077 carry
confidence 2 = high-confidence manual keys) is the disjoint TRAINING set the literature
uses. Anything learned for key detection (the mode-refinement model, meta-classifiers)
must be fit here and evaluated ONCE on GiantSteps Key — never trained on the test set.

Audio comes from Beatport preview URLs with the JKU mirror as fallback (some previews are
dead; drops are logged, never silent). Emits ``eval/data/gsmtg/manifest.jsonl``:
  {"dataset":"gsmtg","track_id":"5061","ref_key":"D# minor","confidence":2,
   "audio_path":"...mp3","audio_exists":true}

    uv run --extra eval eval/acquire_gsmtg.py            # high-confidence (C==2) only
    uv run --extra eval eval/acquire_gsmtg.py --all-confidence --limit 50
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "gsmtg"
ANNOT_URL = (
    "https://raw.githubusercontent.com/GiantSteps/giantsteps-mtg-key-dataset/"
    "master/annotations/annotations.txt"
)
BASE_URL = "http://geo-samples.beatport.com/lofi/"
BACKUP_URL = "http://www.cp.jku.at/datasets/giantsteps/mtg_key_backup/"

NOTES = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
         "Db", "Eb", "Gb", "Ab", "Bb", "Cb", "Fb"}


def load_annotations() -> list[dict]:
    txt_path = DATA_DIR / "annotations.txt"
    if not txt_path.exists():
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "-o", str(txt_path), ANNOT_URL], check=True)
    rows = []
    for line in txt_path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        tid, key, conf = parts[0].strip(), parts[1].strip(), parts[2].strip()
        # Keep single unambiguous tonic+mode keys (drop multi-key / atonal annotations).
        kp = key.split()
        if len(kp) != 2 or kp[0] not in NOTES or kp[1] not in ("major", "minor"):
            continue
        rows.append({"track_id": tid, "ref_key": key, "confidence": int(conf or 0)})
    return rows


def download(tid: str, dest: Path) -> Path | None:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    fname = f"{tid}.LOFI.mp3"
    for base in (BASE_URL, BACKUP_URL):
        r = subprocess.run(
            ["curl", "-sL", "--fail", "--max-time", "60", "-o", str(dest), base + fname],
            capture_output=True,
        )
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 10_000:
            return dest
    dest.unlink(missing_ok=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-confidence", action="store_true",
                    help="keep every annotation (default: confidence==2 only, the "
                         "high-confidence subset the literature trains on)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "manifest.jsonl")
    args = ap.parse_args()

    rows = load_annotations()
    if not args.all_confidence:
        rows = [r for r in rows if r["confidence"] == 2]
    if args.limit:
        rows = rows[: args.limit]
    print(f"=> {len(rows)} annotated tracks to fetch", file=sys.stderr)

    audio_dir = DATA_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_rows, n_drop = [], 0
    for i, r in enumerate(rows, 1):
        p = download(r["track_id"], audio_dir / f"{r['track_id']}.LOFI.mp3")
        if p is None:
            n_drop += 1
            print(f"   [drop] {r['track_id']}: preview unavailable", file=sys.stderr)
            continue
        out_rows.append({"dataset": "gsmtg", **r, "audio_path": str(p), "audio_exists": True})
        if i % 100 == 0:
            print(f"   [{i}/{len(rows)}] fetched={len(out_rows)} dropped={n_drop}",
                  file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")
    print(f"=> Wrote {len(out_rows)} rows to {args.out}; dropped {n_drop}", file=sys.stderr)


if __name__ == "__main__":
    main()
