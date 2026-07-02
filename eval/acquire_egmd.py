#!/usr/bin/env -S uv run --extra eval
"""Acquire the Expanded Groove MIDI Dataset (E-GMD) for drum-transcription eval.

E-GMD (Magenta) is 444 h of human drum performances recorded on a Roland TD-17, annotated in
GM-percussion MIDI aligned to the audio — the reference set for the OaF-drums transcriber.

The full audio is ~100 GB and Magenta serves it as a single zip, so a partial HTTP download of
arbitrary tracks is not generally possible. Two supported paths:

  1. --data-home DIR : point at an already-extracted E-GMD dir (contains ``e-gmd-v1.0.0.csv``
                       and the drummer*/session* wav+midi tree). No download. Recommended.
  2. (default)       : best-effort fetch of the info CSV + the first ``--limit`` per-file
                       wav/midi pairs directly from the GCS bucket. If individual-file access
                       is unavailable (404), we print instructions to grab the full/`--data-home`
                       route and exit — never a multi-GB blind download.

Emits ``eval/data/egmd/manifest.jsonl``; each row:
  {"dataset":"egmd","format":"egmd","track_id":"...","drum_audio_path":"...",
   "drum_midi_path":"...","style":"funk/groove1","bpm":120,"split":"test","audio_exists":true}

    uv run --extra eval eval/acquire_egmd.py --data-home /data/e-gmd-v1.0.0 --split test --limit 100
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "egmd"
BUCKET = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0"
CSV_NAME = "e-gmd-v1.0.0.csv"


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed magenta bucket
        return dest.exists() and dest.stat().st_size > 0
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return False


def _load_csv(data_home: Path | None, audio_dir: Path) -> tuple[list[dict], Path]:
    """Return (rows, base_dir) where base_dir is where wav/midi paths are rooted."""
    if data_home:
        csv_path = data_home / CSV_NAME
        if not csv_path.exists():
            sys.exit(f"No {CSV_NAME} under --data-home {data_home}. Not an extracted E-GMD dir?")
        rows = list(csvmod.DictReader(csv_path.read_text().splitlines()))
        return rows, data_home
    # Fetch the info CSV individually from the bucket.
    csv_path = audio_dir / CSV_NAME
    if not _download(f"{BUCKET}/{CSV_NAME}", csv_path):
        sys.exit(
            "Could not fetch the E-GMD info CSV individually. Download the dataset from\n"
            f"  {BUCKET}/e-gmd-v1.0.0.zip\n"
            "extract it, and re-run with --data-home <extracted-dir>."
        )
    rows = list(csvmod.DictReader(csv_path.read_text().splitlines()))
    return rows, audio_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-home", type=Path, default=None,
                    help="Path to an already-extracted E-GMD dir (recommended; no download)")
    ap.add_argument("--split", default="test", help="E-GMD split to keep (train/validation/test)")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "manifest.jsonl")
    args = ap.parse_args()

    audio_dir = DATA_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows, base = _load_csv(args.data_home, audio_dir)
    eligible = [r for r in rows if not args.split or r.get("split") == args.split]
    eligible = [r for r in eligible if r.get("audio_filename") and r.get("midi_filename")]
    if args.limit:
        eligible = eligible[: args.limit]
    if not eligible:
        sys.exit(f"No E-GMD rows for split={args.split!r}.")

    out_rows, n_ok, n_drop = [], 0, 0
    for r in eligible:
        af, mf = r["audio_filename"], r["midi_filename"]
        if args.data_home:
            apath, mpath = base / af, base / mf
            ok = apath.exists() and mpath.exists()
        else:
            apath, mpath = audio_dir / af, audio_dir / mf
            ok = _download(f"{BUCKET}/{af}", apath) and _download(f"{BUCKET}/{mf}", mpath)
        if not ok:
            n_drop += 1
            print(f"   [drop] {r.get('id', af)}: audio/midi unavailable", file=sys.stderr)
            if not args.data_home and n_drop == 1:
                print(
                    "   (individual-file download may be unavailable; if drops persist, grab "
                    f"{BUCKET}/e-gmd-v1.0.0.zip and use --data-home)", file=sys.stderr,
                )
            continue
        n_ok += 1
        out_rows.append({
            "dataset": "egmd", "format": "egmd",
            "track_id": str(r.get("id") or Path(af).stem),
            "drum_audio_path": str(apath), "drum_midi_path": str(mpath),
            "style": r.get("style"), "bpm": float(r["bpm"]) if r.get("bpm") else None,
            "split": r.get("split"), "audio_exists": True,
        })

    with open(args.out, "w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")
    print(f"=> Wrote {len(out_rows)} rows to {args.out}", file=sys.stderr)
    print(f"=> Prepared {n_ok} tracks; dropped {n_drop} (unavailable)", file=sys.stderr)


if __name__ == "__main__":
    main()
