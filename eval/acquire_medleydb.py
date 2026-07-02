#!/usr/bin/env -S uv run --extra eval
"""Acquire MedleyDB (pitch subset) for evaluating jams monophonic pitch / melody tracking.

MedleyDB is real, professionally-recorded multitrack audio; the ``medleydb_pitch`` subset
provides per-track solo instrument stems with dense f0 (pitch) annotations — our real-audio
counterpart to the synthesized Slakh set for pitch-tracking evaluation.

**The audio is GATED**: mirdata can auto-download the indexes and pitch annotations, but NOT
the audio itself (you must request access and place the wavs where mirdata expects them,
under ``--data-home``). This script therefore keeps only tracks that have BOTH a local audio
file AND a pitch annotation, logs a drop count for the rest, and — if ZERO tracks have local
audio — prints instructions and exits non-zero rather than writing an empty manifest.

Row schema (one JSON object per line):
    {"dataset":"medleydb","format":"medleydb","track_id":"MusicDelta_Rock",
     "instrument":"electric bass","audio_path":"/abs/audio.wav",
     "pitch_annotation_path":"/abs/pitch.csv","audio_exists":true}

Examples
--------
    # Build the manifest from a local MedleyDB_Pitch install:
    uv run --extra eval eval/acquire_medleydb.py --data-home /data/medleydb_pitch

    # First 10 tracks only:
    uv run --extra eval eval/acquire_medleydb.py --data-home /data/medleydb_pitch --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mirdata

DATA_DIR = Path(__file__).resolve().parent / "data" / "medleydb"

_HELP = """\
No MedleyDB_Pitch audio was found on disk. MedleyDB audio is GATED — mirdata downloads only
the pitch annotations, not the wavs. Request access and place the audio where mirdata expects
it, then re-run with --data-home:

  1. Register / request access at https://medleydb.weebly.com/ (Zenodo: MedleyDB_Pitch).
  2. Extract the archive so that <data-home>/audio/<track_id>.wav exists.
  3. uv run --extra eval eval/acquire_medleydb.py --data-home /path/to/medleydb_pitch
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-home", type=Path, default=None,
                    help="path to the local MedleyDB_Pitch install, passed to mirdata.initialize")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "manifest.jsonl")
    args = ap.parse_args()

    data_home = str(args.data_home) if args.data_home else None
    dataset = mirdata.initialize("medleydb_pitch", data_home=data_home)

    # The index + pitch annotations are small; fetch only those if the index is missing
    # (this never pulls the gated audio, which has no auto-download remote).
    if not Path(dataset.index_path).exists():
        print("=> Fetching MedleyDB_Pitch index + annotations (small; no audio) ...",
              file=sys.stderr)
        try:
            dataset.download()
        except Exception as exc:  # noqa: BLE001 - offline / gated audio remote unavailable
            print(f"=> Note: mirdata download returned: {exc}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_no_audio, n_no_pitch = 0, 0
    for track_id in dataset.track_ids:
        track = dataset.track(track_id)
        audio_path = track.audio_path
        pitch_path = track.pitch_path
        has_audio = bool(audio_path and Path(audio_path).exists())
        has_pitch = bool(pitch_path and Path(pitch_path).exists())
        if not has_pitch:
            n_no_pitch += 1
            print(f"   [drop] {track_id}: no pitch annotation on disk", file=sys.stderr)
            continue
        if not has_audio:
            n_no_audio += 1
            print(f"   [drop] {track_id}: audio missing (gated — must be obtained manually)",
                  file=sys.stderr)
            continue
        rows.append({
            "dataset": "medleydb",
            "format": "medleydb",
            "track_id": track_id,
            "instrument": track.instrument,
            "audio_path": str(audio_path),
            "pitch_annotation_path": str(pitch_path),
            "audio_exists": True,
        })
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print(f"=> Dropped {n_no_audio} tracks with no local audio, "
              f"{n_no_pitch} with no pitch annotation.", file=sys.stderr)
        print(_HELP, file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"=> Dropped {n_no_audio} tracks (missing gated audio), "
          f"{n_no_pitch} (missing pitch annotation).", file=sys.stderr)
    print(f"=> Wrote {len(rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
