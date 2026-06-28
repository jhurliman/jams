#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
"""Prepare Raveform as a Harmonix-shaped training set for the published All-In-One v1 trainer.

Why this exists: the v1 trainer (`mir-aidj/all-in-one`, `src/allin1/training/`) only ships a
Harmonix DataModule. The *artifact* it produces — a single-dataset, N-class checkpoint in the v1
`AllInOne` layout — is exactly what `structure_worker.py` already loads (it remaps `raveform-fold3`
as a no-op). So training a v1-architecture Raveform model gives a drop-in EDM checkpoint without
needing the unpublished v2 multi-dataset code.

This script converts our local Raveform data into a clean, standard on-disk layout the trainer can
consume after minimal patching (num_labels 10->11, true fold from metadata not index%8, oversampling
+ tempo augmentation — see eval/TRAINING.md). It does NOT train and needs no GPU.

Inputs (all already on disk):
  - segments.json : per-track {key, fold, genre, average_bpm, tempos, sections:[{name,start,end}]}
  - beat CSVs     : per-track `time,downbeat,section` (downbeat==1 marks bar starts)
  - audio         : <audio_dir>/<track_id>.m4a

Outputs (under --out, default data/raveform_train/):
  metadata.csv            File,BPM,fold,genre   (File = track_id; fold = TRUE fold, not index%8)
  beats/<track_id>.txt    one line per beat: "<time>\t<1|0>"   (1 = downbeat)
  segments/<track_id>.txt one line per section: "<start>\t<end>\t<label>"  (11-class raveform vocab)
  transcode.sh            ffmpeg commands to make tracks/<track_id>.mp3 (the trainer globs *.mp3)
  labels.txt              the 11-class label order (matches structure_worker._RAVEFORM_LABELS)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Must match structure_worker._RAVEFORM_LABELS (the trained classifier's index order).
RAVEFORM_LABELS = [
    "start", "end", "altintro", "altoutro", "intro", "outro",
    "breakdown", "buildup", "cooldown", "bridge", "drop",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segments", type=Path,
                    default=Path("eval/data/raveform/segments.json"),
                    help="segments.json with folds/genre/sections "
                         "(unzip -p raveform.zip raveform/structures/segments.json > this)")
    ap.add_argument("--audio-dir", type=Path, default=Path("eval/data/raveform/audio"))
    ap.add_argument("--beats-dir", type=Path,
                    default=Path("eval/data/raveform/raveform/structures/beats"),
                    help="dir of per-track <track_id>.beat.csv files")
    ap.add_argument("--out", type=Path, default=Path("data/raveform_train"))
    args = ap.parse_args()

    segs = json.loads(args.segments.read_text())
    by_key = {e["key"]: e for e in segs}

    out = args.out
    (out / "beats").mkdir(parents=True, exist_ok=True)
    (out / "segments").mkdir(parents=True, exist_ok=True)
    (out / "labels.txt").write_text("\n".join(RAVEFORM_LABELS) + "\n")

    meta_rows: list[tuple[str, int, int, str]] = []
    transcode: list[str] = []
    n_ok = n_skip = 0
    seen_labels: set[str] = set()

    for key, e in by_key.items():
        audio = args.audio_dir / f"{key}.m4a"
        beat_csv = _find_beat_csv(args.beats_dir, key)
        if not audio.exists() or beat_csv is None:
            n_skip += 1
            continue

        # beats: time + downbeat flag from the beat CSV
        beat_lines = []
        for r in csv.DictReader(beat_csv.read_text().splitlines()):
            beat_lines.append(f"{float(r['time']):.4f}\t{1 if r['downbeat'] == '1' else 0}")
        (out / "beats" / f"{key}.txt").write_text("\n".join(beat_lines) + "\n")

        # segments: canonical sections (11-class), preserving same-label phrase splits
        seg_lines = []
        for s in e["sections"]:
            seg_lines.append(f"{s['start']:.4f}\t{s['end']:.4f}\t{s['name']}")
            seen_labels.add(s["name"])
        (out / "segments" / f"{key}.txt").write_text("\n".join(seg_lines) + "\n")

        bpm = int(round(e.get("average_bpm") or 0))
        meta_rows.append((key, bpm, int(e["fold"]), e.get("genre", "")))
        transcode.append(
            f'ffmpeg -y -v error -i "{audio.resolve()}" -ar 44100 -ac 2 '
            f'"{(out / "tracks" / (key + ".mp3")).resolve()}"')
        n_ok += 1

    with (out / "metadata.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["File", "BPM", "fold", "genre"])
        w.writerows(meta_rows)

    (out / "transcode.sh").write_text(
        "#!/bin/sh\nset -e\nmkdir -p '" + str((out / "tracks").resolve()) + "'\n"
        + "\n".join(transcode) + "\n")
    (out / "transcode.sh").chmod(0o755)

    unknown = seen_labels - set(RAVEFORM_LABELS)
    print(f"wrote {n_ok} tracks ({n_skip} skipped for missing audio/beats) to {out}")
    print(f"label vocab ({len(RAVEFORM_LABELS)}): {RAVEFORM_LABELS}")
    if unknown:
        print(f"WARNING: section labels not in vocab: {sorted(unknown)}")
    print(f"next: sh {out}/transcode.sh   # m4a -> mp3 (the trainer globs tracks/*.mp3)")
    print("then follow eval/TRAINING.md")


def _find_beat_csv(beats_dir: Path, key: str) -> Path | None:
    for cand in (beats_dir / f"{key}.beat.csv", beats_dir / f"{key}.csv"):
        if cand.exists():
            return cand
    return None


if __name__ == "__main__":
    main()
