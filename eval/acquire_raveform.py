#!/usr/bin/env -S uv run --extra eval
"""Acquire the Raveform dataset for evaluating jams song-structure analysis.

Raveform (Kim et al., TISMIR 2026 — the All-In-One authors) is our **primary** structure
benchmark: 1,423 EDM tracks with expert beats / downbeats / functional-segment annotations
using a DJ, energy-centered vocabulary (intro, buildup, drop, breakdown, cooldown, outro,
…). Annotations (MIT-licensed) come in the dataset zip; audio is sourced per track from its
YouTube id. Unlike Harmonix, the annotations were made on those same YouTube videos, so no
master-mismatch alignment step is needed.

This builds a common-schema manifest the shared evaluator consumes. Each annotated track is
``structures/beats/<idx>.<ytid>.beat.csv`` with columns ``time, downbeat (1-4 bar position),
section`` — beats, downbeats (``downbeat==1``) and functional segments (runs of equal
``section``) all in one. The model used for scoring is ``harmonix-all``: Raveform is an
out-of-domain test set for our Harmonix-trained All-In-One, so there is no fold/CV leakage —
the numbers read directly as "how well the current model generalizes to EDM".

Examples
--------
    # Unzip annotations + build the manifest (no audio yet):
    uv run --extra eval eval/acquire_raveform.py --no-audio

    # Download a small sample to validate end-to-end:
    uv run --extra eval eval/acquire_raveform.py --limit 8

    # Full set (large, slow YouTube pull — better via a background job):
    uv run --extra eval eval/acquire_raveform.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "raveform"
ZIP_PATH = DATA_DIR / "raveform.zip"
ZIP_URL = "https://huggingface.co/datasets/taejunkim/raveform/resolve/main/raveform.zip"
BEATS_DIR = DATA_DIR / "raveform" / "structures" / "beats"
SEGMENTS_JSON = DATA_DIR / "raveform" / "structures" / "segments.json"


def ensure_annotations() -> Path:
    """Make sure the beat CSVs and segments.json are extracted; download the zip if missing."""
    if BEATS_DIR.exists() and any(BEATS_DIR.glob("*.csv")) and SEGMENTS_JSON.exists():
        return BEATS_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"=> Downloading Raveform annotations zip (~479 MB) from {ZIP_URL}", file=sys.stderr)
        subprocess.run(["curl", "-sL", ZIP_URL, "-o", str(ZIP_PATH)], check=True)
    print("=> Extracting structures/beats/ + segments.json ...", file=sys.stderr)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = [
            n for n in zf.namelist()
            if (("/structures/beats/" in n and n.endswith(".csv")) or n.endswith("/segments.json"))
        ]
        zf.extractall(DATA_DIR, members=members)
    return BEATS_DIR


def load_segments() -> dict[str, dict]:
    """segments.json keyed by track id — carries the canonical sections, fold, genre, and BPM
    that the evaluator needs (held-out CV via `fold`, `--target genre`, canonical segment refs)."""
    if not SEGMENTS_JSON.exists():
        return {}
    return {e["key"]: e for e in json.loads(SEGMENTS_JSON.read_text())}


def parse_annotation(csv_path: Path) -> dict:
    """Beats, downbeats, segments, and a derived BPM from a Raveform beat CSV."""
    beats: list[float] = []
    downbeats: list[float] = []
    seg_times: list[float] = []
    seg_labels: list[str] = []
    prev_section = None
    for row in csv.DictReader(csv_path.open()):
        t = float(row["time"])
        beats.append(t)
        if row["downbeat"] == "1":
            downbeats.append(t)
        section = row["section"]
        if section != prev_section:  # new functional segment starts here
            seg_times.append(t)
            seg_labels.append(section)
            prev_section = section
    # Segments = [start, end) per run; last runs to the final beat.
    end = beats[-1] if beats else 0.0
    segments = [
        {"start": seg_times[i], "end": (seg_times[i + 1] if i + 1 < len(seg_times) else end),
         "label": seg_labels[i]}
        for i in range(len(seg_times))
    ]
    bpm = None
    if len(beats) >= 2:
        diffs = [b - a for a, b in zip(beats, beats[1:], strict=False) if b > a]
        if diffs:
            bpm = round(60.0 / st.median(diffs), 2)
    return {"beats": beats, "downbeats": downbeats, "segments": segments,
            "bpm_ref": bpm, "duration_ref": round(end, 3)}


def download_audio(ytid: str, dest_stem: Path) -> Path | None:
    # NB: track_id contains dots, so build the name by appending (with_suffix would
    # truncate at the last dot, e.g. "0002.kfJQCu-Jbec" -> "0002.m4a").
    target = dest_stem.parent / (dest_stem.name + ".m4a")
    if target.exists():
        return target
    url = f"https://www.youtube.com/watch?v={ytid}"
    cmd = ["yt-dlp", "-q", "--no-warnings", "-f", "bestaudio/best",
           "--extract-audio", "--audio-format", "m4a",
           "-o", str(dest_stem) + ".%(ext)s", url]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    return target if target.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-audio", action="store_true", help="build manifest only; skip downloads")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "manifest.jsonl")
    args = ap.parse_args()

    beats_dir = ensure_annotations()
    segs = load_segments()
    csvs = sorted(beats_dir.glob("*.csv"))
    if args.limit:
        csvs = csvs[: args.limit]
    print(f"=> {len(csvs)} annotated Raveform tracks", file=sys.stderr)

    audio_dir = DATA_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows, n_audio, n_dropped = [], 0, 0
    for csv_path in csvs:
        track_id = csv_path.name[: -len(".beat.csv")]  # "<idx>.<ytid>"
        ytid = track_id.split(".", 1)[1]
        ann = parse_annotation(csv_path)
        seg = segs.get(track_id)
        fold = seg.get("fold") if seg else None
        audio_path = None
        if not args.no_audio:
            audio_path = download_audio(ytid, audio_dir / track_id)
            if audio_path is None:
                n_dropped += 1
                print(f"   [drop] {track_id}: YouTube download failed", file=sys.stderr)
                continue
            n_audio += 1
        rows.append({
            "dataset": "raveform",
            "format": "raveform",
            "track_id": track_id,
            "ytid": ytid,
            # Honest 8-fold CV: score each track with its held-out fold model (see eval/README).
            "model": f"all-fold{fold}" if fold is not None else "all-all",
            "fold": fold,
            "genre": seg.get("genre") if seg else None,
            "bpm_ref": (seg.get("average_bpm") if seg else None) or ann["bpm_ref"],
            "duration_ref": ann["duration_ref"],
            # Canonical functional sections (preserve same-label phrase boundaries the beat-CSV
            # section column merges); the evaluator scores segments against these.
            "sections": seg.get("sections") if seg else None,
            "beats_csv": str(csv_path),
            "audio_path": str(audio_path) if audio_path else None,
            "audio_exists": audio_path is not None,
        })

    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"=> Wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    if not args.no_audio:
        print(f"=> Downloaded {n_audio} audio files; dropped {n_dropped} (dead/blocked URLs)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
