#!/usr/bin/env -S uv run --extra eval
"""Acquire the Harmonix Set for evaluating jams song-structure analysis.

The Harmonix Set (Nieto et al., ISMIR 2019) is the standard benchmark for beats,
downbeats, and functional structure — and the set All-In-One was trained on. Its
annotations are public (github.com/urinieto/harmonixset) but the **audio is not**
(copyright), so we source each track from its YouTube URL.

Two correctness points this script enforces:

1. **Per-fold cross-validation.** All-In-One ships 8 fold models. Track *i* (in the
   sorted track-id list) was held out of model ``harmonix-fold{i % 8}`` — that is the
   only model allowed to score it honestly. We reproduce that exact positional split
   (``allin1.training ... HarmonixDataset``: ``folds = arange(n) % total_folds``) and
   record each track's fold so evaluate_structure.py uses the right model.

2. **Alignment.** Annotations were made on Harmonix's own audio; a YouTube upload can
   be a different master/edit and drift out of sync. The dataset ships per-track
   alignment scores — we keep only tracks at/above ``--min-alignment`` (default 0.95)
   so the timing ground truth is valid for the audio we actually downloaded.

Writes ``eval/data/harmonix/manifest.jsonl`` (one row per usable track) and audio to
``eval/data/harmonix/audio/``. Both live under the gitignored ``eval/data/``.

Examples
--------
    # Annotations + fold map only (no download) — fast, builds a manifest skeleton:
    uv run --extra eval eval/acquire_harmonix.py --no-audio

    # Download a balanced sample (one per fold) to validate the pipeline:
    uv run --extra eval eval/acquire_harmonix.py --limit 8 --balanced

    # Full set (well-aligned tracks only; this is a large, slow YouTube pull):
    uv run --extra eval eval/acquire_harmonix.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HARMONIX_REPO = "https://github.com/urinieto/harmonixset"
TOTAL_FOLDS = 8
DATA_DIR = Path(__file__).resolve().parent / "data" / "harmonix"
ANNOT_DIR = DATA_DIR / "harmonixset"  # cloned annotations repo


def ensure_annotations() -> Path:
    """Clone the harmonixset annotations repo into eval/data if absent."""
    if (ANNOT_DIR / "dataset" / "metadata.csv").exists():
        return ANNOT_DIR / "dataset"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=> Cloning {HARMONIX_REPO} (annotations only) ...", file=sys.stderr)
    subprocess.run(
        ["git", "clone", "--depth", "1", HARMONIX_REPO, str(ANNOT_DIR)], check=True
    )
    return ANNOT_DIR / "dataset"


def fold_map(dataset_dir: Path) -> dict[str, int]:
    """Reproduce All-In-One's positional 8-fold split: fold(track_i) = i % TOTAL_FOLDS.

    Order is the sorted list of track stems (== annotation filenames == mp3 stems used
    in training).
    """
    stems = sorted(p.stem for p in (dataset_dir / "segments").glob("*.txt"))
    return {stem: i % TOTAL_FOLDS for i, stem in enumerate(stems)}


def load_csv_map(path: Path, key: str, val: str) -> dict[str, str]:
    with open(path, newline="") as fh:
        return {row[key]: row[val] for row in csv.DictReader(fh)}


def download_audio(url: str, dest_stem: Path) -> Path | None:
    """Download + extract audio to dest_stem.m4a via yt-dlp. Returns path or None.

    YouTube's bestaudio is usually .webm/.opus, which jams won't decode; yt-dlp's
    ffmpeg extractor re-muxes to m4a (small, universally supported). Idempotent.
    """
    target = dest_stem.with_suffix(".m4a")
    if target.exists():
        return target
    cmd = [
        "yt-dlp", "-q", "--no-warnings", "-f", "bestaudio/best",
        "--extract-audio", "--audio-format", "m4a",
        "-o", str(dest_stem) + ".%(ext)s", url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    return target if target.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-alignment", type=float, default=0.95,
                    help="drop tracks whose YouTube audio aligns below this (default 0.95)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of tracks")
    ap.add_argument("--balanced", action="store_true",
                    help="with --limit, take an even spread across the 8 folds")
    ap.add_argument("--no-audio", action="store_true", help="build manifest only; skip downloads")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "manifest.jsonl")
    args = ap.parse_args()

    dataset = ensure_annotations()
    folds = fold_map(dataset)
    alignment = {k: float(v) for k, v in
                 load_csv_map(dataset / "youtube_alignment_scores.csv", "File", "score").items()}
    urls = load_csv_map(dataset / "youtube_urls.csv", "File", "URL")
    bpm_ref = load_csv_map(dataset / "metadata.csv", "File", "BPM")

    # Eligible = annotated + well-aligned + has a URL.
    eligible = sorted(
        stem for stem in folds
        if alignment.get(stem, 0.0) >= args.min_alignment and stem in urls
    )
    print(f"=> {len(folds)} annotated, {len(eligible)} with alignment >= {args.min_alignment} "
          f"and a YouTube URL", file=sys.stderr)

    if args.limit:
        if args.balanced:
            by_fold: dict[int, list[str]] = {}
            for stem in eligible:
                by_fold.setdefault(folds[stem], []).append(stem)
            picked, f = [], 0
            while len(picked) < args.limit and any(by_fold.values()):
                bucket = by_fold.get(f % TOTAL_FOLDS)
                if bucket:
                    picked.append(bucket.pop(0))
                f += 1
            eligible = sorted(picked)
        else:
            eligible = eligible[: args.limit]

    audio_dir = DATA_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows, n_audio, n_dropped = [], 0, 0
    for stem in eligible:
        audio_path = None
        if not args.no_audio:
            audio_path = download_audio(urls[stem], audio_dir / stem)
            if audio_path is None:
                n_dropped += 1
                print(f"   [drop] {stem}: YouTube download failed", file=sys.stderr)
                continue
            n_audio += 1
        try:
            ref_bpm = float(bpm_ref.get(stem, "")) or None
        except ValueError:
            ref_bpm = None
        rows.append({
            "file": stem,
            "fold": folds[stem],
            "model": f"harmonix-fold{folds[stem]}",
            "alignment": round(alignment[stem], 4),
            "bpm_ref": ref_bpm,
            "youtube_url": urls[stem],
            "segments_path": str(dataset / "segments" / f"{stem}.txt"),
            "beats_path": str(dataset / "beats_and_downbeats" / f"{stem}.txt"),
            "audio_path": str(audio_path) if audio_path else None,
            "audio_exists": audio_path is not None,
        })

    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(f"=> Wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    if not args.no_audio:
        print(f"=> Downloaded {n_audio} audio files; dropped {n_dropped} (dead/unavailable URLs)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
