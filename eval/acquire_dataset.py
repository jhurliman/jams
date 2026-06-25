#!/usr/bin/env -S uv run --extra eval
"""Acquire a labeled MIR dataset for evaluating the ableton-mcp audio-analysis service.

The service produces, per track:
  - key   : 24-class major/minor  (detect_key, librosa Krumhansl-Schmuckler)
  - bpm   : global tempo          (analyze_song_structure, All-In-One via Replicate)
  - beats / downbeats / segments  (All-In-One)

We need ground truth in the *same domain* the tool is used for: electronic / DJ
music. The GiantSteps Key dataset is the canonical fit — 600 two-minute Beatport
previews, each with an expert key label, and most also carry a Beatport tempo.
Critically (unlike GiantSteps Tempo) its audio is freely downloadable from Zenodo,
so the dataset is self-contained and the eval can actually run the MIR functions.

mirdata loads it as JAMS-native annotations; this script downloads it and flattens
it into a single newline-delimited manifest that the scorer (evaluate.py) consumes:

    {"track_id", "audio_path", "audio_exists",
     "ref_key_raw", "ref_key", "ref_tonic", "ref_mode",
     "ref_tempo", "genres", "artists", "title"}

`ref_key` is normalized to the exact spelling detect_key() emits ("C major",
"A# minor") so scoring is a string compare plus the MIREX relation table.

Examples
--------
    # Annotations only — fast, builds the manifest with labels, no large download:
    uv run eval/acquire_dataset.py --no-audio

    # Full dataset incl. ~600 audio previews (the real eval set):
    uv run eval/acquire_dataset.py

    # Quick smoke subset:
    uv run eval/acquire_dataset.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASET = "giantsteps_key"

# Spelling used by MCP_Server.audio_analysis.detect_key (sharps, no flats).
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B", "Fb": "E", "E#": "F", "B#": "C",
}
MINOR_MODES = ("minor", "aeolian", "dorian", "phrygian", "locrian")
MAJOR_MODES = ("major", "ionian", "lydian", "mixolydian")


def normalize_key(raw: str | None) -> tuple[str | None, str | None, str | None]:
    """Map a GiantSteps key label to the service's 24-class spelling.

    GiantSteps labels look like "C minor", "Db major", "F minor dorian",
    "E major ionian". Returns (full, tonic, mode) e.g. ("A# minor", "A#", "minor"),
    or (None, None, None) when the label can't be reduced to major/minor (rare:
    "modal"/silence markers).
    """
    if not raw:
        return None, None, None
    raw = raw.strip()
    low = raw.lower()
    parts = raw.split()
    if not parts:
        return None, None, None

    tonic = parts[0].strip()
    # Normalize spelling: capitalize the letter, keep accidental, map flats -> sharps.
    if tonic:
        tonic = tonic[0].upper() + tonic[1:]
    tonic = FLAT_TO_SHARP.get(tonic, tonic)
    if tonic not in NOTES:
        return None, None, None

    if any(m in low for m in MINOR_MODES):
        mode = "minor"
    elif any(m in low for m in MAJOR_MODES):
        mode = "major"
    else:
        return None, None, None

    return f"{tonic} {mode}", tonic, mode


def _safe(getter):
    try:
        return getter()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--data-home", type=Path, default=here / "data" / DATASET,
                    help="Where mirdata stores the dataset (default: eval/data/%s)" % DATASET)
    ap.add_argument("--manifest", type=Path, default=here / "data" / "manifest.jsonl",
                    help="Output manifest path (default: eval/data/manifest.jsonl)")
    ap.add_argument("--no-audio", action="store_true",
                    help="Skip the audio download; build the label-only manifest")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only emit the first N tracks (smoke testing)")
    ap.add_argument("--force", action="store_true",
                    help="Force re-download of dataset files")
    args = ap.parse_args()

    import mirdata

    args.data_home.mkdir(parents=True, exist_ok=True)
    ds = mirdata.initialize(DATASET, data_home=str(args.data_home))

    remotes = ["keys", "metadata"]
    if not args.no_audio:
        remotes = ["audio"] + remotes
    print(f"Downloading {DATASET} remotes={remotes} -> {args.data_home}", file=sys.stderr)
    ds.download(partial_download=remotes, force_overwrite=args.force, cleanup=True)

    track_ids = ds.track_ids
    if args.limit:
        track_ids = track_ids[: args.limit]

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    n_total = n_key = n_tempo = n_audio = n_skipped = 0
    with args.manifest.open("w", encoding="utf-8") as out:
        for tid in track_ids:
            tr = ds.track(tid)
            raw_key = _safe(lambda: tr.key)
            ref_key, tonic, mode = normalize_key(raw_key)
            if ref_key is None:
                n_skipped += 1
                continue

            audio_path = _safe(lambda: tr.audio_path)
            audio_exists = bool(audio_path and Path(audio_path).is_file())
            tempo = _safe(lambda: tr.tempo)
            meta = _safe(lambda: tr.genres) or {}

            rec = {
                "track_id": tid,
                "audio_path": audio_path,
                "audio_exists": audio_exists,
                "ref_key_raw": raw_key,
                "ref_key": ref_key,
                "ref_tonic": tonic,
                "ref_mode": mode,
                "ref_tempo": float(tempo) if tempo else None,
                "genres": meta.get("genres") if isinstance(meta, dict) else None,
                "artists": _safe(lambda: tr.artists),
                "title": _safe(lambda: tr.title),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_total += 1
            n_key += 1
            n_tempo += rec["ref_tempo"] is not None
            n_audio += audio_exists

    print("\n=== Manifest written ===", file=sys.stderr)
    print(f"  path          : {args.manifest}", file=sys.stderr)
    print(f"  tracks        : {n_total}", file=sys.stderr)
    print(f"  with key ref  : {n_key}", file=sys.stderr)
    print(f"  with tempo ref: {n_tempo}", file=sys.stderr)
    print(f"  audio present : {n_audio}", file=sys.stderr)
    if n_skipped:
        print(f"  skipped (unparseable key): {n_skipped}", file=sys.stderr)
    if not args.no_audio and n_audio == 0:
        print("  WARNING: audio download requested but no files found on disk.", file=sys.stderr)
    if args.no_audio:
        print("  NOTE: --no-audio set; run without it to fetch previews before scoring.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
