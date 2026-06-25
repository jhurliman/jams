#!/usr/bin/env -S uv run --extra eval
"""Evaluate the jams service against the labeled GiantSteps dataset.

Runs the production `jams.detect_key` / `jams.detect_tempo` over every track with audio
present and reports:

  key  : MIREX-weighted score (correct 1.0, fifth 0.5, relative 0.3, parallel 0.2)
         + exact accuracy.
  tempo: Acc1 (within tol) and Acc2 (octave-tolerant). Wrong labels are overridden from
         tempo_corrections.csv (see build_corrections.py); genre hints drive octave
         resolution.

Run from the repo root:
    uv run --extra eval eval/evaluate.py
    uv run --extra eval eval/evaluate.py --limit 25 --out eval/data/results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from jams import detect_key, detect_tempo

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH = {n: i for i, n in enumerate(NOTES)}


def parse_key(key: str):
    if not key:
        return None
    parts = key.split()
    if len(parts) != 2 or parts[0] not in PITCH:
        return None
    return PITCH[parts[0]], parts[1]


def mirex_score(ref: str, pred: str) -> float:
    r, p = parse_key(ref), parse_key(pred)
    if r is None or p is None:
        return 0.0
    r_pc, r_mode = r
    p_pc, p_mode = p
    interval = (p_pc - r_pc) % 12
    if interval == 0 and p_mode == r_mode:
        return 1.0
    if p_mode == r_mode and interval in (7, 5):
        return 0.5
    if r_mode == "major" and p_mode == "minor" and interval == 9:
        return 0.3
    if r_mode == "minor" and p_mode == "major" and interval == 3:
        return 0.3
    if interval == 0 and p_mode != r_mode:
        return 0.2
    return 0.0


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=here / "data" / "manifest.jsonl")
    ap.add_argument("--corrections", type=Path, default=here / "tempo_corrections.csv",
                    help="CSV of curated tempo-label fixes (track_id,corrected_tempo); overrides ref_tempo")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tempo-tol", type=float, default=0.04, help="Acc1 tolerance fraction (default 4%%)")
    ap.add_argument("--no-tempo", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}\nRun: uv run --extra eval eval/acquire_dataset.py", file=sys.stderr)
        return 2

    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    records = [r for r in records if r.get("audio_exists")]
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("No tracks with audio. Run acquire_dataset.py without --no-audio first.", file=sys.stderr)
        return 2

    corrections: dict[str, float] = {}
    if args.corrections.is_file():
        for row in csv.DictReader(args.corrections.open()):
            try:
                corrections[row["track_id"]] = float(row["corrected_tempo"])
            except (KeyError, ValueError):
                pass
        if corrections:
            print(f"Applying {len(corrections)} tempo label corrections from {args.corrections.name}", file=sys.stderr)

    from tqdm import tqdm

    results, key_weights, exact = [], [], []
    t_acc1 = t_acc2 = t_n = 0
    key_methods: dict = {}
    tempo_methods: dict = {}

    for r in tqdm(records, desc="scoring", unit="trk"):
        row = {"track_id": r["track_id"], "ref_key": r["ref_key"], "ref_tempo": r.get("ref_tempo")}
        try:
            k = detect_key(r["audio_path"])
            w = mirex_score(r["ref_key"], k["key"])
            row.update(pred_key=k["key"], key_weight=w, key_method=k.get("method"))
            key_methods[k.get("method")] = key_methods.get(k.get("method"), 0) + 1
            key_weights.append(w)
            exact.append(1.0 if w == 1.0 else 0.0)
        except Exception as e:
            row.update(pred_key=None, key_weight=0.0, error=str(e))
            key_weights.append(0.0)
            exact.append(0.0)

        genres = r.get("genres") or [None]
        if not args.no_tempo and r.get("ref_tempo"):
            try:
                t = detect_tempo(r["audio_path"], genre=genres[0] if genres else None)
                bpm = float(t["bpm"])
                ref = float(corrections.get(r["track_id"], r["ref_tempo"]))
                row.update(pred_tempo=round(bpm, 2), tempo_method=t.get("method"))
                tempo_methods[t.get("method")] = tempo_methods.get(t.get("method"), 0) + 1
                t_n += 1
                if abs(bpm - ref) <= args.tempo_tol * ref:
                    t_acc1 += 1
                    t_acc2 += 1
                elif any(abs(bpm * f - ref) <= args.tempo_tol * ref for f in (0.5, 2.0, 1 / 3, 3.0)):
                    t_acc2 += 1
            except Exception as e:
                row.setdefault("error", str(e))
        results.append(row)

    n = len(key_weights)
    print("\n=== Key detection (jams.detect_key) ===")
    print(f"  tracks         : {n}")
    print(f"  MIREX score    : {sum(key_weights)/n:.4f}")
    print(f"  exact accuracy : {sum(exact)/n:.4f}")
    print(f"  methods used   : {key_methods}")
    if not args.no_tempo and t_n:
        print("\n=== Tempo (jams.detect_tempo) ===")
        print(f"  tracks scored  : {t_n}")
        print(f"  Acc1 (±{args.tempo_tol:.0%})   : {t_acc1/t_n:.4f}")
        print(f"  Acc2 (octave)  : {t_acc2/t_n:.4f}")
        print(f"  methods used   : {tempo_methods}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nPer-track results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
