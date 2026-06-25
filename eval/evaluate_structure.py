#!/usr/bin/env -S uv run --extra eval
"""Evaluate jams song-structure analysis on the Harmonix Set (per-fold CV).

For every track in the manifest (built by acquire_harmonix.py) this runs the
production ``jams.analysis.structure.analyze_structure`` with the track's held-out
fold model (``harmonix-fold{i % 8}``) — honest cross-validation, no train-set leakage
— and scores against the Harmonix annotations with mir_eval:

  beats / downbeats : F-measure (mir_eval.beat, 70 ms window)
  segment boundaries: Hit-Rate F @ 0.5 s and @ 3 s (mir_eval.segment.detection)
  segment labeling  : pairwise F + V-measure (mir_eval.segment)

``--target`` controls the beat-tracking BPM constraint, to quantify how much it buys:
  jams  (default) : jams' own detect_tempo feeds target_bpm  (what production does)
  ref             : Harmonix reference BPM as target_bpm      (oracle upper bound)
  none            : no constraint — raw All-In-One            (the ablation baseline)

Run from the repo root (structure runs locally via the uv worker):
    uv run --extra eval eval/evaluate_structure.py
    uv run --extra eval eval/evaluate_structure.py --target none   # ablation
    uv run --extra eval eval/evaluate_structure.py --limit 8 --out eval/data/structure_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import mir_eval
import numpy as np

from jams.analysis.structure import analyze_structure

MANIFEST = Path(__file__).resolve().parent / "data" / "harmonix" / "manifest.jsonl"


def load_beats(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (beats, downbeats) from a Harmonix beats_and_downbeats file."""
    beats, downbeats = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        t = float(parts[0])
        beats.append(t)
        if parts[1] == "1":  # beat position within the bar == 1 -> downbeat
            downbeats.append(t)
    return np.array(beats), np.array(downbeats)


def load_segments(path: str) -> tuple[np.ndarray, list[str]]:
    """Return (intervals Nx2, labels) from a Harmonix segments file (boundary+label)."""
    times, labels = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        times.append(float(parts[0]))
        labels.append(parts[1])
    return _boundaries_to_intervals(times, labels)


def _boundaries_to_intervals(times, labels) -> tuple[np.ndarray, list[str]]:
    intervals, lab = [], []
    for i in range(len(times) - 1):
        intervals.append([times[i], times[i + 1]])
        lab.append(labels[i])
    return np.array(intervals), lab


def est_segments(structure: dict) -> tuple[np.ndarray, list[str]]:
    segs = structure.get("segments") or []
    intervals = np.array([[s["start"], s["end"]] for s in segs]) if segs else np.empty((0, 2))
    return intervals, [s["label"] for s in segs]


def estimate_offset(ref_beats: np.ndarray, est_beats: np.ndarray,
                    search: float = 12.0, step: float = 0.05) -> float:
    """Global lag (seconds) to add to estimated times to best match the reference.

    Harmonix annotations were made on Harmonix's masters; YouTube uploads start at a
    different point, so audio-derived events carry a constant per-track offset. We
    recover it by maximizing beat F over a grid (≈ cross-correlating the beat impulse
    trains) and apply the SAME offset to downbeats and segments.
    """
    if ref_beats.size < 2 or est_beats.size < 2:
        return 0.0
    offsets = np.arange(-search, search + step, step)
    best_off, best_f = 0.0, -1.0
    for off in offsets:
        f = mir_eval.beat.f_measure(ref_beats, est_beats + off)
        if f > best_f:
            best_f, best_off = f, float(off)
    return best_off


def shift_structure(structure: dict, offset: float) -> dict:
    """Return a copy of the structure with all times shifted by ``offset`` seconds."""
    if offset == 0.0:
        return structure
    return {
        **structure,
        "beats": [b + offset for b in (structure.get("beats") or [])],
        "downbeats": [d + offset for d in (structure.get("downbeats") or [])],
        "segments": [{**s, "start": s["start"] + offset, "end": s["end"] + offset}
                     for s in (structure.get("segments") or [])],
    }


def score_track(ref_beats, ref_down, ref_seg_int, ref_seg_lab, structure) -> dict:
    est_beats = np.array(structure.get("beats") or [])
    est_down = np.array(structure.get("downbeats") or [])
    est_int, est_lab = est_segments(structure)

    out = {
        "beat_f": mir_eval.beat.f_measure(ref_beats, est_beats) if est_beats.size else 0.0,
        "downbeat_f": mir_eval.beat.f_measure(ref_down, est_down) if est_down.size else 0.0,
    }
    if est_int.size and ref_seg_int.size:
        # Align both to a common [0, t_max] span for boundary + label metrics.
        t_max = float(max(ref_seg_int[-1, 1], est_int[-1, 1]))
        adjust = mir_eval.util.adjust_intervals
        ref_i, ref_l = adjust(ref_seg_int, ref_seg_lab, t_min=0.0, t_max=t_max)
        est_i, est_l = adjust(est_int, est_lab, t_min=0.0, t_max=t_max)
        out["bound_f_0.5"] = mir_eval.segment.detection(ref_i, est_i, window=0.5, trim=True)[2]
        out["bound_f_3.0"] = mir_eval.segment.detection(ref_i, est_i, window=3.0, trim=True)[2]
        out["pairwise_f"] = mir_eval.segment.pairwise(ref_i, ref_l, est_i, est_l)[2]
        # V-measure: harmonic mean of over- and under-segmentation NCE scores.
        over, under, _ = mir_eval.segment.nce(ref_i, ref_l, est_i, est_l)
        out["v_measure"] = 0.0 if (over + under) == 0 else 2 * over * under / (over + under)
    else:
        out.update({"bound_f_0.5": 0.0, "bound_f_3.0": 0.0, "pairwise_f": 0.0, "v_measure": 0.0})
    return out


def resolve_target(mode: str, audio: str, bpm_ref: float | None) -> float | None:
    if mode == "none":
        return None
    if mode == "ref":
        return bpm_ref
    from jams.analysis.tempo import detect_tempo  # jams' production tempo

    return detect_tempo(audio)["bpm"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--target", choices=["jams", "ref", "none"], default="jams")
    ap.add_argument("--align", action="store_true",
                    help="best-effort per-track beat-offset correction for YouTube audio. "
                         "Aliases on periodic beats and can't reconcile beats+segments — "
                         "diagnostic only; default off (report raw timings).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.manifest.exists():
        sys.exit(f"No manifest at {args.manifest}. Run eval/acquire_harmonix.py first.")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("audio_exists")]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("No tracks with audio in the manifest.")

    print(f"=> Scoring {len(rows)} tracks (per-fold CV, target_bpm={args.target})", file=sys.stderr)
    per_track, metrics = [], ["beat_f", "downbeat_f", "bound_f_0.5", "bound_f_3.0",
                              "pairwise_f", "v_measure"]
    for i, r in enumerate(rows, 1):
        ref_beats, ref_down = load_beats(r["beats_path"])
        ref_int, ref_lab = load_segments(r["segments_path"])
        target = resolve_target(args.target, r["audio_path"], r.get("bpm_ref"))
        structure = analyze_structure(r["audio_path"], target_bpm=target, model=r["model"])
        est_beats = np.array(structure.get("beats") or [])
        offset = estimate_offset(ref_beats, est_beats) if args.align else 0.0
        structure = shift_structure(structure, offset)
        s = score_track(ref_beats, ref_down, ref_int, ref_lab, structure)
        s.update(file=r["file"], fold=r["fold"], offset=round(offset, 3))
        per_track.append(s)
        print(f"   [{i}/{len(rows)}] {r['file']} (fold {r['fold']}, off {offset:+.2f}s): "
              + " ".join(f"{m}={s[m]:.3f}" for m in metrics), file=sys.stderr)

    agg = {m: round(st.mean(t[m] for t in per_track), 4) for m in metrics}
    mean_abs_off = round(st.mean(abs(t["offset"]) for t in per_track), 3)
    print("\n=== Harmonix structure (per-fold CV) ===")
    print(f"tracks: {len(per_track)}   target_bpm: {args.target}   "
          f"align: {f'on (mean |offset| {mean_abs_off}s)' if args.align else 'off (raw timings)'}")
    for m in metrics:
        print(f"  {m:<12} {agg[m]:.4f}")

    if args.out:
        args.out.write_text(json.dumps(
            {"n": len(per_track), "target": args.target, "aligned": args.align,
             "mean_abs_offset": mean_abs_off, "aggregate": agg, "per_track": per_track},
            indent=2,
        ))
        print(f"=> Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
