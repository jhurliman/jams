#!/usr/bin/env -S uv run --extra eval
"""Evaluate jams song-structure analysis against an annotated dataset.

Dataset-agnostic: it consumes a common-schema manifest produced by one of the
``acquire_*`` scripts and scores the production ``analyze_structure`` with mir_eval:

  beats / downbeats : F-measure (mir_eval.beat, 70 ms window)
  segment boundaries: Hit-Rate F @ 0.5 s and @ 3 s (mir_eval.segment.detection)
  segment labeling  : pairwise F + V-measure (mir_eval.segment)

Each manifest row carries the ``model`` to score it with — Harmonix uses per-fold CV
(``harmonix-fold{i%8}``, the held-out model), while out-of-domain sets (Raveform, EDM-98)
use ``harmonix-all``. Reference annotations are loaded per ``format`` ("raveform" reads the
combined beat CSV; Harmonix reads its beats/segments files). Datasets without beat
annotations (EDM-98) skip the beat/downbeat metrics automatically.

If an ``alignment.jsonl`` (from ``align_harmonix.py``) sits next to the manifest, each
track's affine map ``t_audio = a·t_anno + b`` is applied to the model output (mapping it
back to annotation time) and ``case3`` tracks (different edit — unusable) are dropped. Sets
whose audio matches the annotations natively (Raveform) need no alignment file.

``--target`` sets the beat-tracking BPM constraint: jams' own tempo (default) / dataset
``ref`` BPM / ``none``.

    uv run --extra eval eval/evaluate_structure.py                                   # Harmonix
    uv run --extra eval eval/evaluate_structure.py --manifest eval/data/raveform/manifest.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import csv as csvmod
import json
import statistics as st
import sys
from pathlib import Path

import mir_eval
import numpy as np

from jams.analysis.structure import analyze_structure

MANIFEST = Path(__file__).resolve().parent / "data" / "harmonix" / "manifest.jsonl"
METRICS = ["beat_f", "downbeat_f", "bound_f_0.5", "bound_f_3.0", "pairwise_f", "v_measure"]


# --- reference-annotation loading (per dataset format) ---------------------


def _boundaries_to_intervals(times, labels) -> tuple[np.ndarray, list[str]]:
    intervals, lab = [], []
    for i in range(len(times) - 1):
        intervals.append([times[i], times[i + 1]])
        lab.append(labels[i])
    return np.array(intervals), lab


def _harmonix_refs(row: dict):
    beats, downbeats = [], []
    for line in Path(row["beats_path"]).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            beats.append(float(parts[0]))
            if parts[1] == "1":
                downbeats.append(float(parts[0]))
    times, labels = [], []
    for line in Path(row["segments_path"]).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            times.append(float(parts[0]))
            labels.append(parts[1])
    seg_int, seg_lab = _boundaries_to_intervals(times, labels)
    return np.array(beats), np.array(downbeats), seg_int, seg_lab


def _raveform_refs(row: dict):
    """Beats/downbeats/segments from a Raveform combined beat CSV (time,downbeat,section)."""
    beats, downbeats, seg_times, seg_labels = [], [], [], []
    prev = None
    for r in csvmod.DictReader(Path(row["beats_csv"]).read_text().splitlines()):
        t = float(r["time"])
        beats.append(t)
        if r["downbeat"] == "1":
            downbeats.append(t)
        if r["section"] != prev:
            seg_times.append(t)
            seg_labels.append(r["section"])
            prev = r["section"]
    end = beats[-1] if beats else 0.0
    seg_int, seg_lab = _boundaries_to_intervals(seg_times + [end], seg_labels + ["end"])
    return np.array(beats), np.array(downbeats), seg_int, seg_lab


def load_refs(row: dict):
    return _raveform_refs(row) if row.get("format") == "raveform" else _harmonix_refs(row)


# --- alignment (optional, from align_harmonix.py) --------------------------


def load_alignment(manifest: Path) -> dict:
    path = manifest.parent / "alignment.jsonl"
    if not path.exists():
        return {}
    return {r["stem"]: r
            for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}


def warp_structure(structure: dict, a: float, b: float) -> dict:
    """Map model output (audio time) back to annotation time: t_anno = (t_audio - b) / a."""
    if a == 1.0 and b == 0.0:
        return structure

    def f(t: float) -> float:
        return (t - b) / a

    return {
        **structure,
        "beats": [f(x) for x in (structure.get("beats") or [])],
        "downbeats": [f(x) for x in (structure.get("downbeats") or [])],
        "segments": [{**s, "start": f(s["start"]), "end": f(s["end"])}
                     for s in (structure.get("segments") or [])],
    }


# --- scoring ---------------------------------------------------------------


def est_segments(structure: dict) -> tuple[np.ndarray, list[str]]:
    segs = structure.get("segments") or []
    intervals = np.array([[s["start"], s["end"]] for s in segs]) if segs else np.empty((0, 2))
    return intervals, [s["label"] for s in segs]


def _positive_intervals(intervals: np.ndarray, labels: list[str]):
    """Drop zero/negative-duration intervals — mir_eval rejects them (All-In-One can
    emit a degenerate segment where start==end)."""
    if intervals.size == 0:
        return intervals, labels
    keep = intervals[:, 1] > intervals[:, 0]
    return intervals[keep], [lab for lab, k in zip(labels, keep, strict=True) if k]


def score_track(ref_beats, ref_down, ref_seg_int, ref_seg_lab, structure) -> dict:
    est_beats = np.array(structure.get("beats") or [])
    est_down = np.array(structure.get("downbeats") or [])
    est_int, est_lab = _positive_intervals(*est_segments(structure))
    ref_seg_int, ref_seg_lab = _positive_intervals(np.asarray(ref_seg_int), ref_seg_lab)

    # None when the dataset lacks that reference (e.g. segments-only EDM-98).
    out: dict = {
        "beat_f": mir_eval.beat.f_measure(ref_beats, est_beats)
        if (ref_beats.size and est_beats.size) else None,
        "downbeat_f": mir_eval.beat.f_measure(ref_down, est_down)
        if (ref_down.size and est_down.size) else None,
        "bound_f_0.5": None, "bound_f_3.0": None, "pairwise_f": None, "v_measure": None,
    }
    if est_int.size and ref_seg_int.size:
        try:
            t_max = float(max(ref_seg_int[-1, 1], est_int[-1, 1]))
            adjust = mir_eval.util.adjust_intervals
            ref_i, ref_l = adjust(ref_seg_int, ref_seg_lab, t_min=0.0, t_max=t_max)
            est_i, est_l = adjust(est_int, est_lab, t_min=0.0, t_max=t_max)
            out["bound_f_0.5"] = mir_eval.segment.detection(ref_i, est_i, window=0.5, trim=True)[2]
            out["bound_f_3.0"] = mir_eval.segment.detection(ref_i, est_i, window=3.0, trim=True)[2]
            out["pairwise_f"] = mir_eval.segment.pairwise(ref_i, ref_l, est_i, est_l)[2]
            over, under, _ = mir_eval.segment.nce(ref_i, ref_l, est_i, est_l)
            out["v_measure"] = 0.0 if (over + under) == 0 else 2 * over * under / (over + under)
        except ValueError:  # degenerate intervals mir_eval still rejects — skip this track's segs
            pass
    return out


def resolve_target(mode: str, audio: str, bpm_ref: float | None) -> float | None:
    if mode == "none":
        return None
    if mode == "ref":
        return bpm_ref
    from jams.analysis.tempo import detect_tempo

    return detect_tempo(audio)["bpm"]


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(st.mean(vals), 4) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--target", choices=["jams", "ref", "none"], default="jams")
    ap.add_argument("--keep-case3", action="store_true",
                    help="keep aligner-flagged case3 tracks (different edit); default drops them")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.manifest.exists():
        sys.exit(f"No manifest at {args.manifest}. Run an eval/acquire_*.py first.")
    rows = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("audio_exists")]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("No tracks with audio in the manifest.")

    align = load_alignment(args.manifest)
    dataset = rows[0].get("dataset", "?")

    # Checkpoint: append each scored track to a partial JSONL so a crash/sleep resumes
    # instead of restarting from track 1 (the full run is multi-hour). Needs --out.
    ckpt = args.out.with_suffix(".partial.jsonl") if args.out else None
    per_track: list[dict] = []
    done: set[str] = set()
    if ckpt and ckpt.exists():
        per_track = [json.loads(x) for x in ckpt.read_text().splitlines() if x.strip()]
        done = {t["track_id"] for t in per_track}
        print(f"=> Resuming from {ckpt}: {len(done)} tracks already scored", file=sys.stderr)

    print(f"=> Scoring {len(rows)} {dataset} tracks "
          f"(target_bpm={args.target}, alignment={'on' if align else 'native'})", file=sys.stderr)

    dropped = 0
    with (open(ckpt, "a") if ckpt else contextlib.nullcontext()) as ckpt_fh:
        for i, r in enumerate(rows, 1):
            tid = r.get("track_id") or r.get("file")
            if tid in done:
                continue  # already scored in a previous run
            al = align.get(tid)
            if al and al.get("klass") == "case3" and not args.keep_case3:
                dropped += 1
                continue
            ref_beats, ref_down, ref_int, ref_lab = load_refs(r)
            target = resolve_target(args.target, r["audio_path"], r.get("bpm_ref"))
            structure = analyze_structure(r["audio_path"], target_bpm=target, model=r["model"])
            if al:
                structure = warp_structure(structure, al["a"], al["b"])
            s = score_track(ref_beats, ref_down, ref_int, ref_lab, structure)
            s.update(track_id=tid, model=r["model"])
            per_track.append(s)
            if ckpt_fh is not None:
                ckpt_fh.write(json.dumps(s) + "\n")
                ckpt_fh.flush()
            shown = " ".join(f"{m}={s[m]:.3f}" for m in METRICS if s[m] is not None)
            print(f"   [{i}/{len(rows)}] {tid} ({r['model']}): {shown}", file=sys.stderr)

    agg = {m: _mean(t[m] for t in per_track) for m in METRICS}
    print(f"\n=== {dataset} structure ===")
    print(f"tracks: {len(per_track)} scored, {dropped} dropped (case3)   target_bpm: {args.target}")
    for m in METRICS:
        print(f"  {m:<12} {agg[m] if agg[m] is not None else 'n/a'}")

    if args.out:
        args.out.write_text(json.dumps(
            {"dataset": dataset, "n": len(per_track), "dropped_case3": dropped,
             "target": args.target, "aggregate": agg, "per_track": per_track}, indent=2))
        print(f"=> Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
