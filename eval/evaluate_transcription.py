#!/usr/bin/env -S uv run --extra eval
"""Evaluate jams stem-separation + per-stem MIDI transcription against a labelled dataset.

Dataset-agnostic: consumes a common-schema manifest from an ``acquire_*`` script and scores
the production stems pipeline (``jams.analysis.stems.analyze_stems``) with mir_eval.

Two modes (decouples transcription quality from separation quality):

  --mode oracle : transcribe the dataset's GROUND-TRUTH stems (separation skipped). Isolates
                  the transcribers (basic-pitch / OaF-drums). This is the headline number and
                  works before separation is polished.
  --mode e2e    : separate the mix with Demucs, then transcribe. Also scores separation SDR
                  (needs ground-truth stem wavs in the manifest).

Metrics (mir_eval):
  pitched stems (bass/other/vocals) : note onset+pitch F/P/R (transcription.precision_recall_
                                      f1_overlap, offsets ignored) + velocity-aware F
  drum stem                         : per-GM-instrument onset F (onset.f_measure, 50 ms) macro
  separation (e2e)                  : SI-SDR per stem

Reference notes come from the manifest per ``format``: slakh/egmd read a ground-truth ``.mid``
with pretty_midi; medleydb converts its f0 annotation to notes. Drum pitches on both sides are
normalised to the canonical GM percussion set (shared with the worker via ``stems_worker``).

    uv run --extra eval eval/evaluate_transcription.py --manifest eval/data/slakh/manifest.jsonl
    uv run --extra eval eval/evaluate_transcription.py \
        --manifest eval/data/egmd/manifest.jsonl --mode oracle
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics as st
import sys
from pathlib import Path

import mir_eval
import numpy as np

from jams.analysis import gm
from jams.analysis.stems import analyze_stems

PITCHED_STEMS = ("bass", "other", "vocals")
GM_DRUM_CLASSES = gm.GM_DRUM_CLASSES
MANIFEST = Path(__file__).resolve().parent / "data" / "slakh" / "manifest.jsonl"


# --- reference-note loading (per dataset format) ---------------------------


def _midi_to_notes(midi_path: str, canon_drums: bool) -> list[dict]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)
    notes: list[dict] = []
    for inst in pm.instruments:
        for n in inst.notes:
            pitch = gm.canon_drum_pitch(n.pitch) if canon_drums else n.pitch
            notes.append(
                {"onset": float(n.start), "offset": float(n.end),
                 "pitch": int(pitch), "velocity": int(n.velocity)}
            )
    notes.sort(key=lambda x: x["onset"])
    return notes


def _f0_to_notes(csv_path: str, min_dur: float = 0.05) -> list[dict]:
    """Convert a MedleyDB pitch annotation (time,freq[Hz]) to note events.

    Voiced frames (freq>0) are quantised to the nearest MIDI pitch; runs of the same pitch
    become one note. A pragmatic melody note-ify for note-level scoring.
    """
    import csv as csvmod

    rows = []
    for r in csvmod.reader(Path(csv_path).read_text().splitlines()):
        if len(r) < 2:
            continue
        try:
            rows.append((float(r[0]), float(r[1])))
        except ValueError:
            continue
    notes: list[dict] = []
    cur_pitch: int | None = None
    start = 0.0
    prev_t = 0.0

    def _emit() -> None:
        if cur_pitch is not None and prev_t - start >= min_dur:
            notes.append({"onset": start, "offset": prev_t, "pitch": cur_pitch, "velocity": 100})

    for t, f in rows:
        p = int(round(69 + 12 * np.log2(f / 440.0))) if f > 0 else None
        if p != cur_pitch:
            _emit()
            cur_pitch, start = p, t
        prev_t = t
    _emit()
    return notes


def ref_notes_for_stem(row: dict, stem_type: str) -> list[dict] | None:
    """Ground-truth notes for one stem class, or None if the dataset lacks it."""
    fmt = row.get("format")
    if fmt in ("slakh",):
        mp = (row.get("midi") or {}).get(stem_type)
        return _midi_to_notes(mp, canon_drums=(stem_type == "drums")) if mp else None
    if fmt == "egmd":
        if stem_type != "drums":
            return None
        return _midi_to_notes(row["drum_midi_path"], canon_drums=True)
    if fmt == "medleydb":
        # Single melodic line; score it against the 'other' (or 'vocals') transcription.
        if stem_type not in ("other", "vocals"):
            return None
        return _f0_to_notes(row["pitch_annotation_path"])
    return None


def oracle_stems(row: dict, no_drums: bool = False) -> dict[str, str]:
    """Ground-truth stem wavs to transcribe in oracle mode."""
    fmt = row.get("format")
    if fmt == "slakh":
        out = {k: v for k, v in (row.get("stems") or {}).items() if v}
    elif fmt == "egmd":
        out = {"drums": row["drum_audio_path"]}
    elif fmt == "medleydb":
        out = {"other": row["audio_path"]}
    else:
        out = {}
    if no_drums:
        out.pop("drums", None)
    return out


# --- metrics ----------------------------------------------------------------


def _midi_hz(pitches: list[int]) -> np.ndarray:
    return np.array([440.0 * 2 ** ((p - 69) / 12) for p in pitches], dtype=float)


def note_prf(ref: list[dict], est: list[dict]) -> dict:
    """Note onset+pitch precision/recall/F (offsets ignored) via mir_eval.transcription."""
    if not ref:
        return {"note_f": None, "note_p": None, "note_r": None}
    if not est:
        return {"note_f": 0.0, "note_p": 0.0, "note_r": 0.0}
    ref_i = np.array([[n["onset"], max(n["onset"] + 1e-3, n["offset"])] for n in ref])
    est_i = np.array([[n["onset"], max(n["onset"] + 1e-3, n["offset"])] for n in est])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, _midi_hz([n["pitch"] for n in ref]),
        est_i, _midi_hz([n["pitch"] for n in est]),
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    return {"note_f": float(f), "note_p": float(p), "note_r": float(r)}


def drum_prf(ref: list[dict], est: list[dict], classes: str = "adtof5") -> dict:
    """Per-drum-class onset F (50 ms window), macro-averaged over classes present.

    ``classes='adtof5'`` (default) scores in the standard 5-class ADT vocabulary
    (kick/snare/hats/toms/cymbals — the drum CNN's output space; the flag name predates
    it, from the retired ADTOF port, and is kept for artifact/CLI compatibility);
    ``'gm10'`` scores the full canonical GM set (penalises the model for distinctions
    it cannot make, e.g. open vs closed hat).
    """
    if classes == "adtof5":
        ref = [{**n, "pitch": gm.reduce_drum_pitch_5(n["pitch"])} for n in ref]
        est = [{**n, "pitch": gm.reduce_drum_pitch_5(n["pitch"])} for n in est]
        class_set = gm.GM_DRUM_5CLASSES
    else:
        class_set = GM_DRUM_CLASSES
    per_class: dict[str, float] = {}
    for pitch in class_set:
        ref_on = np.array(sorted(n["onset"] for n in ref if n["pitch"] == pitch))
        est_on = np.array(sorted(n["onset"] for n in est if n["pitch"] == pitch))
        if ref_on.size == 0 and est_on.size == 0:
            continue
        if ref_on.size == 0 or est_on.size == 0:
            per_class[str(pitch)] = 0.0
            continue
        f, _p, _r = mir_eval.onset.f_measure(ref_on, est_on, window=0.05)
        per_class[str(pitch)] = float(f)
    macro = round(st.mean(per_class.values()), 4) if per_class else None
    return {"drum_onset_f": macro, "drum_per_class": per_class}


def si_sdr(ref_wav: str, est_wav: str) -> float | None:
    """Scale-invariant SDR (dB) between a reference and estimated stem wav."""
    try:
        import librosa

        r, sr = librosa.load(ref_wav, sr=None, mono=True)
        e, _ = librosa.load(est_wav, sr=sr, mono=True)
    except Exception:
        return None
    n = min(len(r), len(e))
    r, e = r[:n], e[:n]
    if n == 0 or np.allclose(r, 0):
        return None
    alpha = float(np.dot(e, r) / (np.dot(r, r) + 1e-9))
    target = alpha * r
    noise = e - target
    denom = float(np.sum(noise**2)) + 1e-9
    return round(10 * np.log10((float(np.sum(target**2)) + 1e-9) / denom), 3)


# --- scoring one track ------------------------------------------------------


def score_track(row: dict, result: dict, mode: str, bass_octave_shift: int = 0,
                drum_classes: str = "adtof5") -> dict:
    out: dict = {"track_id": row.get("track_id"), "stems": {}}
    for tr in result.get("transcriptions", []):
        st_type = tr["stem_type"]
        ref = ref_notes_for_stem(row, st_type)
        if ref is None:
            continue
        notes = tr["notes"]
        # basic-pitch transcribes bass one octave below Slakh's written-MIDI convention; this
        # optional shift aligns them so the score reflects pitch-class + relative-melody accuracy.
        if st_type == "bass" and bass_octave_shift:
            notes = [{**n, "pitch": n["pitch"] + 12 * bass_octave_shift} for n in notes]
        scores = (drum_prf(ref, notes, drum_classes) if tr["is_drums"]
                  else note_prf(ref, notes))
        out["stems"][st_type] = scores

    if mode == "e2e" and row.get("format") == "slakh":
        est_by_type = {s["stem_type"]: s["audio_path"] for s in result.get("stems", [])}
        sdr: dict[str, float] = {}
        for st_type, gt in (row.get("stems") or {}).items():
            if gt and est_by_type.get(st_type):
                v = si_sdr(gt, est_by_type[st_type])
                if v is not None:
                    sdr[st_type] = v
        if sdr:
            out["sdr"] = sdr
    return out


# --- aggregation ------------------------------------------------------------


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(st.mean(vals), 4) if vals else None


def aggregate(per_track: list[dict]) -> dict:
    agg: dict = {"note_f": {}, "drum_onset_f": None, "sdr": {}}
    for st_type in PITCHED_STEMS:
        vals = [t["stems"][st_type]["note_f"] for t in per_track
                if st_type in t.get("stems", {}) and "note_f" in t["stems"][st_type]]
        agg["note_f"][st_type] = _mean(vals)
    agg["note_f"]["overall"] = _mean(
        [t["stems"][s].get("note_f") for t in per_track for s in t.get("stems", {})
         if "note_f" in t["stems"][s]]
    )
    agg["drum_onset_f"] = _mean(
        [t["stems"]["drums"]["drum_onset_f"] for t in per_track
         if "drums" in t.get("stems", {})]
    )
    for st_type in ("drums", "bass", "other", "vocals"):
        agg["sdr"][st_type] = _mean([t["sdr"][st_type] for t in per_track
                                     if st_type in t.get("sdr", {})])
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--mode", choices=["oracle", "e2e"], default="oracle")
    ap.add_argument("--no-drums", action="store_true",
                    help="skip drum transcription; still scores pitched stems + "
                         "separation SDR")
    ap.add_argument("--drum-classes", choices=["adtof5", "gm10"], default="adtof5",
                    help="drum scoring vocabulary: 5-class kick/snare/hats/toms/cymbals "
                         "(standard ADT eval, default) or the full 10-class GM set")
    ap.add_argument("--bass-octave-shift", type=int, default=0,
                    help="shift est bass pitches by N octaves before scoring; +1 aligns "
                         "basic-pitch's output with Slakh's written-MIDI bass convention")
    ap.add_argument("--quantize", action="store_true",
                    help="snap onsets to a beat grid before scoring (default off for eval)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="discard any existing --out checkpoint instead of resuming from it "
                         "(use after changing the pipeline, or stale scores are reported)")
    args = ap.parse_args()

    if not args.manifest.exists():
        sys.exit(f"No manifest at {args.manifest}. Run an eval/acquire_*.py first.")
    rows = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("audio_exists")]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("No tracks with audio in the manifest.")
    dataset = rows[0].get("dataset", "?")

    ckpt = args.out.with_suffix(".partial.jsonl") if args.out else None
    if args.fresh and ckpt and ckpt.exists():
        ckpt.unlink()
    per_track: list[dict] = []
    done: set[str] = set()
    if ckpt and ckpt.exists():
        per_track = [json.loads(x) for x in ckpt.read_text().splitlines() if x.strip()]
        done = {t["track_id"] for t in per_track}
        print(f"=> Resuming from {ckpt}: {len(done)} tracks already scored", file=sys.stderr)

    print(f"=> Scoring {len(rows)} {dataset} tracks (mode={args.mode})", file=sys.stderr)
    failed = 0
    with (open(ckpt, "a") if ckpt else contextlib.nullcontext()) as ckpt_fh:
        for i, r in enumerate(rows, 1):
            tid = r.get("track_id")
            if tid in done:
                continue
            try:
                if args.mode == "oracle":
                    result = analyze_stems(None, stems=oracle_stems(r, args.no_drums),
                                           quantize=args.quantize,
                                           transcribe_drums=not args.no_drums)
                else:
                    result = analyze_stems(r.get("mix_path") or r.get("audio_path"),
                                           quantize=args.quantize,
                                           transcribe_drums=not args.no_drums)
                s = score_track(r, result, args.mode, args.bass_octave_shift,
                                args.drum_classes)
            except Exception as exc:  # noqa: BLE001 - transient worker/IO error; skip & resume
                failed += 1
                print(f"   [{i}/{len(rows)}] {tid}: FAILED ({exc}); skipping", file=sys.stderr)
                continue
            per_track.append(s)
            if ckpt_fh is not None:
                ckpt_fh.write(json.dumps(s) + "\n")
                ckpt_fh.flush()
            summary = {k: v.get("note_f", v.get("drum_onset_f")) for k, v in s["stems"].items()}
            print(f"   [{i}/{len(rows)}] {tid}: {summary}", file=sys.stderr)

    agg = aggregate(per_track)
    print(f"\n=== {dataset} transcription ({args.mode}) ===")
    print(f"tracks: {len(per_track)} scored, {failed} failed")
    print(f"  note_f (onset+pitch):  {agg['note_f']}")
    print(f"  drum_onset_f (macro):  {agg['drum_onset_f']}")
    if args.mode == "e2e":
        print(f"  SI-SDR dB per stem:    {agg['sdr']}")

    if args.out:
        args.out.write_text(json.dumps(
            {"dataset": dataset, "mode": args.mode, "n": len(per_track), "failed": failed,
             "aggregate": agg, "per_track": per_track}, indent=2))
        print(f"=> Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
