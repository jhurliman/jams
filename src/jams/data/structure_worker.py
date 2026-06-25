#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["all-in-one-mps>=0.1", "numpy>=1.26"]
# ///
"""All-In-One (PyTorch/MPS) song-structure worker for jams.

Self-contained uv script: ``uv run --script structure_worker.py ...`` resolves and
caches its own environment from the inline metadata above. It runs in a *separate*
interpreter from jams itself (jams is pinned to Python 3.14 for essentia, while
All-In-One needs torch/natten-mps/demucs, which have no 3.14 wheels). jams invokes
it as a subprocess via ``uv run`` and never imports it. Two modes:

  single-shot:  structure_worker.py --audio FILE [--target-bpm F] [--model NAME]
                -> prints one JSON object to stdout

  serve (JSONL): structure_worker.py --serve
                reads one JSON request per line on stdin:
                  {"audio": "...", "target_bpm": 174.0|null, "model": "harmonix-all"}
                writes one JSON response per line on stdout:
                  {"ok": true, "result": {...}} | {"ok": false, "error": "..."}
                The All-In-One model is loaded lazily and kept resident, so the
                ~15-20s cold start is paid once per worker, not per request.

``target_bpm`` is applied by re-running All-In-One's own DBN beat tracker on the
frame-level activations with ``min_bpm/max_bpm = target_bpm +/- 1`` — the same
constraint the Replicate fork used, and the diagnostic note for half-time genres
(D&B/dubstep) where the tracker otherwise lands an octave low.

Heavy imports live inside functions so the module is import-safe in jams' env.
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import sys

# All-In-One emits at 100 frames/sec (44100 Hz / 441-sample hop).
_FPS = 100


def _beat_index(timestamp: float, beats: list[float]) -> int:
    """Nearest beat number (1-indexed) for a segment boundary timestamp."""
    if not beats:
        return 0
    idx = bisect.bisect_left(beats, timestamp)
    if idx == 0:
        return 1
    if idx >= len(beats):
        return len(beats)
    if timestamp - beats[idx - 1] <= beats[idx] - timestamp:
        return idx
    return idx + 1


def _retrack_with_target_bpm(result, target_bpm: float):
    """Re-run the DBN on activations constrained to target_bpm +/- 1 BPM.

    Returns (beats, downbeats, bpm). Mirrors allin1's metrical postprocessing.
    """
    import numpy as np
    from allin1.postprocessing.dbn_native import DBNDownBeatTrackingProcessor
    from allin1.postprocessing.tempo import estimate_tempo_from_beats

    act_beat = np.asarray(result.activations["beat"], dtype=np.float64)
    act_down = np.asarray(result.activations["downbeat"], dtype=np.float64)
    no_beat = 1.0 - act_beat
    no_down = 1.0 - act_down
    no = (no_beat + no_down) / 2.0
    xbeat = np.maximum(1e-8, act_beat - act_down)
    combined = np.stack([xbeat, act_down, no], axis=-1)
    combined /= combined.sum(axis=-1, keepdims=True)

    dbn = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], threshold=0.5, fps=_FPS,
        min_bpm=max(1.0, target_bpm - 1.0), max_bpm=target_bpm + 1.0,
    )
    pred = dbn(combined[:, :2])
    beats = pred[:, 0].astype(float).tolist()
    positions = pred[:, 1].astype(int)
    downbeats = pred[positions == 1, 0].astype(float).tolist()
    est = estimate_tempo_from_beats(beats)  # None if < 2 beats
    bpm = float(est) if est is not None else None
    return beats, downbeats, bpm


def analyze(audio: str, target_bpm: float | None, model: str) -> dict:
    import allin1

    need_act = target_bpm is not None
    # All-In-One prints progress to stdout; keep our stdout protocol clean.
    with contextlib.redirect_stdout(sys.stderr):
        result = allin1.analyze(
            paths=audio, model=model, include_activations=need_act, keep_byproducts=False,
        )

    beats = [float(b) for b in result.beats]
    downbeats = [float(d) for d in result.downbeats]
    bpm = float(result.bpm) if result.bpm is not None else None
    method = f"allin1-mps-local:{model}"

    if target_bpm is not None and result.activations is not None:
        beats, downbeats, bpm = _retrack_with_target_bpm(result, target_bpm)
        method += f"+targetbpm{target_bpm:g}"

    segments = [
        {
            "start": float(s.start), "end": float(s.end), "label": s.label,
            "start_beat": _beat_index(float(s.start), beats),
            "end_beat": _beat_index(float(s.end), beats),
        }
        for s in result.segments
    ]
    return {
        "bpm": bpm, "beats": beats, "downbeats": downbeats,
        "segments": segments, "method": method,
    }


def _serve() -> None:
    """Read JSONL requests on stdin, write JSONL responses on stdout (model resident)."""
    # Touch the import once up front so failures surface immediately.
    import allin1  # noqa: F401

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = analyze(req["audio"], req.get("target_bpm"), req.get("model", "harmonix-all"))
            out = {"ok": True, "result": res}
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="All-In-One structure worker for jams")
    ap.add_argument("--serve", action="store_true", help="persistent JSONL stdin/stdout mode")
    ap.add_argument("--audio", help="audio file (single-shot mode)")
    ap.add_argument("--target-bpm", type=float, default=None)
    ap.add_argument("--model", default="harmonix-all")
    args = ap.parse_args()

    if args.serve:
        _serve()
        return
    if not args.audio:
        ap.error("either --serve or --audio is required")
    print(json.dumps(analyze(args.audio, args.target_bpm, args.model)))


if __name__ == "__main__":
    main()
