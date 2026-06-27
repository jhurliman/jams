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

``target_bpm`` is applied as a *post-hoc octave correction* of All-In-One's native
beat grid (see ``_octave_correct``): the native beats are high quality and usually at
the right octave, so we keep them untouched and only densify/thin the grid when
``target_bpm`` shows the native tempo is a clean half/double (the half-time-genre case,
e.g. D&B/dubstep read an octave low). An earlier approach re-ran the DBN beat tracker
with a tight ``min_bpm/max_bpm = target_bpm +/- 1`` window — that wrecked beat-F (0.99 ->
0.70) even at the correct tempo, so it was replaced.

Heavy imports live inside functions so the module is import-safe in jams' env.
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import sys


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


def _densify(times: list[float]) -> list[float]:
    """Insert the midpoint between each consecutive pair (doubles the grid density)."""
    if len(times) < 2:
        return times
    out: list[float] = []
    for a, b in zip(times, times[1:], strict=False):
        out.append(a)
        out.append((a + b) / 2.0)
    out.append(times[-1])
    return out


def _octave_correct(beats: list[float], downbeats: list[float], native_bpm: float,
                    target_bpm: float) -> tuple[list[float], list[float], float]:
    """Scale the native beat grid to ``target_bpm``'s octave on a clean half/double.

    All-In-One's native beats are high quality and usually at the right octave already; a
    tight DBN re-track to fix the rare half/double-time error wrecks beat-F. Instead we keep
    the native grid and only adjust its *density* when the native tempo is ~2× (heard
    half-time → densify) or ~0.5× (heard double-time → thin) the target. When the octave
    already matches — the common case — the native beats pass through untouched, so there is
    no precision penalty.
    """
    if not native_bpm or native_bpm <= 0:
        return beats, downbeats, native_bpm
    ratio = target_bpm / native_bpm
    if 1.6 <= ratio <= 2.4:        # native heard half-time → double the grid
        return _densify(beats), _densify(downbeats), native_bpm * 2.0
    if 0.42 <= ratio <= 0.62:      # native heard double-time → thin the grid
        return beats[::2], downbeats[::2], native_bpm / 2.0
    return beats, downbeats, native_bpm  # octave already correct → untouched


def analyze(audio: str, target_bpm: float | None, model: str) -> dict:
    import allin1

    # All-In-One prints progress to stdout; keep our stdout protocol clean.
    with contextlib.redirect_stdout(sys.stderr):
        result = allin1.analyze(
            paths=audio, model=model, include_activations=False, keep_byproducts=False,
        )

    beats = [float(b) for b in result.beats]
    downbeats = [float(d) for d in result.downbeats]
    bpm = float(result.bpm) if result.bpm is not None else None
    method = f"allin1-mps-local:{model}"

    # Keep the native beats (high quality); only octave-correct when target_bpm says the
    # native tempo is a clean half/double. Octave-correct tracks pass through untouched.
    if target_bpm is not None and bpm:
        new_beats, new_downbeats, new_bpm = _octave_correct(
            beats, downbeats, bpm, float(target_bpm))
        if new_bpm != bpm:
            beats, downbeats, bpm = new_beats, new_downbeats, new_bpm
            method += f"+octave{bpm:g}"

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
