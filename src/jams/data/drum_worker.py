#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "adtof-pytorch @ git+https://github.com/xavriley/ADTOF-pytorch",
#   "torch>=2.0",
#   "librosa>=0.10",
#   "pretty_midi>=0.2.10",
#   "numpy",
# ]
# ///
"""Drum-transcription worker for jams — ADTOF Frame_RNN (PyTorch port).

Transcribes a drum stem to notes with `ADTOF-pytorch
<https://github.com/xavriley/ADTOF-pytorch>`_, a faithful torch port of the ADTOF CRNN
(Zehren et al., 2021/2023; F 88.5 vs the original's 88.7 on MDBDrums++) trained on
crowdsourced real music. Unlike the original (TF/madmom) or Magenta's OaF drums
(tensorflow==2.9.1, no Apple-Silicon wheel), this stack is torch+librosa+pretty_midi only —
so drum transcription runs on macOS arm64, Linux, and CI identically.

Kept in its own uv env (not merged into stems_worker) so the pipeline pieces stay
independently replaceable, and because adtof-pytorch is a git dependency with no declared
license (it ports GPL'd ADTOF): jams never imports it — it is installed and executed only
inside this isolated subprocess env, keeping jams itself MIT.

The model emits 5 drum classes at ADTOF's conventional pitches — 35 kick, 38 snare,
42 hi-hat, 47 tom, 49 crash+ride — with fixed velocity (the model does not predict
dynamics). Pitches are returned raw; the orchestrator (``jams.analysis.gm``) normalises
them onto the canonical GM percussion set.

Modes:
  single-shot:  drum_worker.py --drums-wav FILE           -> prints one JSON object
  serve (JSONL): drum_worker.py --serve
     request:  {"drums_wav": "drums.wav"}
     response: {"ok": true, "result": {"notes": [{"onset","offset","pitch","velocity"}, ...]}}
               | {"ok": false, "error": "..."}
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path


def transcribe_drums(wav: str) -> list[dict]:
    """Transcribe a drum stem to notes via the ADTOF Frame_RNN model (torch)."""
    import pretty_midi
    import torch
    from adtof_pytorch import transcribe_to_midi

    device = "cuda" if torch.cuda.is_available() else "cpu"  # tiny CRNN; cpu is fast enough
    with tempfile.TemporaryDirectory() as td:
        midi_out = Path(td) / "drums_adtof.mid"
        transcribe_to_midi(wav, midi_out, device=device)
        pm = pretty_midi.PrettyMIDI(str(midi_out))
    notes = [
        {"onset": float(n.start), "offset": float(n.end),
         "pitch": int(n.pitch), "velocity": int(n.velocity)}
        for inst in pm.instruments for n in inst.notes
    ]
    notes.sort(key=lambda x: x["onset"])
    return notes


def _serve() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            # Keep protocol stdout clean: model/lib prints go to stderr.
            with contextlib.redirect_stdout(sys.stderr):
                notes = transcribe_drums(req["drums_wav"])
            out = {"ok": True, "result": {"notes": notes}}
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--drums-wav")
    args = ap.parse_args()
    if args.serve:
        _serve()
        return
    if not args.drums_wav:
        ap.error("provide --drums-wav FILE or --serve")
    with contextlib.redirect_stdout(sys.stderr):  # model-load prints must not precede the JSON
        notes = transcribe_drums(args.drums_wav)
    print(json.dumps({"notes": notes}, indent=2))


if __name__ == "__main__":
    main()
