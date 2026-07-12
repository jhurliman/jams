#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "adtof-pytorch @ git+https://github.com/xavriley/ADTOF-pytorch",
#   "torch==2.8.*",
#   "torchaudio==2.8.*",
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


def _velocities_from_audio(wav: str, notes: list[dict]) -> list[dict]:
    """Replace the model's fixed velocity with dynamics measured from the stem signal.

    ADTOF predicts hits but not dynamics, so every note arrives at velocity 100 — flat,
    machine-gun MIDI. Each onset's loudness is measured directly (RMS over a 30 ms window
    starting at the hit) and mapped onto MIDI 30..127 relative to the loudest hit of the
    same class in the track. Per-class normalisation keeps quiet instruments (e.g. hats
    under a loud kick) expressive.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(wav, sr=None, mono=True)
    if not len(y) or not notes:
        return notes
    win = max(1, int(0.03 * sr))
    peak_of_note: list[float] = []
    peaks_by_class: dict[int, list[float]] = {}
    for n in notes:
        i = int(n["onset"] * sr)
        seg = y[i: i + win]
        peak = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
        peak_of_note.append(peak)
        peaks_by_class.setdefault(n["pitch"], []).append(peak)
    ref = {p: (max(v) or 1e-9) for p, v in peaks_by_class.items()}
    out = []
    for n, peak in zip(notes, peak_of_note, strict=True):
        # sqrt compresses the range so ghost notes stay audible (perceptual-ish mapping)
        vel = 30 + int(round(97 * (peak / ref[n["pitch"]]) ** 0.5))
        out.append({**n, "velocity": max(1, min(127, vel))})
    return out


_warned_cpu_fallback = False


def _warn_if_cpu_with_gpu() -> None:
    """Loud, once-per-process, NON-fatal warning when an NVIDIA GPU is present but this
    torch build has no CUDA (wrong wheel/driver pairing). The CRNN is small enough that
    CPU works, but on a GPU box a missing-CUDA torch means the env resolved wrong."""
    global _warned_cpu_fallback
    if _warned_cpu_fallback:
        return
    import shutil

    if Path("/proc/driver/nvidia").exists() or shutil.which("nvidia-smi"):
        _warned_cpu_fallback = True
        print(
            "[drums] " + "!" * 70 + "\n[drums] WARNING: NVIDIA GPU present but torch has "
            "no CUDA support — check the torch build vs driver pairing. Running on CPU."
            "\n[drums] " + "!" * 70,
            file=sys.stderr, flush=True,
        )


def transcribe_drums(wav: str) -> list[dict]:
    """Transcribe a drum stem to notes via the ADTOF Frame_RNN model (torch)."""
    import pretty_midi
    import torch
    from adtof_pytorch import transcribe_to_midi

    device = "cuda" if torch.cuda.is_available() else "cpu"  # tiny CRNN; cpu is fast enough
    if device == "cpu":
        _warn_if_cpu_with_gpu()
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
    return _velocities_from_audio(wav, notes)


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
