#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "mt3-infer>=0.1.3",
#   "torch==2.8.*",
#   "torchaudio==2.8.*",
#   "pytorch-lightning>=2.2",
#   "transformers==4.45.1",
#   "pretty_midi>=0.2.10",
#   "librosa>=0.10",
#   "soundfile>=0.12",
# ]
# ///
"""YourMT3+ pitched-stem transcription worker for jams.

Transcribes a (separated) pitched stem to note events with **YourMT3+** (Chang et al.,
MLSP 2024) — the strongest openly-available multi-instrument transcriber on Slakh. On our
Slakh2100-redux test harness (151 tracks, ground-truth stems) it scores note-F **0.849 on
bass and 0.849 on "other"** vs basic-pitch's 0.789/0.490. Runs via the MIT-licensed
`mt3-infer` toolkit with the Apache-2.0 YourMT3 checkpoint (YPTF.MoE+Multi), which avoids
the GPL-3.0 upstream research repo entirely.

Env notes (why the pins):
  * ``transformers`` is pinned to 4.45.1 — newer releases removed modules the YourMT3
    checkpoint's T5 code imports (``model_parallel_utils``).
  * ``pytorch-lightning`` is required by the checkpoint loader but not declared by
    mt3-infer.
  * torch/torchaudio are pinned **as a matched pair** (2.8.*). Leaving them unpinned let
    the resolver pick a cu130 torch newer than the box's CUDA driver, which silently fell
    back to CPU (~20x slower); pinning only one of the pair caused an ABI clash
    (``undefined symbol: aoti_torch_abi_version``). Bump both together, and only after
    verifying mt3-infer against the new pair on GPU.
  * **System requirement: git-lfs.** The first run clones the checkpoint from Hugging
    Face via git-lfs (~536 MB). If it is missing, install it (``apt install git-lfs`` /
    ``brew install git-lfs``) and run ``git lfs install`` once.

No fallback by design: if the model can't load or transcription fails, the error
propagates — accuracy must never silently degrade to a weaker transcriber.

Notes are emitted at SOUNDING pitch; the orchestrator (``jams.analysis.gm``) applies the
written-pitch bass convention (+12) and the monophonic post-filter uniformly across
transcribers.

Modes:
  single-shot:  yourmt3_worker.py --audio FILE           -> prints one JSON object
  serve (JSONL): yourmt3_worker.py --serve
     request:  {"audio": "stem.wav"}
     response: {"ok":true,"result":{"notes":[{"onset","offset","pitch","velocity"},...]}}
               | {"ok":false,"error":"..."}
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile

_warned_cpu_fallback = False


def _warn_if_cpu_with_gpu() -> None:
    """Loud, once-per-process, NON-fatal warning when an NVIDIA GPU is present but this
    torch build has no CUDA (wrong wheel/driver pairing) — mt3-infer would then run on
    CPU ~20x slower with no other signal."""
    global _warned_cpu_fallback
    if _warned_cpu_fallback:
        return
    import shutil
    from pathlib import Path

    import torch

    if torch.cuda.is_available():
        _warned_cpu_fallback = True
        return
    if Path("/proc/driver/nvidia").exists() or shutil.which("nvidia-smi"):
        _warned_cpu_fallback = True
        print(
            "[yourmt3] " + "!" * 70 + "\n[yourmt3] WARNING: NVIDIA GPU present but torch "
            "has no CUDA support — check the torch build vs driver pairing. Running on "
            "CPU (~20x slower).\n[yourmt3] " + "!" * 70,
            file=sys.stderr, flush=True,
        )


def transcribe_pitched(wav: str) -> list[dict]:
    """Transcribe one stem with YourMT3+; flat (instrument-agnostic) non-drum notes."""
    import librosa
    import pretty_midi
    from mt3_infer import transcribe

    _warn_if_cpu_with_gpu()
    y, _ = librosa.load(wav, sr=16000, mono=True)
    try:
        midi = transcribe(y, model="yourmt3", sr=16000)  # mido MidiFile
    except Exception as exc:
        if "clone" in str(exc).lower() or "lfs" in str(exc).lower():
            raise RuntimeError(
                "YourMT3 checkpoint download failed — git-lfs is required "
                "(install git-lfs, then run `git lfs install`)."
            ) from exc
        raise
    # mido MidiFile -> pretty_midi for absolute-time note events.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.mid")
        midi.save(p)
        pm = pretty_midi.PrettyMIDI(p)
    notes = [
        {"onset": round(float(n.start), 4), "offset": round(float(n.end), 4),
         "pitch": int(n.pitch), "velocity": int(n.velocity)}
        for inst in pm.instruments if not inst.is_drum for n in inst.notes
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
            # Keep protocol stdout clean: model-load prints go to stderr.
            with contextlib.redirect_stdout(sys.stderr):
                notes = transcribe_pitched(req["audio"])
            out = {"ok": True, "result": {"notes": notes}}
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--audio", help="Stem wav to transcribe (single-shot)")
    args = ap.parse_args()
    if args.serve:
        _serve()
        return
    if not args.audio:
        ap.error("provide --audio FILE or --serve")
    with contextlib.redirect_stdout(sys.stderr):  # model-load prints must not precede JSON
        notes = transcribe_pitched(args.audio)
    print(json.dumps({"notes": notes}, indent=2))


if __name__ == "__main__":
    main()
