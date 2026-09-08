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

YourMT3+ emits one MIDI track per detected instrument (GM program). We keep that
information (OV1 A-lab, paper/EXPERIMENTS.md): every note carries its ``program`` and the
response includes a compact per-program ``instruments`` summary. The flat ``notes`` list is
unchanged otherwise — additive fields only.

Modes:
  single-shot:  yourmt3_worker.py --audio FILE           -> prints one JSON object
  serve (JSONL): yourmt3_worker.py --serve
     request:  {"audio": "stem.wav"}
     response: {"ok":true,"result":{
                  "notes":[{"onset","offset","pitch","velocity","program"},...],
                  "instruments":[{"program","name","n_notes"},...]}}
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


_device = None


def _select_device() -> str:
    """cuda -> mps -> cpu. mt3-infer's own "auto" only knows cuda-or-cpu, so on Apple
    Silicon it silently runs the autoregressive decoder on CPU (~2.5x slower than MPS,
    measured 26 s vs 10 s per 60 s of audio on an M-series laptop)."""
    global _device
    if _device is None:
        import torch

        if torch.cuda.is_available():
            _device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            _device = "mps"
            # The T5 decoder's KV cache grows every step, so each step allocates new,
            # differently sized tensors that the MPS caching allocator never reuses. On a
            # note-dense segment that reached ~38 GB and swapped (128 s for one 60 s window on
            # a 34 GB M2 Max). Capping the allocator forces block recycling: 20 s, 3.5 GB.
            # 15% of a 34 GB machine (3.7 GiB) OOMs on that window, so cap by an absolute
            # budget rather than a bare fraction; JAMS_MPS_MEMORY_GB overrides.
            budget = float(os.environ.get("JAMS_MPS_MEMORY_GB", "8")) * 1e9
            frac = min(0.9, budget / torch.mps.recommended_max_memory())
            torch.mps.set_per_process_memory_fraction(frac)
            print(f"[yourmt3] mps allocator cap: {frac:.2f} of recommended working set",
                  file=sys.stderr, flush=True)
        else:
            _device = "cpu"
        print(f"[yourmt3] device: {_device}", file=sys.stderr, flush=True)
    return _device


def _model(device: str):
    """mt3-infer's cached model for `device` (same call `transcribe()` makes internally).
    On MPS the adapter's fixed batch of 8 segments fragments the allocator during the
    autoregressive decode; a smaller batch keeps the footprint under the cap."""
    from mt3_infer import load_model

    m = load_model(model="yourmt3", device=device)
    if device == "mps" and not getattr(m, "_jams_mps_batch", False):
        orig = m.model.inference_file
        bsz = int(os.environ.get("JAMS_YOURMT3_MPS_BATCH", "4"))

        def inference_file(bsz=bsz, audio_segments=None, _orig=orig, _bsz=bsz):
            return _orig(bsz=_bsz, audio_segments=audio_segments)

        m.model.inference_file = inference_file
        m._jams_mps_batch = True
    return m


def transcribe_pitched(wav: str) -> dict:
    """Transcribe one stem with YourMT3+.

    Returns ``{"notes": [...], "instruments": [...]}``: a flat onset-sorted non-drum note
    list where each note carries the GM ``program`` of its source instrument track, plus a
    per-program summary (``program``/``name``/``n_notes``). Instrument-labeling quality of
    these retained programs is measured in the experiment ledger (OV1 A-lab).
    """
    import librosa
    import pretty_midi

    _warn_if_cpu_with_gpu()
    y, _ = librosa.load(wav, sr=16000, mono=True)
    try:
        try:
            midi = _model(_select_device()).transcribe(y, sr=16000)  # mido MidiFile
        except RuntimeError as exc:
            if _select_device() != "mps" or "out of memory" not in str(exc).lower():
                raise
            import torch

            torch.mps.empty_cache()
            print("[yourmt3] MPS out of memory on this stem; retrying on CPU",
                  file=sys.stderr, flush=True)
            midi = _model("cpu").transcribe(y, sr=16000)
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
    # RETAIN the per-instrument MIDI tracks (programs) — previously discarded here.
    notes: list[dict] = []
    note_counts: dict[int, int] = {}
    for inst in pm.instruments:
        if inst.is_drum or not inst.notes:
            continue
        program = int(inst.program)
        note_counts[program] = note_counts.get(program, 0) + len(inst.notes)
        notes.extend(
            {"onset": round(float(n.start), 4), "offset": round(float(n.end), 4),
             "pitch": int(n.pitch), "velocity": int(n.velocity), "program": program}
            for n in inst.notes
        )
    notes.sort(key=lambda x: x["onset"])
    instruments = [
        {"program": p, "name": pretty_midi.program_to_instrument_name(p), "n_notes": c}
        for p, c in sorted(note_counts.items())
    ]
    return {"notes": notes, "instruments": instruments}


def _serve() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            # Keep protocol stdout clean: model-load prints go to stderr.
            with contextlib.redirect_stdout(sys.stderr):
                result = transcribe_pitched(req["audio"])
            out = {"ok": True, "result": result}
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
        result = transcribe_pitched(args.audio)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
