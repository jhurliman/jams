#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "skey @ git+https://github.com/deezer/skey",
#   "torch~=2.7.0",
#   "torchaudio~=2.7.0",
#   "numpy~=2.2.0",
#   "einops==0.8.*",
#   "nnAudio==0.3.3",
#   "soundfile>=0.13",
# ]
# ///
"""EVAL-ONLY — not on the production path. S-KEY key-estimation worker.

Production key detection is the in-process key CNN (``jams.analysis.key_cnn``); no jams
code launches this worker. It is preserved so the banked S-KEY features
(``eval/data/gsmtg/skey_gskey.jsonl``) that ``eval/stats_significance.py`` replays for
the paper's K4/K6 baseline rows (paper/EXPERIMENTS.md) stay regenerable from committed
code. Do not delete while the paper reports those rows. Inventory:
src/jams/data/README.md.

Runs `deezer/skey <https://github.com/deezer/skey>`_ (Kong et al., ICASSP 2025; MIT, the
checkpoint ships inside the package) and returns the softmaxed, mean-pooled 24-class key
posterior. The retired fusion pipeline combined this with the edma estimate — the two
systems were trained on disjoint data with different objectives, so their errors
decorrelate.

Kept in its own uv env so its pinned torch/nnAudio stack stays independent of jams' env.
Same resident-worker JSONL pattern as the stems/drums workers. The model is tiny; CPU
inference is a few seconds per track.

Modes:
  single-shot:  skey_worker.py --audio FILE            -> prints one JSON object
  serve (JSONL): skey_worker.py --serve
     request:  {"audio": "track.wav"}
     response: {"ok": true, "result": {"skey_key": "D Major", "posterior": [24 floats]}}
               | {"ok": false, "error": "..."}
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys

_MODEL = None  # (hcqt, chromanet, crop_fn, sr) resident across requests


def _load():
    global _MODEL
    if _MODEL is None:
        import torch  # noqa: F401 - ensure torch initializes before skey
        from skey.key_detection import load_checkpoint, load_model_components

        ckpt = load_checkpoint(None)  # packaged default checkpoint
        hcqt, chromanet, crop_fn = load_model_components(ckpt, __import__("torch").device("cpu"))
        _MODEL = (hcqt, chromanet, crop_fn, ckpt["audio"]["sr"])
    return _MODEL


def analyze(audio: str) -> dict:
    import torch
    from skey.key_detection import key_map, load_audio

    hcqt, chromanet, crop_fn, sr = _load()
    batch = load_audio(audio, sr)
    with torch.no_grad():
        cropped = crop_fn(hcqt(batch.unsqueeze(0)), torch.zeros(1))
        logits = chromanet(cropped)
        post = torch.softmax(torch.mean(logits, dim=0), dim=-1)
    return {
        "skey_key": key_map[int(post.argmax())],
        "posterior": [float(x) for x in post],
    }


def _serve() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            # Keep protocol stdout clean: model/lib prints go to stderr.
            with contextlib.redirect_stdout(sys.stderr):
                res = analyze(req["audio"])
            out = {"ok": True, "result": res}
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true", help="JSONL stdin/stdout server mode")
    ap.add_argument("--audio", help="Audio file to analyze (single-shot)")
    args = ap.parse_args()
    if args.serve:
        _serve()
        return
    if not args.audio:
        ap.error("provide --audio FILE or --serve")
    with contextlib.redirect_stdout(sys.stderr):  # model-load prints must not precede JSON
        res = analyze(args.audio)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
