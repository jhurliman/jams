# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["librosa>=0.10,<1.0", "numpy>=1.26,<2.3", "torch==2.8.*"]
# ///
"""Re-run the MIREX K10 package (eval/mirex2026_key/predict_key.py, bundled weights) on the
GiantSteps Key audio and compare with the committed development predictions in
paper/evidence/key/gskey_predictions.jsonl.

Two questions are answered per excerpt:
  1. Does the packaged inference path reproduce the committed K10 prediction?
  2. Does the training-time feature cache (log-CQT stored as float16 and cast back to
     float32) change the argmax relative to the package's float32 features?

Audio: the GiantSteps Key mp3s as distributed by mirdata (giantsteps_key/audio/*.mp3).
Excerpts are matched by the file name recorded in eval/data/manifest.jsonl.

Usage: uv run paper/key_rerun_check.py --audio-dir /path/to/giantsteps_key/audio
Writes paper/evidence/key/k10_rerun_check.json (summary only) and exits nonzero on any
mismatch between the package and the committed predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "eval" / "mirex2026_key"
MANIFEST = ROOT / "eval" / "data" / "manifest.jsonl"
PRED = ROOT / "paper" / "evidence" / "key" / "gskey_predictions.jsonl"
OUT = ROOT / "paper" / "evidence" / "key" / "k10_rerun_check.json"
sys.path.insert(0, str(PKG))


def one(args):
    tid, path = args
    import librosa
    import numpy as np
    import predict_key as pk
    import torch

    torch.set_num_threads(1)
    model = pk.build_model()
    model.load_state_dict(torch.load(pk.WEIGHTS, map_location="cpu"))
    model.eval()
    t0 = time.perf_counter()
    y, _ = librosa.load(path, sr=pk.SR, mono=True)
    n_bins = (pk.N_OCT * 12 + 2 * pk.PAD_SEMI) * (pk.BINS_PER_OCT // 12)
    fmin = librosa.note_to_hz("C1") * 2 ** (-pk.PAD_SEMI / 12)
    C = np.log1p(
        np.abs(
            librosa.cqt(
                y,
                sr=pk.SR,
                hop_length=pk.HOP,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=pk.BINS_PER_OCT,
            )
        )
    )
    per = pk.BINS_PER_OCT // 12
    x32 = C[pk.PAD_SEMI * per : pk.PAD_SEMI * per + pk.N_OCT * pk.BINS_PER_OCT].astype(np.float32)
    x16 = x32.astype(np.float16).astype(np.float32)
    with torch.no_grad():
        l32 = model(torch.from_numpy(x32)[None, None])[0]
        l16 = model(torch.from_numpy(x16)[None, None])[0]
    p32, p16 = torch.softmax(l32, 0), torch.softmax(l16, 0)

    def lab(c):
        return f"{pk.NOTES[c // 2]} {'minor' if c % 2 else 'major'}"

    return {
        "track_id": tid,
        "pred_f32": lab(int(l32.argmax())),
        "pred_f16": lab(int(l16.argmax())),
        "max_abs_logit_diff": float((l32 - l16).abs().max()),
        "max_abs_prob_diff": float((p32 - p16).abs().max()),
        "dur_s": float(len(y) / pk.SR),
        "wall_s": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    pred = {
        json.loads(line)["track_id"]: json.loads(line)
        for line in PRED.read_text().splitlines()
        if line.strip()
    }
    usable = {t for t, r in pred.items() if r["k10_key"]}
    jobs, missing = [], []
    for tid in sorted(usable, key=int):
        p = a.audio_dir / pred[tid]["audio_file"]
        (jobs if p.exists() else missing).append((tid, str(p)))
    if missing:
        sys.exit(
            f"{len(missing)} of {len(usable)} usable excerpts have no audio under {a.audio_dir}"
        )
    results = []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(one, jobs, chunksize=4):
            res["k10_committed"] = pred[res["track_id"]]["k10_key"]
            results.append(res)
    walls = sorted(r["wall_s"] for r in results)
    summary = {
        "n_excerpts": len(results),
        "package_matches_committed": sum(r["pred_f32"] == r["k10_committed"] for r in results),
        "float16_cache_matches_committed": sum(
            r["pred_f16"] == r["k10_committed"] for r in results
        ),
        "float32_eq_float16_argmax": sum(r["pred_f32"] == r["pred_f16"] for r in results),
        "max_abs_logit_diff_float16_vs_float32": max(r["max_abs_logit_diff"] for r in results),
        "max_abs_prob_diff_float16_vs_float32": max(r["max_abs_prob_diff"] for r in results),
        "mismatches": [
            (r["track_id"], r["pred_f32"], r["k10_committed"])
            for r in results
            if r["pred_f32"] != r["k10_committed"]
        ],
        "median_excerpt_duration_s": round(
            sorted(r["dur_s"] for r in results)[len(results) // 2], 2
        ),
        "median_in_process_wall_s_one_thread": round(walls[len(walls) // 2], 3),
        "note": "in-process wall time excludes interpreter start and model load; see the "
        "package README for end-to-end timing",
    }
    OUT.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    ok = summary["package_matches_committed"] == len(results) == len(usable)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
