#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "madmom @ git+https://github.com/CPJKU/madmom",
#   "Cython",
#   "numpy>=1.23,<2",
#   "scipy",
# ]
# ///
"""madmom CNNKeyRecognitionProcessor baseline on the GiantSteps Key manifest.

Korzeniowski & Widmer's all-conv key CNN (ISMIR 2018) is the strongest classic
supervised baseline with shipped weights (published 74.6 weighted on GiantSteps Key,
trained on GS-MTG high-confidence + Billboard + classical — no test contamination).
Running it on OUR manifest puts it in the same table as edma/S-KEY/fusion under
identical conditions.

LICENSE NOTE: madmom's model weights are CC BY-NC-SA 4.0 — used here for evaluation
comparison only; they are NOT shipped with or used by the jams service.

Writes one JSON line per track: {"track_id", "madmom_key", "probs": [24]}.

    uv run --script eval/madmom_key_baseline.py \
        --manifest eval/data/manifest.jsonl --out eval/data/gsmtg/madmom_gskey.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label

    proc = CNNKeyRecognitionProcessor()
    rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("audio_exists")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(x)["track_id"] for x in out_path.read_text().splitlines() if x.strip()}
        print(f"resuming: {len(done)} done", flush=True)

    with open(out_path, "a") as fh:
        for i, r in enumerate(rows, 1):
            tid = str(r["track_id"])
            if tid in done:
                continue
            try:
                probs = proc(r["audio_path"])
                rec = {
                    "track_id": tid,
                    "madmom_key": key_prediction_to_label(probs),
                    "probs": [round(float(x), 6) for x in probs[0]],
                }
            except Exception as exc:  # noqa: BLE001
                rec = {"track_id": tid, "error": f"{type(exc).__name__}: {exc}"}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"[{i}/{len(rows)}]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
