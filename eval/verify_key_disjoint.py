# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mirdata>=0.3.8"]
# ///
"""Verify train/test id-disjointness for the key CNN (K10).

Checks, from the mirdata indexes alone (no audio download), that the
beatport_key training corpus shares zero Beatport track ids with the
giantsteps_key test set. Both datasets' mirdata track ids are sequential
index numbers; the actual Beatport preview id is the leading integer of
each audio filename, so that is what we compare.

The K10 ledger entry in paper/EXPERIMENTS.md records this as verified
empirically; this script makes the check reproducible.

Usage: uv run eval/verify_key_disjoint.py
"""

from __future__ import annotations

import os
import re
import sys

import mirdata


def beatport_ids(dataset) -> set[str]:
    ids = set()
    for entry in dataset._index["tracks"].values():
        fname = os.path.basename(entry["audio"][0])
        m = re.match(r"(\d+)\D", fname)
        if not m:
            raise ValueError(f"no beatport id in filename: {fname!r}")
        ids.add(m.group(1))
    return ids


def main() -> int:
    bp = mirdata.initialize("beatport_key")
    gs = mirdata.initialize("giantsteps_key")
    bp.download(["index"])
    gs.download(["index"])

    bp_ids = beatport_ids(bp)
    gs_ids = beatport_ids(gs)
    overlap = bp_ids & gs_ids

    print(f"beatport_key tracks: {len(bp_ids)}")
    print(f"giantsteps_key tracks: {len(gs_ids)}")
    print(f"id overlap: {len(overlap)}")
    if overlap:
        print("OVERLAPPING IDS:", sorted(overlap)[:20], file=sys.stderr)
        return 1
    print("OK: training corpus is id-disjoint from the test set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
