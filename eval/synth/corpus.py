"""Batch corpus driver: render N sub-style-balanced D&B tracks with a frozen val split.

Writes, per track, 4 premaster stems + premaster mix + mastered mix (FLAC 44.1 kHz/16-bit) plus
a per-track JSON (spec, loudness, master gain-ratio, seed, timbres, asset attributions). Emits a
corpus manifest and a frozen, sub-style-balanced ES2-synth-val split (~20%) with a sha256.
Deterministic per seed; resumable (skips tracks already fully rendered).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config, oneshots, render

SR = 44100
_STEMS = ("drums", "bass", "other", "vocals")
_FILES = _STEMS + ("mix_premaster", "mix_master")

_SOURCES: dict = {}
_ASSETS_DIR = ""


def _init_worker(oneshots_pkl: str, kit_808: str) -> None:
    global _SOURCES
    _SOURCES = oneshots.load_drum_sources(oneshots_pkl, kit_808)


def _write_flac(path: Path, audio: np.ndarray) -> int:
    sf.write(str(path), audio.T, SR, format="FLAC", subtype="PCM_16")
    return path.stat().st_size


def _track_dir(out_dir: str, track_id: str) -> Path:
    return Path(out_dir) / "audio" / track_id


def _is_done(out_dir: str, track_id: str) -> bool:
    d = _track_dir(out_dir, track_id)
    return all((d / f"{f}.flac").exists() for f in _FILES) and (d / "track.json").exists()


def _attributions() -> list[dict]:
    return [
        {"asset": "E-GMD (Expanded Groove MIDI Dataset)", "license": "CC-BY 4.0",
         "url": "https://magenta.tensorflow.org/datasets/e-gmd",
         "use": "real drum one-shots (sliced via aligned MIDI), re-sequenced into D&B patterns"},
        {"asset": "TidalCycles TR-808 (sounds-tr808-fischer)", "license": "CC0-1.0",
         "url": "https://github.com/tidalcycles/sounds-tr808-fischer",
         "use": "real CC0 TR-808 one-shots, re-sequenced into D&B patterns"},
        {"asset": "Surge XT", "license": "GPL-3 + output grant",
         "url": "https://surge-synthesizer.github.io",
         "use": "bass + synth voices (procedural params, no factory-preset content)"},
        {"asset": "Dexed", "license": "GPL-3",
         "url": "https://asb2m10.github.io/dexed", "use": "FM voices (own procedural patches)"},
        {"asset": "DawDreamer", "license": "MIT", "url": "https://github.com/DBraun/DawDreamer",
         "use": "offline render engine"},
    ]


def _render_one(args: tuple[int, str, int, str]) -> dict:
    index, substyle, seed, out_dir = args
    track_id = f"{substyle}_{index:04d}_s{seed}"
    if _is_done(out_dir, track_id):
        d = _track_dir(out_dir, track_id)
        return json.loads((d / "track.json").read_text())
    spec = config.sample_track_spec(seed, substyle)
    result = render.render_track(spec, _SOURCES)
    d = _track_dir(out_dir, track_id)
    d.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for stem in _STEMS:
        sizes[stem] = _write_flac(d / f"{stem}.flac", result["stems"][stem])
    sizes["mix_premaster"] = _write_flac(d / "mix_premaster.flac", result["premaster"])
    sizes["mix_master"] = _write_flac(d / "mix_master.flac", result["master"])
    row = {
        "track_id": track_id, "index": index, "substyle": substyle, "seed": seed,
        **spec.as_dict(), **result["info"],
        "files": {f: f"audio/{track_id}/{f}.flac" for f in _FILES},
        "bytes": sizes, "total_bytes": int(sum(sizes.values())),
        "assets": _attributions(),
    }
    (d / "track.json").write_text(json.dumps(row, indent=2))
    return row


def _frozen_val_split(rows: list[dict], val_frac: float = 0.2) -> dict:
    """~val_frac of each sub-style, deterministic + disjoint, with a sha256 over val ids."""
    val_ids: list[str] = []
    by_sub: dict[str, list[dict]] = {}
    for r in rows:
        by_sub.setdefault(r["substyle"], []).append(r)
    for sub in sorted(by_sub):
        tracks = sorted(by_sub[sub], key=lambda r: r["seed"])
        k = max(1, round(len(tracks) * val_frac))
        step = max(1, len(tracks) // k)
        picked = [tracks[i]["track_id"] for i in range(0, len(tracks), step)][:k]
        val_ids.extend(picked)
    val_set = set(val_ids)
    train_ids = [r["track_id"] for r in rows if r["track_id"] not in val_set]
    sha = hashlib.sha256("\n".join(sorted(val_ids)).encode()).hexdigest()
    return {"val": sorted(val_ids), "train": sorted(train_ids), "val_sha256": sha}


def build(out_dir: str, n: int, oneshots_pkl: str, kit_808: str, seed_base: int,
          workers: int) -> dict:
    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    subs = config.SUBSTYLES_ORDER
    jobs = [(i, subs[i % len(subs)], seed_base + i, out_dir) for i in range(n)]

    t0 = time.time()
    rows: list[dict] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(oneshots_pkl, kit_808)) as pool:
        for i, row in enumerate(pool.imap_unordered(_render_one, jobs, chunksize=2)):
            rows.append(row)
            if (i + 1) % 20 == 0 or i + 1 == n:
                el = time.time() - t0
                print(f"[{i + 1}/{n}] {el:.0f}s  ({el / (i + 1):.1f}s/track)", flush=True)
    wall = time.time() - t0

    rows.sort(key=lambda r: r["index"])
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    split = _frozen_val_split(rows)
    (out / "split.json").write_text(json.dumps(split, indent=2))

    sub_counts: dict[str, int] = {}
    for r in rows:
        sub_counts[r["substyle"]] = sub_counts.get(r["substyle"], 0) + 1
    total_bytes = sum(r["total_bytes"] for r in rows)
    summary = {
        "n_tracks": len(rows), "substyle_counts": sub_counts,
        "total_gb": round(total_bytes / 1e9, 2),
        "wall_clock_s": round(wall, 1), "s_per_track": round(wall / max(1, len(rows)), 2),
        "val_n": len(split["val"]), "train_n": len(split["train"]),
        "val_sha256": split["val_sha256"],
        "total_dur_min": round(sum(r["duration_sec"] for r in rows) / 60, 1),
    }
    (out / "corpus_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--oneshots", required=True)
    ap.add_argument("--kit-808", default="")
    ap.add_argument("--seed-base", type=int, default=20260715)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    summary = build(args.out_dir, args.n, args.oneshots, args.kit_808, args.seed_base,
                    args.workers)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
