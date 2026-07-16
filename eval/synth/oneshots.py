"""Build a CC-BY drum one-shot library by slicing isolated hits out of E-GMD.

E-GMD (Magenta, CC-BY 4.0) is real Roland TD-17 drum audio with time-aligned GM-percussion
MIDI. We use the MIDI onsets to slice *isolated* single-drum hits (no other drum within a tight
window, and a clean tail) into a per-category one-shot library. The generator then re-sequences
these real timbres into D&B patterns (in ``drums.py``), which closes the "un-modern synthetic
drums" gap the ES2 probe flagged — with a clean, attributable license (no copyrighted breaks).

Provenance for every source track is recorded so the dataset card can attribute E-GMD.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100

# GM percussion note -> our drum category.
GM_CATEGORY: dict[int, str] = {
    35: "kick", 36: "kick",
    37: "rim", 38: "snare", 40: "snare",
    39: "clap",
    42: "closed_hat", 44: "closed_hat",
    46: "open_hat",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    49: "crash", 52: "crash", 55: "crash", 57: "crash",
    51: "ride", 53: "ride", 59: "ride",
}

CATEGORIES = ["kick", "snare", "rim", "clap", "closed_hat", "open_hat", "tom", "ride", "crash"]

# Per-category slicing (min clean tail before the next onset, max window) in seconds.
_TAIL = {"kick": 0.12, "snare": 0.12, "rim": 0.06, "clap": 0.10, "closed_hat": 0.045,
         "open_hat": 0.08, "tom": 0.12, "ride": 0.08, "crash": 0.18}
_MAXWIN = {"kick": 0.32, "snare": 0.30, "rim": 0.14, "clap": 0.22, "closed_hat": 0.11,
           "open_hat": 0.34, "tom": 0.34, "ride": 0.30, "crash": 0.70}
_MINVEL = 40


def _midi_onsets(path: str) -> list[tuple[float, int, int]]:
    """Return [(time_s, pitch, velocity)] note-ons; time via MidiFile real-time iteration."""
    import mido

    out: list[tuple[float, int, int]] = []
    t = 0.0
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            out.append((t, msg.note, msg.velocity))
    out.sort(key=lambda x: x[0])
    return out


def _slice_hit(audio: np.ndarray, t: float, tail: float, maxwin: float) -> np.ndarray | None:
    win = min(maxwin, max(_TAIL_MIN, tail - 0.005))
    i0 = int(t * SR)
    n = int(win * SR)
    if i0 < 0 or i0 + n > len(audio) or n < int(0.02 * SR):
        return None
    seg = audio[i0:i0 + n].astype(np.float64).copy()
    # Trim to the transient: start ~3 ms before the local peak in the first 30 ms.
    head = seg[:int(0.03 * SR)]
    if head.size == 0:
        return None
    pk = int(np.argmax(np.abs(head)))
    start = max(0, pk - int(0.003 * SR))
    seg = seg[start:]
    if seg.size < int(0.02 * SR):
        return None
    peak = float(np.abs(seg).max())
    if peak < 1e-4:
        return None
    seg *= 0.9 / peak
    # Fade out the last 25 % so tails don't click when re-triggered.
    fn = int(len(seg) * 0.25)
    if fn > 1:
        seg[-fn:] *= np.linspace(1.0, 0.0, fn)
    return seg.astype(np.float32)


_TAIL_MIN = 0.03


def extract_library(
    egmd_manifest: str,
    out_dir: str,
    n_files: int = 140,
    per_cat: int = 48,
    seed: int = 0,
) -> dict:
    """Slice one-shots from the first ``n_files`` E-GMD tracks; cache library + provenance.

    Writes ``<out_dir>/oneshots.pkl`` (dict category -> list[np.ndarray]) and
    ``<out_dir>/oneshots_provenance.json`` (source tracks + license). Returns a summary dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(egmd_manifest).read_text().splitlines() if line]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    rows = rows[:n_files]

    # candidates[cat] = list of (velocity, sample, source_track_id)
    candidates: dict[str, list[tuple[int, np.ndarray, str]]] = defaultdict(list)
    sources: set[str] = set()
    for row in rows:
        apath, mpath = row.get("audio"), row.get("midi")
        if not apath or not mpath or not Path(apath).exists() or not Path(mpath).exists():
            continue
        try:
            audio, sr = sf.read(apath)
        except Exception:
            continue
        if sr != SR:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        onsets = _midi_onsets(mpath)
        times = np.array([o[0] for o in onsets])
        for k, (t, pitch, vel) in enumerate(onsets):
            cat = GM_CATEGORY.get(pitch)
            if cat is None or vel < _MINVEL:
                continue
            # concurrency: any *other* onset within +/-15 ms disqualifies (want isolation)
            lo, hi = t - 0.015, t + 0.015
            near = np.sum((times >= lo) & (times <= hi))
            if near > 1:
                continue
            tail = (times[k + 1] - t) if k + 1 < len(times) else _MAXWIN[cat]
            if tail < _TAIL[cat]:
                continue
            seg = _slice_hit(audio, t, tail, _MAXWIN[cat])
            if seg is None:
                continue
            candidates[cat].append((vel, seg, row.get("track_id", Path(apath).stem)))
    library: dict[str, list[np.ndarray]] = {}
    counts: dict[str, int] = {}
    for cat in CATEGORIES:
        cand = candidates.get(cat, [])
        if not cand:
            continue
        # Spread across MANY source tracks for timbral variety: round-robin the sources,
        # taking the loudest/cleanest hit from each in turn.
        by_src: dict[str, list] = defaultdict(list)
        for vel, seg, src in cand:
            by_src[src].append((vel, seg))
        for src in by_src:
            by_src[src].sort(key=lambda x: x[0], reverse=True)
        keep: list = []
        order = sorted(by_src, key=lambda s: -by_src[s][0][0])
        while len(keep) < per_cat and any(by_src.values()):
            for src in order:
                if by_src[src] and len(keep) < per_cat:
                    vel, seg = by_src[src].pop(0)
                    keep.append((seg, src))
        library[cat] = [s for s, _ in keep]
        counts[cat] = len(keep)
        sources.update(src for _, src in keep)

    with open(out / "oneshots.pkl", "wb") as f:
        pickle.dump(library, f, protocol=4)
    provenance = {
        "source_dataset": "E-GMD (Expanded Groove MIDI Dataset)",
        "license": "CC-BY 4.0",
        "attribution": "Callender, Hawthorne, Engel — E-GMD, Magenta/Google. "
                       "Real Roland TD-17 drum recordings, GM-percussion MIDI aligned.",
        "url": "https://magenta.tensorflow.org/datasets/e-gmd",
        "usage": "Isolated single-drum hits sliced via aligned MIDI onsets into a one-shot "
                 "library; re-sequenced into D&B patterns. No E-GMD grooves reproduced.",
        "n_source_tracks_used": len(sources),
        "source_track_ids": sorted(sources),
        "per_category_counts": counts,
    }
    (out / "oneshots_provenance.json").write_text(json.dumps(provenance, indent=2))
    return {"counts": counts, "n_sources": len(sources)}


def load_library(path: str) -> dict[str, list[np.ndarray]]:
    with open(path, "rb") as f:
        return pickle.load(f)


# TidalCycles TR-808 (CC0-1.0) folder -> our categories.
_808_MAP = {
    "bd8": ["kick"], "sd8": ["snare"], "rs8": ["rim"], "cp8": ["clap"],
    "ch8": ["closed_hat"], "oh8": ["open_hat"], "cy8": ["crash", "ride"],
    "lt8": ["tom"], "mt8": ["tom"], "ht8": ["tom"],
    "lc8": ["tom"], "mc8": ["tom"], "hc8": ["tom"], "cl8": ["rim"],
}


def load_808_library(kit_dir: str) -> dict[str, list[np.ndarray]]:
    """Load the CC0 TR-808 one-shots into the same category->list[np.ndarray] shape."""
    from pathlib import Path as _P
    lib: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES + ["clap"]}
    root = _P(kit_dir)
    for folder, cats in _808_MAP.items():
        for wav in sorted((root / folder).glob("*")):
            if wav.suffix.lower() != ".wav":
                continue
            try:
                y, sr = sf.read(str(wav))
            except Exception:
                continue
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SR:
                continue
            y = 0.9 * y.astype(np.float32) / (np.abs(y).max() + 1e-9)
            for c in cats:
                lib[c].append(y)
    return {c: v for c, v in lib.items() if v}


def load_drum_sources(egmd_pkl: str, kit_808_dir: str | None = None) -> dict[str, dict]:
    """Return {"egmd": <lib>, "808": <lib>} for the DrumEngine to choose per track."""
    sources = {"egmd": load_library(egmd_pkl)}
    if kit_808_dir and Path(kit_808_dir).exists():
        eight = load_808_library(kit_808_dir)
        if eight:
            sources["808"] = eight
    return sources


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-files", type=int, default=140)
    ap.add_argument("--per-cat", type=int, default=48)
    args = ap.parse_args()
    summary = extract_library(args.manifest, args.out_dir, args.n_files, args.per_cat)
    print(json.dumps(summary, indent=2))
