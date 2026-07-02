#!/usr/bin/env -S uv run --extra eval
"""Acquire the Slakh2100 dataset for evaluating jams source separation (demucs stems).

Slakh2100 is a synthesized multitrack dataset: each song is rendered from a Lakh MIDI
file through professional sample-based instruments, giving per-instrument stems with
clean ground-truth audio *and* aligned MIDI. It is **instrumental** — there are no
vocals — so it is our reference set for the demucs ``drums`` / ``bass`` / ``other``
stem classes (``vocals`` is always null here).

We regroup the many per-instrument stems into the 4 demucs stem classes and render one
ground-truth wav + one merged MIDI per class:

  drums : stems where ``is_drum`` is true                  (drums.wav / drums.mid)
  bass  : GM bass programs 32..39, or inst_class == "Bass" (bass.wav  / bass.mid)
  other : all remaining pitched stems                      (other.wav / other.mid)
  vocals: none in Slakh                                    (null)

Grouped audio = element-wise SUM of the class stems (truncated to the common minimum
length, stereo kept if any stem is stereo). Grouped MIDI = all class instruments merged
into one PrettyMIDI (drums keep ``is_drum=True``).

The full Slakh download is 100 GB+, so this script never triggers ``.download()``. Use
the small ``babyslakh`` subset (default) for a quick end-to-end check, or point
``--data-home`` at a local full ``slakh2100_flac_redux`` install and pass ``--subset full``.
If the data is not present on disk, the script prints download instructions and exits
non-zero.

Examples
--------
    # Babyslakh smoke test (point --data-home at a local babyslakh_16k install):
    uv run --extra eval eval/acquire_slakh.py --subset babyslakh --data-home /data/babyslakh_16k

    # Just the first 3 multitracks:
    uv run --extra eval eval/acquire_slakh.py --data-home /data/babyslakh_16k --limit 3

    # Full Slakh2100 test split:
    uv run --extra eval eval/acquire_slakh.py --subset full --split test \
        --data-home /data/slakh2100_flac_redux
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mirdata
import numpy as np
import pretty_midi
import soundfile as sf

DATA_DIR = Path(__file__).resolve().parent / "data" / "slakh"
STEM_CLASSES = ["drums", "bass", "other", "vocals"]

_BABYSLAKH_HELP = """\
Babyslakh (the small ~2 GB, 16 kHz demo subset) was not found. Download it and point
--data-home at the extracted folder:

    # 2 GB tarball:
    curl -LO https://zenodo.org/record/4599666/files/babyslakh_16k.tar.gz
    tar xzf babyslakh_16k.tar.gz            # -> babyslakh_16k/
    uv run --extra eval eval/acquire_slakh.py --subset babyslakh --data-home $(pwd)/babyslakh_16k
"""

_FULL_HELP = """\
The full Slakh2100 (slakh2100_flac_redux, ~100 GB+) was not found on disk. This script
will NOT auto-download it. Obtain it (e.g. from https://zenodo.org/record/4599666) and
point --data-home at the extracted 'slakh2100_flac_redux' folder:

    uv run --extra eval eval/acquire_slakh.py --subset full \
        --data-home /path/to/slakh2100_flac_redux
"""


def classify(track) -> str:
    """Map one Slakh stem to a demucs stem class."""
    if track.is_drum:
        return "drums"
    prog = track.program_number
    if (prog is not None and 32 <= prog <= 39) or track.instrument == "Bass":
        return "bass"
    return "other"


def sum_audio(paths: list[str], out_path: Path) -> bool:
    """SUM the given stem wavs into one file. Returns False if nothing loadable."""
    loaded: list[tuple[np.ndarray, int]] = []
    for p in paths:
        try:
            data, sr = sf.read(p, always_2d=True)  # (n, channels)
        except Exception as exc:  # noqa: BLE001 - a corrupt/missing stem shouldn't kill the group
            print(f"      [warn] could not read stem {p}: {exc}", file=sys.stderr)
            continue
        loaded.append((data, sr))
    if not loaded:
        return False
    sr = loaded[0][1]
    n = min(d.shape[0] for d, _ in loaded)
    ch = max(d.shape[1] for d, _ in loaded)  # stereo if any stem is stereo
    mix = np.zeros((n, ch), dtype=np.float64)
    for data, _ in loaded:
        d = data[:n]
        if d.shape[1] == 1 and ch == 2:
            d = np.repeat(d, 2, axis=1)
        mix[:, : d.shape[1]] += d
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, mix.astype(np.float32), sr)
    return True


def merge_midi(paths: list[str], is_drum_group: bool, out_path: Path) -> bool:
    """Merge every instrument from the given MIDI files into one PrettyMIDI."""
    merged = pretty_midi.PrettyMIDI()
    any_notes = False
    for p in paths:
        try:
            pm = pretty_midi.PrettyMIDI(p)
        except Exception as exc:  # noqa: BLE001 - skip an unreadable stem MIDI
            print(f"      [warn] could not read MIDI {p}: {exc}", file=sys.stderr)
            continue
        for inst in pm.instruments:
            new = pretty_midi.Instrument(
                program=inst.program,
                is_drum=is_drum_group or inst.is_drum,
                name=inst.name,
            )
            new.notes = list(inst.notes)
            new.control_changes = list(inst.control_changes)
            new.pitch_bends = list(inst.pitch_bends)
            merged.instruments.append(new)
            any_notes = any_notes or bool(inst.notes)
    if not merged.instruments:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write(str(out_path))
    return any_notes or True


def build_mtrack(mtrack, out_root: Path) -> dict | None:
    """Group one multitrack into demucs stems; return a manifest row (or None to drop)."""
    if not mtrack.mix_path or not Path(mtrack.mix_path).exists():
        print(f"   [drop] {mtrack.mtrack_id}: mix audio missing", file=sys.stderr)
        return None

    groups: dict[str, list] = {"drums": [], "bass": [], "other": []}
    for stem in mtrack.tracks.values():
        if not stem.audio_path or not Path(stem.audio_path).exists():
            continue
        groups[classify(stem)].append(stem)

    if not any(groups.values()):
        print(f"   [drop] {mtrack.mtrack_id}: no stems with audio on disk", file=sys.stderr)
        return None

    track_dir = out_root / mtrack.mtrack_id
    stems: dict[str, str | None] = dict.fromkeys(STEM_CLASSES)
    midis: dict[str, str | None] = dict.fromkeys(STEM_CLASSES)

    for cls, members in groups.items():
        if not members:
            continue
        wav_paths = [s.audio_path for s in members]
        wav_out = track_dir / f"{cls}.wav"
        if sum_audio(wav_paths, wav_out):
            stems[cls] = str(wav_out)
        midi_paths = [s.midi_path for s in members if s.midi_path and Path(s.midi_path).exists()]
        if midi_paths:
            midi_out = track_dir / f"{cls}.mid"
            if merge_midi(midi_paths, is_drum_group=(cls == "drums"), out_path=midi_out):
                midis[cls] = str(midi_out)

    return {
        "dataset": "slakh",
        "format": "slakh",
        "track_id": mtrack.mtrack_id,
        "split": mtrack.split,
        "mix_path": str(mtrack.mix_path),
        "audio_exists": True,
        "stems": stems,
        "midi": midis,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["babyslakh", "full"], default="babyslakh",
                    help="'babyslakh' (small demo) or 'full' Slakh2100 "
                         "(100 GB+, not auto-downloaded)")
    ap.add_argument("--split", default="test",
                    help="Slakh split to keep for --subset full; '' or 'all' keeps every split "
                         "(babyslakh has no splits; ignored there)")
    ap.add_argument("--data-home", type=Path, default=None,
                    help="path to the local Slakh/babyslakh install, passed to mirdata.initialize")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "manifest.jsonl")
    args = ap.parse_args()

    # mirdata version keys: "baby" (babyslakh) and "2100-redux" (full). "default" -> "2100-redux".
    version = "baby" if args.subset == "babyslakh" else "2100-redux"
    data_home = str(args.data_home) if args.data_home else None
    dataset = mirdata.initialize("slakh", data_home=data_home, version=version)

    # The (small, ~MB) index JSON ships separately from the audio. Fetch ONLY it if absent
    # (partial_download=["index"] never pulls the 2-100 GB audio tarball).
    if not Path(dataset.index_path).exists():
        print("=> Fetching Slakh index (small JSON, not audio) ...", file=sys.stderr)
        try:
            dataset.download(partial_download=["index"])
        except Exception as exc:  # noqa: BLE001 - offline / unreachable index host
            print(f"=> Could not fetch the Slakh index: {exc}", file=sys.stderr)
            print(_BABYSLAKH_HELP if args.subset == "babyslakh" else _FULL_HELP, file=sys.stderr)
            sys.exit(1)

    # Never trigger the giant audio download; just check the data is already present. Probe a
    # sample of tracks, not only the first — a partial local copy (e.g. a truncated tarball
    # stream) can miss the first indexed track while holding hundreds of complete ones.
    mtrack_ids = list(dataset.mtrack_ids)
    data_present = False
    for mid in mtrack_ids[:: max(1, len(mtrack_ids) // 50)]:
        try:
            probe = dataset.multitrack(mid)
            if probe.mix_path and Path(probe.mix_path).exists():
                data_present = True
                break
        except Exception:  # noqa: BLE001 - missing files surface as load errors
            continue
    if not data_present:
        print(f"=> Slakh {args.subset} data not found at data_home={dataset.data_home}",
              file=sys.stderr)
        print(_BABYSLAKH_HELP if args.subset == "babyslakh" else _FULL_HELP, file=sys.stderr)
        sys.exit(1)

    # Grouped stems/MIDI land in a per-subset dir: babyslakh keeps the original flat layout
    # (existing manifests stay valid); the full redux uses its own subdir because it shares
    # Track IDs with babyslakh and would otherwise overwrite its grouped files.
    out_root = DATA_DIR if args.subset == "babyslakh" else DATA_DIR / "redux"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows, n_dropped = [], 0
    for mid in mtrack_ids:
        mtrack = dataset.multitrack(mid)
        # babyslakh has split == None; only filter when the subset actually carries splits.
        keep_all = args.split in ("", "all")
        try:
            if args.subset == "full" and not keep_all and mtrack.split != args.split:
                continue
            row = build_mtrack(mtrack, out_root)
        except Exception as exc:  # noqa: BLE001 - partial local copies lack metadata/stem files
            print(f"   [drop] {mid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            row = None
        if row is None:
            n_dropped += 1
            continue
        rows.append(row)
        present = ",".join(k for k, v in row["stems"].items() if v) or "none"
        print(f"   [ok] {mid}: stems={present}", file=sys.stderr)
        if args.limit and len(rows) >= args.limit:
            break

    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"=> Dropped {n_dropped} multitracks (missing audio)", file=sys.stderr)
    print(f"=> Wrote {len(rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
