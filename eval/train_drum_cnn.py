#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "mirdata>=0.3.8",
#   "librosa>=0.10",
#   "numpy>=1.26,<2.3",
#   "pretty_midi>=0.2.10",
#   "soundfile>=0.12",
#   "torch==2.8.*",
#   "tqdm>=4.66",
# ]
# ///
"""D1: train an MIT-licensed drum-transcription model (pre-registered in
paper/EXPERIMENTS.md, commit 64002a1, BEFORE any training) to replace the shipped
adtof-pytorch weights (upstream CC BY-NC-SA originals, unlicensed port).

Corpus (all CC BY 4.0, train-side splits only): E-GMD train split (human e-kit
performances with velocity) + Slakh2100-redux train-split drum stems, both oracle
(sum of ground-truth drum stems) and separator-processed (mix -> the SHIPPED SCNet
XL via src/jams/data/stems_worker.py -> separated drum stem, MIDI labels unchanged).
The committed eval protocols — Slakh redux TEST split (n=151) and the E-GMD TEST
split (the n=500 eval subset is its first 500 rows in CSV order) — are hard-excluded
by id in `acquire`, with counts logged and asserted. StemGMD is deferred this phase
(ledger note owned by the coordinator).

Targets: 5 onset classes (kick / snare / hi-hat / tom / cymbal) + per-onset velocity
(0..1, from MIDI velocity/127; a product win over ADTOF's flat 100 — NOT part of the
gate). The class vocabulary and pitch maps mirror src/jams/analysis/gm.py (source of
truth; copied here because this uv script cannot import jams): class representatives
[36, 38, 42, 47, 49] — eval-side reduce_drum_pitch_5 lands ADTOF's 35/38/42/47/49
output on the same buckets, so the paired gate compares like for like.

Features: log1p-mel, SR 22050, N_FFT 1024 (~46 ms window), HOP 220 (~10 ms — matches
the frame-target hop), 96 mels, float16 .npy per track (same conventions as
eval/train_tempo_cnn.py). Labels: per-track event lists [time_s, class_idx, velocity]
in labels.json; rasterized to frame targets at train time.

Stages:
  acquire   E-GMD (extracted dir) + Slakh redux (mirdata) -> labels.json (+ oracle
            drum-stem flacs). Test-split ids excluded and ASSERTED.
  separate  drive stems_worker.py --serve over Slakh train/val mixes -> separated
            drum stems (mono flac), added to labels.json as sep.* entries.
  features  parallel log-mel extraction -> features/<tid>.npy (skips existing).
  train     SKELETON this phase (CRNN + dataset + loop wired; finalized and ledgered
            in the training phase — selection on validation splits only).
  infer     stub; finalized alongside the one-shot gate phase.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SR = 22050
N_FFT = 1024
HOP = 220                 # ~9.98 ms -> ~100.2 frames/s ("10 ms hop")
N_MELS = 96
SEED = 0

# --- 5-class drum vocabulary (mirrors src/jams/analysis/gm.py — source of truth) ----
GM_KICK, GM_SNARE, GM_HAT, GM_TOM, GM_CYM = 36, 38, 42, 47, 49
CLASSES = [GM_KICK, GM_SNARE, GM_HAT, GM_TOM, GM_CYM]   # class idx 0..4
CLASS_NAMES = ["kick", "snare", "hihat", "tom", "cymbal"]

# GM / E-GMD (Roland TD-17, incl. 22/26 hat articulations) pitch -> class idx.
# Superset of gm.DRUM_PITCH_CANON∘DRUM_5CLASS; 39/54/56/75-82 percussion left unmapped
# on purpose (claps/tambourine/cowbell etc. are not in the 5-class gate vocabulary).
PITCH_TO_CLASS = {
    35: 0, 36: 0,
    37: 1, 38: 1, 40: 1,
    22: 2, 26: 2, 42: 2, 44: 2, 46: 2,
    41: 3, 43: 3, 45: 3, 47: 3, 48: 3, 50: 3, 58: 3,
    49: 4, 51: 4, 52: 4, 53: 4, 55: 4, 57: 4, 59: 4,
}


def _json_load(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _json_dump(obj: dict, path: Path, **kw) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, **kw)


def sanitize(tid: str) -> str:
    return tid.replace("/", "__")


# ------------------------------------------------------------------ label extraction
def midi_drum_events(midi_path: str | Path) -> list[list[float]]:
    """[time_s, class_idx, velocity 0..1] for every mapped drum note, onset-sorted."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    events = []
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            cls = PITCH_TO_CLASS.get(int(n.pitch))
            if cls is None:
                continue
            events.append([round(float(n.start), 5), cls, round(n.velocity / 127.0, 4)])
    events.sort(key=lambda e: e[0])
    return events


# --------------------------------------------------------------------------- acquire
def _acquire_egmd(egmd_home: Path, limit: int) -> tuple[dict, dict]:
    """E-GMD train+validation rows from an extracted e-gmd dir. Returns (entries, stats)."""
    import csv as csvmod

    csv_path = egmd_home / "e-gmd-v1.0.0.csv"
    if not csv_path.exists():
        sys.exit(f"missing {csv_path} — not an extracted E-GMD dir")
    rows = list(csvmod.DictReader(csv_path.read_text().splitlines()))
    test_ids = {str(r.get("id") or Path(r["audio_filename"]).stem)
                for r in rows if r.get("split") == "test"}
    entries: dict = {}
    kept = {"train": 0, "validation": 0}
    dropped = 0
    for r in rows:
        split = r.get("split")
        if split not in ("train", "validation"):
            continue
        if limit and kept["train"] + kept["validation"] >= limit:
            break
        af, mf = r.get("audio_filename"), r.get("midi_filename")
        tid = str(r.get("id") or (Path(af).stem if af else ""))
        if not af or not mf or not (egmd_home / af).exists() or not (egmd_home / mf).exists():
            dropped += 1
            continue
        events = midi_drum_events(egmd_home / mf)
        if not events:
            dropped += 1
            continue
        entries[f"egmd.{sanitize(tid)}"] = {
            "source": "egmd", "split": split,
            "audio": str(egmd_home / af), "events": events,
        }
        kept[split] += 1
    overlap = {k for k in entries if k.split(".", 1)[1] in {sanitize(t) for t in test_ids}}
    assert not overlap, f"E-GMD test-split leakage: {sorted(overlap)[:5]}"
    stats = {"egmd_train": kept["train"], "egmd_val": kept["validation"],
             "egmd_dropped": dropped, "egmd_test_excluded": len(test_ids)}
    return entries, stats


def _acquire_slakh(slakh_home: Path, out_root: Path, limit: int) -> tuple[dict, dict]:
    """Slakh redux train+validation drum groups: oracle mono flac + events + mix path."""
    import mirdata
    import soundfile as sf

    ds = mirdata.initialize("slakh", data_home=str(slakh_home), version="2100-redux")
    if not Path(ds.index_path).exists():
        ds.download(partial_download=["index"])
    entries: dict = {}
    kept = {"train": 0, "validation": 0}
    dropped = 0
    test_ids = []
    for mid in ds.mtrack_ids:
        try:
            mt = ds.multitrack(mid)
            split = mt.split
        except Exception:  # noqa: BLE001 — split-only local copies lack some metadata
            continue
        if split == "test":
            test_ids.append(mid)
            continue
        if split not in ("train", "validation"):
            continue
        if limit and kept["train"] + kept["validation"] >= limit:
            continue
        try:
            drum_stems = [s for s in mt.tracks.values()
                          if s.is_drum and s.audio_path and Path(s.audio_path).exists()]
            if not drum_stems or not mt.mix_path or not Path(mt.mix_path).exists():
                dropped += 1
                continue
            # Oracle drum stem: mono sum of the ground-truth drum stems -> flac.
            mix = None
            sr = None
            for s in drum_stems:
                data, sr = sf.read(s.audio_path, always_2d=True)
                mono = data.mean(axis=1)
                if mix is None:
                    mix = mono.copy()
                else:
                    n = min(len(mix), len(mono))
                    mix = mix[:n] + mono[:n]
            events = []
            for s in drum_stems:
                if s.midi_path and Path(s.midi_path).exists():
                    try:
                        events.extend(midi_drum_events(s.midi_path))
                    except Exception as exc:  # noqa: BLE001
                        print(f"   [warn] {mid}: bad stem MIDI {s.midi_path}: {exc}",
                              file=sys.stderr)
            events.sort(key=lambda e: e[0])
            if not events:
                dropped += 1
                continue
            wav_out = out_root / "oracle" / f"{mid}.flac"
            wav_out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav_out, mix.astype(np.float32), sr)
            entries[f"slakh.{mid}"] = {
                "source": "slakh_oracle", "split": split,
                "audio": str(wav_out), "mix": str(mt.mix_path), "events": events,
            }
            kept[split] += 1
        except Exception as exc:  # noqa: BLE001 — log and count, never silent
            dropped += 1
            print(f"   [drop] {mid}: {type(exc).__name__}: {exc}", file=sys.stderr)
    overlap = {k for k in entries if k.split(".", 1)[1] in set(test_ids)}
    assert not overlap, f"Slakh test-split leakage: {sorted(overlap)[:5]}"
    stats = {"slakh_train": kept["train"], "slakh_val": kept["validation"],
             "slakh_dropped": dropped, "slakh_test_excluded": len(test_ids)}
    return entries, stats


def cmd_acquire(args) -> None:
    args.data_home.mkdir(parents=True, exist_ok=True)
    entries: dict = {}
    stats: dict = {}
    if args.egmd_home:
        e, s = _acquire_egmd(args.egmd_home, args.egmd_limit)
        entries.update(e)
        stats.update(s)
    if args.slakh_home:
        e, s = _acquire_slakh(args.slakh_home, args.data_home, args.slakh_limit)
        entries.update(e)
        stats.update(s)
    if not entries:
        sys.exit("acquire produced no entries (pass --egmd-home and/or --slakh-home)")
    _json_dump(entries, args.data_home / "labels.json")
    _json_dump(stats, args.data_home / "acquire_stats.json", indent=1)
    print(json.dumps(stats, indent=1))
    print(f"total usable: {len(entries)} -> {args.data_home}/labels.json")


# -------------------------------------------------------------------------- separate
def cmd_separate(args) -> None:
    """Separated-domain training pairs: Slakh mix -> shipped separator -> drum stem.

    Drives src/jams/data/stems_worker.py in --serve mode (JSONL stdin/stdout), i.e.
    the EXACT separator the product ships (SCNet XL IHF by default). MIDI labels are
    reused unchanged from the oracle entry. Output converted to mono flac.
    """
    import shutil
    import time

    import soundfile as sf

    labels = _json_load(args.data_home / "labels.json")
    todo = [(tid, rec) for tid, rec in labels.items()
            if rec["source"] == "slakh_oracle" and rec.get("mix")
            and f"sep.{tid}" not in labels]
    # Validation mixes first (selection needs separated-domain val coverage), then train.
    todo.sort(key=lambda kv: (kv[1]["split"] != "validation", kv[0]))
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to separate")
        return
    print(f"separating {len(todo)} mixes with {args.model} via {args.worker}")

    proc = subprocess.Popen(  # noqa: S603 — our own worker script
        [args.uv, "run", "--script", str(args.worker), "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
    )
    sep_root = args.data_home / "sep"
    sep_root.mkdir(parents=True, exist_ok=True)
    added, failed, t0 = 0, 0, time.time()
    try:
        for i, (tid, rec) in enumerate(todo):
            work = sep_root / "_work"
            shutil.rmtree(work, ignore_errors=True)
            req = {"audio": rec["mix"], "out_dir": str(work),
                   "model": args.model, "transcribe": False}
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            res = json.loads(proc.stdout.readline())
            if not res.get("ok"):
                failed += 1
                print(f"   [fail] {tid}: {res.get('error')}", file=sys.stderr)
                continue
            drums = next((s["audio_path"] for s in res["result"]["stems"]
                          if s["stem_type"] == "drums"), None)
            if not drums or not Path(drums).exists():
                failed += 1
                print(f"   [fail] {tid}: no drums stem in result", file=sys.stderr)
                continue
            data, sr = sf.read(drums, always_2d=True)
            out = sep_root / f"{tid.split('.', 1)[1]}.flac"
            sf.write(out, data.mean(axis=1).astype(np.float32), sr)
            labels[f"sep.{tid}"] = {
                "source": "slakh_sep", "split": rec["split"],
                "audio": str(out), "events": rec["events"],
            }
            added += 1
            if i == 2:
                per = (time.time() - t0) / 3
                print(f"   pace: {per:.1f}s/mix -> est {per * len(todo) / 3600:.1f} h total",
                      flush=True)
            if added % 25 == 0:
                _json_dump(labels, args.data_home / "labels.json")
                print(f"   {added}/{len(todo)} done", flush=True)
    finally:
        shutil.rmtree(sep_root / "_work", ignore_errors=True)
        proc.stdin.close()
        proc.terminate()
    _json_dump(labels, args.data_home / "labels.json")
    print(f"separate: {added} added, {failed} failed")
    if failed and failed >= added:
        raise SystemExit("separation mostly failing — investigate before training")


# -------------------------------------------------------------------------- features
def _feature_one(task: tuple[str, str, str]) -> tuple[str, str | None]:
    tid, audio, out = task
    try:
        import librosa

        y, _ = librosa.load(audio, sr=SR, mono=True)
        m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                           n_mels=N_MELS)
        np.save(out, np.log1p(m).astype(np.float16))
        return tid, None
    except Exception as exc:  # noqa: BLE001 — collected and reported by the caller
        return tid, f"{type(exc).__name__}: {exc}"


def cmd_features(args) -> None:
    labels = _json_load(args.data_home / "labels.json")
    fdir = args.data_home / "features"
    fdir.mkdir(exist_ok=True)
    tasks = []
    for tid, rec in labels.items():
        out = fdir / f"{sanitize(tid)}.npy"
        if not out.exists():
            tasks.append((tid, rec["audio"], str(out)))
    print(f"features: {len(tasks)} to extract ({len(labels) - len(tasks)} already present)")
    fails = []
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_feature_one, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            tid, err = fut.result()
            if err:
                fails.append(tid)
                print(f"FAIL {tid}: {err}", file=sys.stderr)
    if fails:
        raise SystemExit(f"{len(fails)} feature failures — resolve before training")
    print(f"features complete -> {fdir}")


# ---------------------------------------------------------------- model (SKELETON)
def build_model():
    """CRNN onset+velocity model — SKELETON; finalized (and ledgered) in the training
    phase. Conv over (mel, time) -> BiGRU over time -> per-frame 5x onset sigmoid +
    5x velocity."""
    import torch.nn as nn

    class CRNN(nn.Module):
        def __init__(self, n_mels=N_MELS, hidden=64):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ELU(),
                nn.MaxPool2d((3, 1)),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(),
                nn.MaxPool2d((3, 1)),
                nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ELU(),
                nn.MaxPool2d((2, 1)),
                nn.Dropout2d(0.1),
            )
            feat = 48 * (n_mels // 3 // 3 // 2)
            self.rnn = nn.GRU(feat, hidden, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.1)
            self.onset = nn.Linear(2 * hidden, len(CLASSES))
            self.velocity = nn.Linear(2 * hidden, len(CLASSES))

        def forward(self, x):            # x: (B, 1, mels, T)
            h = self.conv(x)             # (B, C, mels', T)
            h = h.permute(0, 3, 1, 2).flatten(2)   # (B, T, C*mels')
            h, _ = self.rnn(h)
            return self.onset(h), self.velocity(h)  # (B, T, 5) logits, (B, T, 5)

    return CRNN()


def rasterize(events: list[list[float]], n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Events -> (onset_target, velocity_target) at HOP frames; neighbors get 0.5."""
    fps = SR / HOP
    on = np.zeros((n_frames, len(CLASSES)), dtype=np.float32)
    vel = np.zeros((n_frames, len(CLASSES)), dtype=np.float32)
    for t, cls, v in events:
        f = int(round(t * fps))
        if not 0 <= f < n_frames:
            continue
        c = int(cls)
        on[f, c] = 1.0
        vel[f, c] = max(vel[f, c], v)
        for nb in (f - 1, f + 1):
            if 0 <= nb < n_frames:
                on[nb, c] = max(on[nb, c], 0.5)
    return on, vel


def cmd_train(args) -> None:  # noqa: ARG001 — wired next phase
    sys.exit("train is a SKELETON this phase — finalized and pre-reg-ledgered in the "
             "training phase (selection on validation splits only; see EXPERIMENTS.md D1)")


def cmd_infer(args) -> None:  # noqa: ARG001
    sys.exit("infer lands with the one-shot gate phase (see EXPERIMENTS.md D1)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("acquire", "separate", "features", "train", "infer"):
        p = sub.add_parser(name)
        p.add_argument("--data-home", type=Path, default=Path("eval/data/drumtrain"))
        if name == "acquire":
            p.add_argument("--egmd-home", type=Path, default=None,
                           help="extracted e-gmd-v1.0.0 dir")
            p.add_argument("--slakh-home", type=Path, default=None,
                           help="parent of slakh2100_flac_redux/")
            p.add_argument("--egmd-limit", type=int, default=0)
            p.add_argument("--slakh-limit", type=int, default=0)
        if name == "separate":
            p.add_argument("--worker", type=Path, required=True,
                           help="path to src/jams/data/stems_worker.py")
            p.add_argument("--model", default="scnet_xl_ihf")
            p.add_argument("--uv", default="uv")
            p.add_argument("--limit", type=int, default=0)
        if name == "features":
            p.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    {"acquire": cmd_acquire, "separate": cmd_separate, "features": cmd_features,
     "train": cmd_train, "infer": cmd_infer}[args.cmd](args)


if __name__ == "__main__":
    main()
