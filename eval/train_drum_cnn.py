#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "mirdata>=0.3.8",
#   "mir_eval>=0.7",
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
  train     ledgered recipe (EXPERIMENTS.md D1 2026-07-14): CRNN 32/64/96 conv blocks
            (freq-pool 2 each) -> BiGRU 2x128 -> 5-class onset + velocity heads;
            Adam 1e-3, batch 48 x 10 s crops, gain/SpecAugment aug, patience 15 on
            the selection metric = mean macro onset-F(50 ms, mir_eval matching) over
            {egmd_val, slakh_val_oracle, slakh_val_sep}; per-class thresholds
            grid-searched on validation (coarse per epoch, fine on the best ckpt).
  infer     audio -> [[time_s, gm_pitch, velocity_0_127], ...] JSONL, using best.pt
            + the thresholds chosen on validation.
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
EGMD_TRAIN_TARGET = 10_000   # stratified subset size (ledgered 2026-07-14)
EGMD_VAL_CAP = 1_500


def _stratified_rows(rows: list[dict], target: int, rng) -> list[dict]:
    """Round-robin over (kit, style) groups until `target` rows: every group
    contributes before any group contributes twice, so all kits appear; row order
    within a group is a seeded shuffle (deterministic)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r.get("kit_name") or "?", r.get("style") or "?")
        groups.setdefault(key, []).append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r["audio_filename"])
        rng.shuffle(groups[key])
    ordered = sorted(groups)
    picked: list[dict] = []
    rank = 0
    while len(picked) < target:
        took = 0
        for key in ordered:
            if rank < len(groups[key]):
                picked.append(groups[key][rank])
                took += 1
                if len(picked) >= target:
                    break
        if not took:
            break
        rank += 1
    return picked


def _acquire_egmd(egmd_home: Path, limit: int) -> tuple[dict, dict]:
    """E-GMD train+validation rows from an extracted e-gmd dir. Returns (entries, stats).

    Keys are the full relative audio path (drummer/session/file, kit suffix included) —
    E-GMD's `id` column repeats across the 43 kits, which silently collapsed the corpus
    to ~936 tracks in phase 1 (ledgered 2026-07-14). Train rows are a seeded stratified
    subset (~EGMD_TRAIN_TARGET across kit×style so all kits appear); validation rows are
    kept up to EGMD_VAL_CAP with the same stratification.
    """
    import csv as csvmod
    import random as pyrandom

    csv_path = egmd_home / "e-gmd-v1.0.0.csv"
    if not csv_path.exists():
        sys.exit(f"missing {csv_path} — not an extracted E-GMD dir")
    rows = list(csvmod.DictReader(csv_path.read_text().splitlines()))
    test_paths = {r["audio_filename"] for r in rows
                  if r.get("split") == "test" and r.get("audio_filename")}

    eligible: dict[str, list[dict]] = {"train": [], "validation": []}
    dropped = 0
    for r in rows:
        split = r.get("split")
        if split not in ("train", "validation"):
            continue
        af, mf = r.get("audio_filename"), r.get("midi_filename")
        if not af or not mf or not (egmd_home / af).exists() or not (egmd_home / mf).exists():
            dropped += 1
            continue
        eligible[split].append(r)

    rng = pyrandom.Random(SEED)
    train_target = limit or EGMD_TRAIN_TARGET
    sel = (_stratified_rows(eligible["train"], train_target, rng)
           + _stratified_rows(eligible["validation"], EGMD_VAL_CAP, rng))

    entries: dict = {}
    kept = {"train": 0, "validation": 0}
    hours = 0.0
    kits: dict[str, int] = {}
    for r in sel:
        af, mf = r["audio_filename"], r["midi_filename"]
        assert af not in test_paths, f"E-GMD test-split leakage: {af}"
        events = midi_drum_events(egmd_home / mf)
        if not events:
            dropped += 1
            continue
        key = f"egmd.{sanitize(Path(af).with_suffix('').as_posix())}"
        entries[key] = {
            "source": "egmd", "split": r["split"],
            "audio": str(egmd_home / af), "events": events,
        }
        kept[r["split"]] += 1
        hours += float(r.get("duration") or 0.0) / 3600.0
        kit = r.get("kit_name") or "?"
        kits[kit] = kits.get(kit, 0) + 1
    # Collision guard: uniqueness must hold row-for-row (phase-1 bug regression check).
    assert len(entries) == kept["train"] + kept["validation"], (
        f"E-GMD key collision: {len(entries)} entries != {kept} kept")
    stats = {"egmd_train": kept["train"], "egmd_val": kept["validation"],
             "egmd_dropped": dropped, "egmd_test_excluded": len(test_paths),
             "egmd_pool_train": len(eligible["train"]),
             "egmd_pool_val": len(eligible["validation"]),
             "egmd_hours": round(hours, 1), "egmd_kits": len(kits),
             "egmd_per_kit": dict(sorted(kits.items()))}
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
    """Per-source re-acquire: existing labels.json entries survive unless their source
    is being re-acquired (egmd.* replaced when --egmd-home given; slakh.* when
    --slakh-home). sep.* entries are owned by `separate` and never touched here."""
    args.data_home.mkdir(parents=True, exist_ok=True)
    labels_path = args.data_home / "labels.json"
    stats_path = args.data_home / "acquire_stats.json"
    entries: dict = _json_load(labels_path) if labels_path.exists() else {}
    stats: dict = _json_load(stats_path) if stats_path.exists() else {}
    if args.egmd_home:
        entries = {k: v for k, v in entries.items() if not k.startswith("egmd.")}
        e, s = _acquire_egmd(args.egmd_home, args.egmd_limit)
        entries.update(e)
        stats.update(s)
    if args.slakh_home:
        entries = {k: v for k, v in entries.items() if not k.startswith("slakh.")}
        e, s = _acquire_slakh(args.slakh_home, args.data_home, args.slakh_limit)
        entries.update(e)
        stats.update(s)
    if not entries:
        sys.exit("acquire produced no entries (pass --egmd-home and/or --slakh-home)")
    _json_dump(entries, labels_path)
    _json_dump(stats, stats_path, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "egmd_per_kit"}, indent=1))
    print(f"total usable: {len(entries)} -> {labels_path}")


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


# ------------------------------------------------------------------------------ model
def build_model():
    """CRNN per the ledgered recipe (EXPERIMENTS.md D1, 2026-07-14): 3 conv blocks
    (32/64/96, two 3x3 convs each + BN + ReLU, freq-pool 2) -> freq-flatten ->
    2-layer BiGRU 128 -> 5-class onset logits + 5-class velocity head."""
    import torch.nn as nn

    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d((2, 1))]

    class CRNN(nn.Module):
        def __init__(self, n_mels=N_MELS, hidden=128):
            super().__init__()
            self.conv = nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 96))
            feat = 96 * (n_mels // 8)                      # 96 mels -> 12 bands
            self.rnn = nn.GRU(feat, hidden, num_layers=2, batch_first=True,
                              bidirectional=True)
            self.onset = nn.Linear(2 * hidden, len(CLASSES))
            self.velocity = nn.Linear(2 * hidden, len(CLASSES))

        def forward(self, x):            # x: (B, 1, mels, T)
            h = self.conv(x)             # (B, 96, mels/8, T)
            h = h.permute(0, 3, 1, 2).flatten(2)   # (B, T, 96*mels/8)
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


CROP_FRAMES = 1000           # 10 s at ~100 fps
FPS = SR / HOP


class DrumDataset:
    """One random 10 s crop per track per epoch. Gain aug ±6 dB (exact, in the power
    domain via expm1/log1p) + light SpecAugment (<=2 freq masks <=8 bins, <=1 time
    mask <=20 frames). Raw log1p-mel goes to the model — the first BatchNorm adapts,
    and per-crop normalization would cancel the gain augmentation."""

    def __init__(self, items, fdir, train: bool):
        self.items, self.fdir, self.train = items, fdir, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import random as pyrandom

        import torch

        tid, events = self.items[i]
        X = np.load(self.fdir / f"{sanitize(tid)}.npy").astype(np.float32)
        T = X.shape[1]
        s = pyrandom.randint(0, T - CROP_FRAMES) if self.train and T > CROP_FRAMES else 0
        X = X[:, s:s + CROP_FRAMES]
        if X.shape[1] < CROP_FRAMES:
            X = np.pad(X, ((0, 0), (0, CROP_FRAMES - X.shape[1])))
        t0 = s / FPS
        local = [[t - t0, c, v] for t, c, v in events
                 if t0 <= t < t0 + CROP_FRAMES / FPS]
        on, vel = rasterize(local, CROP_FRAMES)
        if self.train:
            db = pyrandom.uniform(-6.0, 6.0)
            X = np.log1p(np.expm1(X) * 10.0 ** (db / 10.0))   # power-domain gain
            for _ in range(pyrandom.randint(0, 2)):           # freq masks
                w = pyrandom.randint(1, 8)
                f0 = pyrandom.randint(0, N_MELS - w)
                X[f0:f0 + w, :] = 0.0
            if pyrandom.random() < 0.5:                       # time mask
                w = pyrandom.randint(1, 20)
                s0 = pyrandom.randint(0, CROP_FRAMES - w)
                X[:, s0:s0 + w] = 0.0
        return (torch.from_numpy(X)[None], torch.from_numpy(on), torch.from_numpy(vel))


def _local_max_mask(p: np.ndarray) -> np.ndarray:
    """Vectorized: True where p[f] is the max of its ±2-frame window."""
    m = p.copy()
    for shift in (-2, -1, 1, 2):
        s = np.roll(p, shift)
        if shift > 0:
            s[:shift] = -1.0
        else:
            s[shift:] = -1.0
        m = np.maximum(m, s)
    return p >= m - 1e-9


def _pick_events(prob: np.ndarray, vel: np.ndarray, thresholds: np.ndarray
                 ) -> list[list[float]]:
    """Per-class peak-picking: threshold + local max over ±2 frames + 50 ms min gap.
    Returns [time_s, class_idx, velocity 0..1] sorted by time."""
    events: list[list[float]] = []
    min_gap = int(round(0.05 * FPS))
    for c in range(prob.shape[1]):
        p = prob[:, c]
        peaks = np.where(_local_max_mask(p) & (p >= thresholds[c]))[0]
        last = -10_000
        for f in peaks:
            if f - last < min_gap:
                continue
            last = f
            events.append([f / FPS, c, float(np.clip(vel[f, c], 0.0, 1.0))])
    events.sort(key=lambda e: e[0])
    return events


def _predict_probs(model, X: np.ndarray, dev, chunk: int = 8000, overlap: int = 200):
    """Full-track frame probabilities + velocities, chunked with center-stitching."""
    import torch

    T = X.shape[1]
    on = np.zeros((T, len(CLASSES)), dtype=np.float32)
    vl = np.zeros((T, len(CLASSES)), dtype=np.float32)
    s = 0
    with torch.no_grad():
        while s < T:
            e = min(T, s + chunk)
            xw = torch.from_numpy(X[:, s:e].astype(np.float32))[None, None].to(dev)
            lo_t, ve_t = model(xw)
            po = torch.sigmoid(lo_t)[0].cpu().numpy()
            pv = ve_t[0].cpu().numpy()
            a = s + (overlap if s > 0 else 0)
            on[a:e] = po[a - s:e - s]
            vl[a:e] = pv[a - s:e - s]
            if e == T:
                break
            s = e - 2 * overlap
    return on, vl


THRESH_GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)     # final search (best ckpt)
EPOCH_GRID = np.array([0.20, 0.35, 0.50, 0.65])            # coarse per-epoch selection


def _f_stats(ref: list, est: list, window: float = 0.05) -> tuple[int, int, int]:
    """mir_eval-equivalent onset matching (maximum bipartite via mir_eval)."""
    import mir_eval

    if not ref and not est:
        return 0, 0, 0
    if not ref:
        return 0, len(est), 0
    if not est:
        return 0, 0, len(ref)
    matches = mir_eval.util.match_events(np.array(ref), np.array(est), window)
    tp = len(matches)
    return tp, len(est) - tp, len(ref) - tp


def _evaluate_pools(model, pools: dict, fdir: Path, dev,
                    thresholds: np.ndarray | None = None,
                    grid: np.ndarray | None = None):
    """Macro onset-F per pool at the best (or given) per-class thresholds.

    Accumulates TP/FP/FN per (pool, class, threshold) over full-track predictions,
    then picks per-class thresholds maximizing GLOBAL (all-pool) F when not given.
    Returns (selection_metric, report_dict, chosen_thresholds).
    """
    model.eval()
    grid = (EPOCH_GRID if grid is None else grid) if thresholds is None else None
    n_th = len(grid) if grid is not None else 1
    stats = {p: np.zeros((n_th, len(CLASSES), 3)) for p in pools}   # tp, fp, fn
    vel_err, vel_n = 0.0, 0
    for pool, items in pools.items():
        for tid, events in items:
            X = np.load(fdir / f"{sanitize(tid)}.npy").astype(np.float32)
            prob, vel = _predict_probs(model, X, dev)
            ref_by_c = {c: [e[0] for e in events if e[1] == c] for c in range(5)}
            for ti in range(n_th):
                th = (grid[[ti] * 5] if grid is not None else thresholds)
                est = _pick_events(prob, vel, th)
                est_by_c: dict[int, list[float]] = {c: [] for c in range(5)}
                for t, c, _v in est:
                    est_by_c[int(c)].append(t)
                for c in range(5):
                    stats[pool][ti, c] += _f_stats(ref_by_c[c], est_by_c[c])
            # Velocity MAE at matched reference onsets (threshold-independent enough:
            # read the velocity head at the reference frame).
            for t, c, v in events:
                f = int(round(t * FPS))
                if 0 <= f < vel.shape[0]:
                    vel_err += abs(float(vel[f, int(c)]) - v)
                    vel_n += 1

    def f_of(row) -> float:
        tp, fp, fn = row
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    if grid is not None:
        # Per-class threshold = argmax of pooled-over-all-pools F for that class.
        total = sum(stats.values())                       # (n_th, 5, 3)
        best_ti = [int(np.argmax([f_of(total[ti, c]) for ti in range(n_th)]))
                   for c in range(5)]
        chosen = np.array([grid[i] for i in best_ti])
    else:
        best_ti = [0] * 5
        chosen = thresholds
    report: dict = {"pools": {}, "thresholds": [float(t) for t in chosen]}
    per_pool_macro = []
    for pool in pools:
        per_class = [f_of(stats[pool][best_ti[c], c]) for c in range(5)]
        macro = float(np.mean(per_class))
        per_pool_macro.append(macro)
        report["pools"][pool] = {
            "macro_f": round(macro, 4),
            "per_class_f": {CLASS_NAMES[c]: round(per_class[c], 4) for c in range(5)},
            "n_tracks": len(pools[pool]),
        }
    report["velocity_mae"] = round(vel_err / vel_n, 4) if vel_n else None
    sel = float(np.mean(per_pool_macro))
    report["selection_metric"] = round(sel, 4)
    return sel, report, chosen


VAL_POOLS = {"egmd_val": ("egmd",), "slakh_val_oracle": ("slakh_oracle",),
             "slakh_val_sep": ("slakh_sep",)}


def _build_pools(labels: dict) -> dict:
    pools: dict = {p: [] for p in VAL_POOLS}
    for tid, rec in labels.items():
        if rec["split"] != "validation":
            continue
        for pool, sources in VAL_POOLS.items():
            if rec["source"] in sources:
                pools[pool].append((tid, rec["events"]))
    return pools


def cmd_train(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    import random as pyrandom

    pyrandom.seed(SEED)
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={dev}")

    labels = _json_load(args.data_home / "labels.json")
    fdir = args.data_home / "features"
    train_items = [(tid, rec["events"]) for tid, rec in labels.items()
                   if rec["split"] == "train"]
    pools = _build_pools(labels)
    print(f"train tracks: {len(train_items)}; val pools: "
          f"{ {p: len(v) for p, v in pools.items()} }")
    for pool, items in pools.items():
        if not items:
            sys.exit(f"validation pool {pool} is empty — selection metric undefined")

    # Per-class pos_weight from event counts vs total frames (widened x3), capped.
    counts = np.zeros(len(CLASSES))
    frames = 0.0
    for _tid, events in train_items:
        for _t, c, _v in events:
            counts[int(c)] += 1
        frames += (max(e[0] for e in events) if events else 0) * FPS
    pos_w = np.clip(frames / np.maximum(counts * 3.0, 1.0), 1.0, 30.0)
    print(f"pos_weight: { {CLASS_NAMES[c]: round(float(pos_w[c]), 1) for c in range(5)} }")

    dl = DataLoader(DrumDataset(train_items, fdir, True), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    model = build_model().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params: {n_par}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pw_t = torch.tensor(pos_w, dtype=torch.float32, device=dev)
    best = (0.0, 0)
    hist = []
    args.out.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = n = 0
        for X, on, vel in dl:
            X, on, vel = X.to(dev), on.to(dev), vel.to(dev)
            lo, ve = model(X)
            loss_on = torch.nn.functional.binary_cross_entropy_with_logits(
                lo, on, pos_weight=pw_t)
            mask = (on >= 1.0).float()
            loss_vel = (((ve - vel) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss_on + 0.5 * loss_vel
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * X.shape[0]
            n += X.shape[0]
        sel, report, chosen = _evaluate_pools(model, pools, fdir, dev)
        hist.append({"epoch": ep, "train_loss": tot / n, "selection": sel,
                     "report": report})
        print(f"ep{ep:02d} loss={tot / n:.4f} sel={sel:.4f} "
              f"pools={ {p: report['pools'][p]['macro_f'] for p in report['pools']} }",
              flush=True)
        _json_dump({"history": hist}, args.out / "history.json", indent=1)
        if sel > best[0]:
            best = (sel, ep)
            torch.save(model.state_dict(), args.out / "best.pt")
            _json_dump(report, args.out / "val_report.json", indent=1)
        elif ep - best[1] >= args.patience:
            print(f"early stop at ep{ep} (best sel={best[0]:.4f} @ ep{best[1]})")
            break
    # Final fine threshold search on the selected checkpoint (ledger: grid-searched).
    model.load_state_dict(torch.load(args.out / "best.pt", map_location="cpu"))
    model.to(dev)
    sel, report, chosen = _evaluate_pools(model, pools, fdir, dev, grid=THRESH_GRID)
    report["best_epoch"] = best[1]
    report["coarse_selection_at_best"] = best[0]
    _json_dump(report, args.out / "val_report.json", indent=1)
    print(f"train done: coarse-best sel={best[0]:.4f} @ ep{best[1]}; "
          f"fine-grid sel={sel:.4f}, thresholds={[float(t) for t in chosen]} -> "
          f"{args.out}/best.pt + val_report.json")


def cmd_infer(args) -> None:
    """Audio file(s) -> per-track drum events using best.pt + chosen thresholds.
    Output: JSONL {track_id, events: [[time_s, gm_pitch, velocity_0_127], ...]}."""
    import librosa
    import torch

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(dev).eval()
    report = _json_load(Path(args.model).parent / "val_report.json")
    thresholds = np.array(report["thresholds"])
    out = open(args.out, "w")  # noqa: SIM115
    for audio in args.audio:
        y, _ = librosa.load(audio, sr=SR, mono=True)
        m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                           n_mels=N_MELS)
        X = np.log1p(m).astype(np.float32)
        prob, vel = _predict_probs(model, X, dev)
        events = [[round(t, 4), CLASSES[int(c)], int(round(v * 127))]
                  for t, c, v in _pick_events(prob, vel, thresholds)]
        out.write(json.dumps({"track_id": Path(audio).stem, "events": events}) + "\n")
    out.close()
    print(f"wrote {args.out}")


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
        if name == "train":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--epochs", type=int, default=80)
            p.add_argument("--patience", type=int, default=15)
            p.add_argument("--batch", type=int, default=48)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--workers", type=int, default=8)
        if name == "infer":
            p.add_argument("--model", type=Path, required=True,
                           help="best.pt (val_report.json with thresholds beside it)")
            p.add_argument("--audio", nargs="+", required=True)
            p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"acquire": cmd_acquire, "separate": cmd_separate, "features": cmd_features,
     "train": cmd_train, "infer": cmd_infer}[args.cmd](args)


if __name__ == "__main__":
    main()
