#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "mirdata>=0.3.8",
#   "librosa>=0.10",
#   "numpy>=1.26,<2.3",
#   "torch==2.8.*",
#   "tqdm>=4.66",
# ]
# ///
"""K10: train a permissively-licensed 24-class key CNN (pre-registered in
paper/EXPERIMENTS.md before any training).

Design (fixed): Korzeniowski & Widmer (ISMIR 2018) family — log-CQT input
(24 bins/octave), small conv stack + global average pooling + 24-way softmax.
Training corpus: mirdata `beatport_key` (1,486 tracks; verified superset of
GS-MTG with zero GiantSteps-Key overlap). Augmentation: ±4-semitone pitch shift
via CQT bin-roll with label transposition. ALL model selection is 5-fold CV
within the training corpus; GiantSteps Key is never touched here.

Stages:
  acquire   download annotations+audio via mirdata, build labels.json
  features  log-CQT per track -> features/<id>.npy (float16, bins x frames)
  train     5-fold CV (+ final full-train model at CV-selected budget)
  infer     emit per-track 24-class posteriors for a directory of audio

  uv run eval/train_key_cnn.py acquire  --data-home /data/bpkey
  uv run eval/train_key_cnn.py features --data-home /data/bpkey
  uv run eval/train_key_cnn.py train    --data-home /data/bpkey --out /data/k10
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

SR = 22050
HOP = 4096  # ~5.4 fps
BINS_PER_OCT = 24
N_OCT = 8
PAD_SEMI = 4  # augmentation margin on each side, in semitones
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}
SEED = 0


def parse_key(raw: str | None) -> int | None:
    """'D minor' -> class id tonic*2 + (mode==minor). None if unusable."""
    if not raw:
        return None
    parts = raw.strip().replace("Major", "major").replace("Minor", "minor").split()
    if len(parts) != 2:
        return None
    tonic = FLAT.get(parts[0], parts[0])
    if tonic not in NOTES or parts[1] not in ("major", "minor"):
        return None
    return NOTES.index(tonic) * 2 + (1 if parts[1] == "minor" else 0)


def transpose_label(cls: int, semitones: int) -> int:
    return ((cls // 2 + semitones) % 12) * 2 + (cls % 2)


# --------------------------------------------------------------------------- acquire
def cmd_acquire(args) -> None:
    import mirdata

    bp = mirdata.initialize("beatport_key", data_home=str(args.data_home))
    bp.download()  # index + annotations + audio (idempotent)
    labels, drops = {}, []
    for tid in bp.track_ids:
        t = bp.track(tid)
        cls = parse_key(t.key[0] if isinstance(t.key, list) else t.key)
        audio_ok = t.audio_path and Path(t.audio_path).exists()
        if cls is None or not audio_ok:
            drops.append({"tid": tid, "key": t.key, "audio": bool(audio_ok)})
            continue
        labels[tid] = {"cls": cls, "audio": str(t.audio_path)}
    out = args.data_home / "labels.json"
    json.dump(labels, open(out, "w"))
    print(f"usable {len(labels)} / {len(bp.track_ids)}; dropped {len(drops)} "
          f"(multi-key/unparseable/missing-audio) -> {out}")
    json.dump(drops, open(args.data_home / "drops.json", "w"), indent=1)


# -------------------------------------------------------------------------- features
def cmd_features(args) -> None:
    import librosa
    from tqdm import tqdm

    labels = json.load(open(args.data_home / "labels.json"))
    fdir = args.data_home / "features"
    fdir.mkdir(exist_ok=True)
    n_bins = (N_OCT * 12 + 2 * PAD_SEMI) * (BINS_PER_OCT // 12)
    fmin = librosa.note_to_hz("C1") * 2 ** (-PAD_SEMI / 12)
    fails = 0
    for tid, rec in tqdm(labels.items()):
        out = fdir / f"{tid}.npy"
        if out.exists():
            continue
        try:
            y, _ = librosa.load(rec["audio"], sr=SR, mono=True)
            C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=fmin,
                                   n_bins=n_bins, bins_per_octave=BINS_PER_OCT))
            np.save(out, np.log1p(C).astype(np.float16))
        except Exception as e:  # noqa: BLE001 — log and count, never silent
            fails += 1
            print(f"FAIL {tid}: {e}")
    if fails:
        raise SystemExit(f"{fails} feature failures — resolve before training")
    print(f"features complete -> {fdir}")


# ----------------------------------------------------------------------------- model
def build_model():
    import torch.nn as nn

    def block(cin, cout, pool):
        layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ELU(),
                  nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ELU()]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return layers

    # Head pools over TIME ONLY: key identity lives in absolute frequency position,
    # so frequency structure must reach the readout (a full 2-D global pool made the
    # net transposition-invariant and pinned training at chance — caught in CV).
    return nn.Sequential(
        *block(1, 16, True), *block(16, 32, True), *block(32, 64, True),
        nn.Dropout2d(0.2),
        nn.AdaptiveAvgPool2d((24, 1)),  # (freq stays 24 after 3 halvings, time -> 1)
        nn.Flatten(),
        nn.Linear(64 * 24, 24),
    )


class KeyDataset:
    """Random fixed-length crops with ±PAD_SEMI bin-roll augmentation."""

    def __init__(self, items, fdir, train: bool, crop_frames: int = 320):
        self.items, self.fdir, self.train, self.crop = items, fdir, train, crop_frames
        self.per_semi = BINS_PER_OCT // 12
        self.core_bins = N_OCT * BINS_PER_OCT

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch

        tid, cls = self.items[i]
        X = np.load(self.fdir / f"{tid}.npy").astype(np.float32)
        if self.train:
            shift = random.randint(-PAD_SEMI, PAD_SEMI)
        else:
            shift = 0
        lo = (PAD_SEMI + shift) * self.per_semi
        X = X[lo:lo + self.core_bins]
        cls = transpose_label(cls, shift)
        T = X.shape[1]
        if self.train and T > self.crop:
            s = random.randint(0, T - self.crop)
            X = X[:, s:s + self.crop]
        return torch.from_numpy(X)[None], cls


def mirex_weighted(pred_cls: int, ref_cls: int) -> float:
    pt, pm, rt, rm = pred_cls // 2, pred_cls % 2, ref_cls // 2, ref_cls % 2
    iv = (pt - rt) % 12
    if iv == 0 and pm == rm:
        return 1.0
    if pm == rm and iv in (7, 5):
        return 0.5
    if rm == 0 and pm == 1 and iv == 9:
        return 0.3
    if rm == 1 and pm == 0 and iv == 3:
        return 0.3
    if iv == 0:
        return 0.2
    return 0.0


# ----------------------------------------------------------------------------- train
def cmd_train(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if dev == "cuda":  # cuDNN sanity gate (fleet lesson)
        torch.nn.functional.conv2d(torch.zeros(1, 1, 8, 8, device="cuda"),
                                   torch.zeros(1, 1, 3, 3, device="cuda"))
    print(f"device={dev}")

    labels = json.load(open(args.data_home / "labels.json"))
    fdir = args.data_home / "features"
    tids = sorted(labels)
    rng = random.Random(SEED)
    rng.shuffle(tids)
    folds = [tids[i::5] for i in range(5)]
    args.out.mkdir(parents=True, exist_ok=True)

    def run_fold(k: int, max_epochs: int):
        val_ids = set(folds[k])
        tr = [(t, labels[t]["cls"]) for t in tids if t not in val_ids]
        va = [(t, labels[t]["cls"]) for t in tids if t in val_ids]
        dl_tr = DataLoader(KeyDataset(tr, fdir, True), batch_size=args.batch,
                           shuffle=True, num_workers=args.workers,
                           collate_fn=collate_pad, drop_last=True)
        model = build_model().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
        best = (0.0, 0)
        hist = []
        for ep in range(1, max_epochs + 1):
            model.train()
            tot = n = 0
            for X, y in dl_tr:
                X, y = X.to(dev), y.to(dev)
                loss = torch.nn.functional.cross_entropy(model(X), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(y)
                n += len(y)
            sched.step()
            w = evaluate(model, va, fdir, dev)
            hist.append({"epoch": ep, "train_loss": tot / n, "val_weighted": w})
            print(f"fold{k} ep{ep:02d} loss={tot / n:.4f} val_weighted={w:.4f}")
            if w > best[0]:
                best = (w, ep)
                torch.save(model.state_dict(), args.out / f"fold{k}_best.pt")
            elif ep - best[1] >= args.patience:
                break
        json.dump(hist, open(args.out / f"fold{k}_hist.json", "w"), indent=1)
        return best

    results = [run_fold(k, args.epochs) for k in range(5)]
    cv = {"folds": [{"best_weighted": b[0], "best_epoch": b[1]} for b in results],
          "cv_weighted_mean": float(np.mean([b[0] for b in results])),
          "median_best_epoch": int(np.median([b[1] for b in results]))}
    json.dump(cv, open(args.out / "cv_summary.json", "w"), indent=1)
    print("CV:", json.dumps(cv))

    # Final model: all training data, median best-epoch budget (no val peeking).
    tr = [(t, labels[t]["cls"]) for t in tids]
    dl = DataLoader(KeyDataset(tr, fdir, True), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=collate_pad, drop_last=True)
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_ep = cv["median_best_epoch"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_ep)
    for ep in range(1, n_ep + 1):
        model.train()
        for X, y in dl:
            X, y = X.to(dev), y.to(dev)
            loss = torch.nn.functional.cross_entropy(model(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        print(f"final ep{ep:02d}/{n_ep}")
    torch.save(model.state_dict(), args.out / "final.pt")
    print(f"done -> {args.out}/final.pt")


def collate_pad(batch):
    import torch

    T = max(x.shape[-1] for x, _ in batch)
    X = torch.zeros(len(batch), 1, batch[0][0].shape[1], T)
    y = torch.zeros(len(batch), dtype=torch.long)
    for i, (x, c) in enumerate(batch):
        X[i, :, :, : x.shape[-1]] = x
        y[i] = c
    return X, y


def evaluate(model, items, fdir, dev) -> float:
    import torch

    model.eval()
    scores = []
    with torch.no_grad():
        for tid, cls in items:
            X = np.load(fdir / f"{tid}.npy").astype(np.float32)
            per = BINS_PER_OCT // 12
            X = X[PAD_SEMI * per: PAD_SEMI * per + N_OCT * BINS_PER_OCT]
            logits = model(torch.from_numpy(X)[None, None].to(dev))
            scores.append(mirex_weighted(int(logits.argmax()), cls))
    return float(np.mean(scores))


# ----------------------------------------------------------------------------- infer
def cmd_infer(args) -> None:
    import librosa
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model()
    model.load_state_dict(torch.load(args.model, map_location=dev))
    model.to(dev).eval()
    n_bins = (N_OCT * 12 + 2 * PAD_SEMI) * (BINS_PER_OCT // 12)
    fmin = librosa.note_to_hz("C1") * 2 ** (-PAD_SEMI / 12)
    per = BINS_PER_OCT // 12
    out = open(args.out, "w")
    for p in sorted(Path(args.audio_dir).iterdir()):
        if p.suffix.lower() not in (".mp3", ".wav", ".ogg", ".flac", ".m4a"):
            continue
        y, _ = librosa.load(p, sr=SR, mono=True)
        C = np.log1p(np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=fmin,
                                        n_bins=n_bins, bins_per_octave=BINS_PER_OCT)))
        X = C[PAD_SEMI * per: PAD_SEMI * per + N_OCT * BINS_PER_OCT].astype(np.float32)
        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(X)[None, None].to(dev)), -1)[0]
        cls = int(probs.argmax())
        key = f"{NOTES[cls // 2]} {'minor' if cls % 2 else 'major'}"
        out.write(json.dumps({"track_id": p.stem, "cnn_key": key,
                              "probs": [round(float(v), 6) for v in probs]}) + "\n")
    out.close()
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("acquire", "features", "train", "infer"):
        p = sub.add_parser(name)
        p.add_argument("--data-home", type=Path, default=Path("eval/data/bpkey"))
        if name == "train":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--epochs", type=int, default=60)
            p.add_argument("--patience", type=int, default=10)
            p.add_argument("--batch", type=int, default=32)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--workers", type=int, default=8)
        if name == "infer":
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--audio-dir", type=Path, required=True)
            p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"acquire": cmd_acquire, "features": cmd_features,
     "train": cmd_train, "infer": cmd_infer}[args.cmd](args)


if __name__ == "__main__":
    main()
