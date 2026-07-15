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
"""TP1: train an MIT-licensed tempo CNN (pre-registered in paper/EXPERIMENTS.md
before any training) to replace the production TempoCNN weights + Essentia
inference.

Design (fixed by the pre-registration): clean-room Schreiber & Müller (ISMIR
2018) family — mel input (sr 11,025, 40 mels, hop 512), 256-class softmax
(30–285 BPM, 1-BPM bins), sliding-window softmax averaging at inference.
Training corpus: Raveform bpm_ref (expert-beat-derived) + GiantSteps-Tempo v2
MINUS the 42 tracks whose Beatport catalog ids appear in GiantSteps Key (eval
set). 5-fold CV only; the n=458 GS-Key tempo gate is evaluated exactly once,
elsewhere.

Architecture v2 (CV-stage amendment, see EXPERIMENTS.md TP1 addendum): v1's
TimeAvg readout collapsed all temporal structure into 36 channel means before
the classifier and stalled at CV Acc1 0.445 (loss barely under the ln 256
floor). v2 follows the published architecture (paper only — no AGPL code
consulted): 3 short-filter convs (16 x 1x5 along time), 4 multi-filter modules
(freq avg-pool 5/2/2/2 x1, six parallel 1x{32..256} convs of 24 filters, 1x1
bottleneck to 36), then flatten the intact time axis into FC 64 -> FC 64 ->
FC 256. Windows are rescaled to [0,1] on the magnitude mel (reconstructed from
the stored log1p-power features). Augmentation: discrete time-axis scale
factors {0.8, 0.84, ..., 1.2} with the label adjusted (bpm/f), skipped at
validation. Adam at constant lr, early stop on val Acc1 (patience 20).

Stages:
  acquire   gs_tempo via mirdata (with overlap exclusion) + raveform manifest
  features  log-mel per track -> features/<id>.npy (float16, 40 x frames)
  train     5-fold CV (+ final full-train model at CV-selected budget)
  infer     per-track BPM posterior for a directory of audio or a manifest
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

SR = 11025
N_FFT = 1024
HOP = 512
N_MELS = 40
WIN_FRAMES = 256          # ~11.9 s
BPM_MIN, BPM_MAX = 30, 285  # 256 classes, 1-BPM bins
SEED = 0


def bpm_to_cls(bpm: float) -> int | None:
    c = int(round(bpm)) - BPM_MIN
    return c if 0 <= c < 256 else None


# --------------------------------------------------------------------------- acquire
def cmd_acquire(args) -> None:
    import mirdata

    labels, drops = {}, []

    # GS-Key catalog ids (exclusion list) from the eval manifest
    gskey_ids = set()
    for line in open(args.gskey_manifest):
        r = json.loads(line)
        gskey_ids.add(Path(r["audio_path"]).name.split()[0].split(".")[0])

    gt = mirdata.initialize("giantsteps_tempo", data_home=str(args.data_home / "gstempo"))
    gt.download()  # index + annotations; audio is manual in mirdata -> fetch below
    excluded = 0
    import subprocess

    def fetch_audio(catalog: str, dest: Path) -> bool:
        """Beatport preview, then JKU mirrors (same sources as acquire_gsmtg.py)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        for url in (
            f"http://geo-samples.beatport.com/lofi/{catalog}.LOFI.mp3",
            f"http://www.cp.jku.at/datasets/giantsteps/backup/{catalog}.LOFI.mp3",
            f"http://www.cp.jku.at/datasets/giantsteps/tempo_backup/{catalog}.LOFI.mp3",
        ):
            r = subprocess.run(["curl", "-sfL", "--max-time", "60", "-o", str(dest), url])
            if r.returncode == 0 and dest.exists() and dest.stat().st_size > 10_000:
                return True
            dest.unlink(missing_ok=True)
        return False

    for tid in gt.track_ids:
        t = gt.track(tid)
        catalog = Path(t.audio_path).name.split(".")[0]
        if catalog in gskey_ids:
            excluded += 1
            continue
        if not Path(t.audio_path).exists():
            fetch_audio(catalog, Path(t.audio_path))
        bpm = None
        for attr in ("tempo_v2", "tempo"):
            try:
                v = getattr(t, attr, None)
            except ValueError:  # mirdata raises on empty annotation arrays
                continue
            if v is None:
                continue
            tempos = np.atleast_1d(getattr(v, "tempos", v)).astype(float)
            conf = getattr(v, "confidence", None)
            if tempos.size:
                # v2 lists multiple tempi with salience: take the most salient
                bpm = float(tempos[int(np.argmax(conf))] if conf is not None
                            else tempos[0])
                break
        cls = bpm_to_cls(bpm) if bpm else None
        if cls is None or not Path(t.audio_path).exists():
            drops.append({"src": "gstempo", "tid": tid, "bpm": bpm})
            continue
        labels[f"gst.{catalog}"] = {"bpm": bpm, "audio": str(t.audio_path)}
    print(f"gs_tempo: {sum(k.startswith('gst.') for k in labels)} kept, "
          f"{excluded} excluded (GS-Key overlap), {len(drops)} dropped")

    n_rav = 0
    for line in open(args.raveform_manifest):
        r = json.loads(line)
        bpm = r.get("bpm_ref")
        cls = bpm_to_cls(bpm) if bpm else None
        if cls is None or not r.get("audio_exists"):
            drops.append({"src": "raveform", "tid": r.get("track_id"), "bpm": bpm})
            continue
        labels[f"rav.{r['track_id']}"] = {"bpm": float(bpm), "audio": r["audio_path"]}
        n_rav += 1
    print(f"raveform: {n_rav} kept")

    args.data_home.mkdir(parents=True, exist_ok=True)
    json.dump(labels, open(args.data_home / "labels.json", "w"))
    json.dump(drops, open(args.data_home / "drops.json", "w"), indent=1)
    print(f"total usable: {len(labels)} -> {args.data_home}/labels.json")


# -------------------------------------------------------------------------- features
def cmd_features(args) -> None:
    import librosa
    from tqdm import tqdm

    labels = json.load(open(args.data_home / "labels.json"))
    fdir = args.data_home / "features"
    fdir.mkdir(exist_ok=True)
    fails = 0
    for tid, rec in tqdm(labels.items()):
        out = fdir / f"{tid}.npy"
        if out.exists():
            continue
        try:
            y, _ = librosa.load(rec["audio"], sr=SR, mono=True)
            M = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                               n_mels=N_MELS)
            np.save(out, np.log1p(M).astype(np.float16))
        except Exception as e:  # noqa: BLE001 — log and count, never silent
            fails += 1
            print(f"FAIL {tid}: {e}")
    if fails:
        raise SystemExit(f"{fails} feature failures — resolve before training")
    print(f"features complete -> {fdir}")


# ----------------------------------------------------------------------------- model
def build_model():
    import torch
    import torch.nn as nn

    class MFMod(nn.Module):
        """Multi-filter module: freq avg-pool, BN, six parallel temporal convs
        (24 filters each, ELU), concat, 1x1 bottleneck to 36 (ELU)."""

        KERNELS = (32, 64, 96, 128, 192, 256)

        def __init__(self, cin, pool):
            super().__init__()
            self.pool = nn.AvgPool2d((pool, 1))
            self.bn = nn.BatchNorm2d(cin)
            self.branches = nn.ModuleList(
                nn.Conv2d(cin, 24, (1, k), padding=(0, k // 2)) for k in self.KERNELS)
            self.bottleneck = nn.Conv2d(24 * len(self.KERNELS), 36, 1)
            self.act = nn.ELU()

        def forward(self, x):
            x = self.bn(self.pool(x))
            t = x.shape[-1]  # even kernels pad to T+1: trim back to input length
            outs = [self.act(b(x))[..., :t] for b in self.branches]
            return self.act(self.bottleneck(torch.cat(outs, dim=1)))

    def short_filter(cin):
        return [nn.BatchNorm2d(cin), nn.Conv2d(cin, 16, (1, 5), padding=(0, 2)),
                nn.ELU()]

    # Freq axis: 40 -> 8 -> 4 -> 2 -> 1; time axis stays WIN_FRAMES and is
    # flattened intact into the dense back-end (36 * 256 = 9216).
    return nn.Sequential(
        *short_filter(1), *short_filter(16), *short_filter(16),
        MFMod(16, 5), MFMod(36, 2), MFMod(36, 2), MFMod(36, 2),
        nn.Flatten(),
        nn.BatchNorm1d(36 * WIN_FRAMES),
        nn.Dropout(0.5),
        nn.Linear(36 * WIN_FRAMES, 64), nn.ELU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 64), nn.ELU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 256),
    )


AUG_FACTORS = [round(0.8 + 0.04 * i, 2) for i in range(11)]  # 0.8 .. 1.2


def window_norm(w: np.ndarray) -> np.ndarray:
    """Reconstruct magnitude mel from stored log1p(power) and rescale the
    window to [0,1] (eps-guarded for silent windows)."""
    w = np.sqrt(np.expm1(w))
    lo, hi = float(w.min()), float(w.max())
    return (w - lo) / (hi - lo + 1e-8)


def time_scale(X: np.ndarray, f: float) -> np.ndarray:
    """Rescale the time axis by factor f (linear interpolation)."""
    T = X.shape[1]
    newT = max(WIN_FRAMES, int(round(T * f)))
    pos = np.linspace(0, T - 1, newT)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = (pos - lo).astype(np.float32)
    return X[:, lo] * (1 - frac) + X[:, hi] * frac


class TempoDataset:
    """Random windows with time-scale augmentation (label rescaled)."""

    def __init__(self, items, fdir, train: bool):
        self.items, self.fdir, self.train = items, fdir, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch

        tid, bpm = self.items[i]
        X = np.load(self.fdir / f"{tid}.npy").astype(np.float32)
        if self.train:
            for _ in range(8):  # rejection-sample a factor that keeps bpm in range
                f = random.choice(AUG_FACTORS)
                if bpm_to_cls(bpm / f) is not None:
                    break
            else:
                f = 1.0
            if f != 1.0:
                X = time_scale(X, f)  # time axis stretched by f -> tempo bpm/f
                bpm = bpm / f
            T = X.shape[1]
            s = random.randint(0, T - WIN_FRAMES) if T > WIN_FRAMES else 0
            X = X[:, s:s + WIN_FRAMES]
        else:
            T = X.shape[1]
            s = max(0, (T - WIN_FRAMES) // 2)
            X = X[:, s:s + WIN_FRAMES]
        if X.shape[1] < WIN_FRAMES:
            X = np.pad(X, ((0, 0), (0, WIN_FRAMES - X.shape[1])))
        return torch.from_numpy(window_norm(X))[None], bpm_to_cls(bpm)


def acc1(pred_bpm: float, ref_bpm: float) -> float:
    return 1.0 if abs(pred_bpm - ref_bpm) <= 0.04 * ref_bpm else 0.0


def predict_track(model, X, dev):
    """Sliding windows, averaged softmax -> BPM."""
    import torch

    T = X.shape[1]
    wins = []
    for s in range(0, max(1, T - WIN_FRAMES + 1), WIN_FRAMES // 2):
        w = X[:, s:s + WIN_FRAMES]
        if w.shape[1] < WIN_FRAMES:
            w = np.pad(w, ((0, 0), (0, WIN_FRAMES - w.shape[1])))
        wins.append(window_norm(w))
    batch = torch.from_numpy(np.stack(wins)[:, None]).to(dev)
    with torch.no_grad():
        probs = torch.softmax(model(batch), -1).mean(0)
    return float(int(probs.argmax()) + BPM_MIN), probs.cpu().numpy()


# ----------------------------------------------------------------------------- train
def cmd_train(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "cuda":
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

    def evaluate(model, items):
        model.eval()
        scores = []
        for tid, bpm in items:
            X = np.load(fdir / f"{tid}.npy").astype(np.float32)
            pred, _ = predict_track(model, X, dev)
            scores.append(acc1(pred, bpm))
        return float(np.mean(scores))

    def run_fold(k: int):
        val_ids = set(folds[k])
        tr = [(t, labels[t]["bpm"]) for t in tids if t not in val_ids]
        va = [(t, labels[t]["bpm"]) for t in tids if t in val_ids]
        dl = DataLoader(TempoDataset(tr, fdir, True), batch_size=args.batch,
                        shuffle=True, num_workers=args.workers, drop_last=True)
        model = build_model().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best = (0.0, 0)
        hist = []
        for ep in range(1, args.epochs + 1):
            model.train()
            tot = n = 0
            for X, y in dl:
                X, y = X.to(dev), y.to(dev)
                loss = torch.nn.functional.cross_entropy(model(X), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(y)
                n += len(y)
            a = evaluate(model, va)
            hist.append({"epoch": ep, "train_loss": tot / n, "val_acc1": a})
            print(f"fold{k} ep{ep:02d} loss={tot / n:.4f} val_acc1={a:.4f}")
            if a > best[0]:
                best = (a, ep)
                torch.save(model.state_dict(), args.out / f"fold{k}_best.pt")
            elif ep - best[1] >= args.patience:
                break
        json.dump(hist, open(args.out / f"fold{k}_hist.json", "w"), indent=1)
        return best

    results = [run_fold(k) for k in range(5)]
    cv = {"folds": [{"best_acc1": b[0], "best_epoch": b[1]} for b in results],
          "cv_acc1_mean": float(np.mean([b[0] for b in results])),
          "median_best_epoch": int(np.median([b[1] for b in results]))}
    json.dump(cv, open(args.out / "cv_summary.json", "w"), indent=1)
    print("CV:", json.dumps(cv))

    tr = [(t, labels[t]["bpm"]) for t in tids]
    dl = DataLoader(TempoDataset(tr, fdir, True), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True)
    model = build_model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_ep = cv["median_best_epoch"]
    for ep in range(1, n_ep + 1):
        model.train()
        for X, y in dl:
            X, y = X.to(dev), y.to(dev)
            loss = torch.nn.functional.cross_entropy(model(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"final ep{ep:02d}/{n_ep}")
    torch.save(model.state_dict(), args.out / "final.pt")
    print(f"done -> {args.out}/final.pt")


# ----------------------------------------------------------------------------- infer
def cmd_infer(args) -> None:
    import librosa
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model()
    model.load_state_dict(torch.load(args.model, map_location=dev))
    model.to(dev).eval()
    out = open(args.out, "w")
    for line in open(args.manifest):
        r = json.loads(line)
        y, _ = librosa.load(r["audio_path"], sr=SR, mono=True)
        M = np.log1p(librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)).astype(np.float32)
        bpm, probs = predict_track(model, M, dev)
        out.write(json.dumps({"track_id": str(r["track_id"]), "cnn_bpm": bpm,
                              "top5": sorted(range(256), key=lambda c: -probs[c])[:5]}) + "\n")
    out.close()
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("acquire", "features", "train", "infer"):
        p = sub.add_parser(name)
        p.add_argument("--data-home", type=Path, default=Path("eval/data/tempotrain"))
        if name == "acquire":
            p.add_argument("--gskey-manifest", type=Path, required=True)
            p.add_argument("--raveform-manifest", type=Path, required=True)
        if name == "train":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--epochs", type=int, default=150)
            p.add_argument("--patience", type=int, default=20)
            p.add_argument("--batch", type=int, default=64)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--workers", type=int, default=8)
        if name == "infer":
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--manifest", type=Path, required=True)
            p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"acquire": cmd_acquire, "features": cmd_features,
     "train": cmd_train, "infer": cmd_infer}[args.cmd](args)


if __name__ == "__main__":
    main()
