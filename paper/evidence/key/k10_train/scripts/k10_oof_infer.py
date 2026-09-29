"""K10: out-of-fold CNN posteriors for the training corpus.

Each track is inferred by the fold model that did NOT train on it (exactly the
trainer's split: sorted tids, random.Random(0).shuffle, folds = tids[i::5]).
Output keyed by Beatport catalog id (audio filename prefix) to join with
keyfeat_gsmtg.jsonl / skey_gsmtg.jsonl. Run on the box with cached features:

  uv run --with torch==2.8.*,numpy k10_oof_infer.py
"""
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HOME = Path.home()
spec = importlib.util.spec_from_file_location("tkc", HOME / "train_key_cnn.py")
tkc = importlib.util.module_from_spec(spec)
sys.modules["tkc"] = tkc
spec.loader.exec_module(tkc)

data_home = HOME / "bpkey"
out_dir = HOME / "k10out"
labels = json.load(open(data_home / "labels.json"))
fdir = data_home / "features"

tids = sorted(labels)
rng = random.Random(tkc.SEED)
rng.shuffle(tids)
folds = [tids[i::5] for i in range(5)]

dev = "cuda" if torch.cuda.is_available() else "cpu"
per = tkc.BINS_PER_OCT // 12
core = tkc.N_OCT * tkc.BINS_PER_OCT

out = open(out_dir / "cnn_oof_train.jsonl", "w")
n = 0
for k in range(5):
    model = tkc.build_model()
    model.load_state_dict(torch.load(out_dir / f"fold{k}_best.pt", map_location=dev))
    model.to(dev).eval()
    with torch.no_grad():
        for tid in folds[k]:
            X = np.load(fdir / f"{tid}.npy").astype(np.float32)
            X = X[tkc.PAD_SEMI * per: tkc.PAD_SEMI * per + core]
            probs = torch.softmax(model(torch.from_numpy(X)[None, None].to(dev)), -1)[0]
            cls = int(probs.argmax())
            catalog = Path(labels[tid]["audio"]).name.split()[0].split(".")[0]
            out.write(json.dumps({
                "track_id": catalog, "bp_tid": tid, "fold": k,
                "ref_cls": labels[tid]["cls"], "pred_cls": cls,
                "cnn_key": f"{tkc.NOTES[cls // 2]} {'minor' if cls % 2 else 'major'}",
                "probs": [round(float(v), 6) for v in probs],
            }) + "\n")
            n += 1
out.close()
print(f"wrote {n} OOF rows -> {out_dir}/cnn_oof_train.jsonl")

# quick self-check: OOF weighted score should match cv_weighted_mean
rows = [json.loads(l) for l in open(out_dir / "cnn_oof_train.jsonl")]
w = float(np.mean([tkc.mirex_weighted(r["pred_cls"], r["ref_cls"]) for r in rows]))
print(f"OOF weighted (should ~= cv mean): {w:.4f}")
