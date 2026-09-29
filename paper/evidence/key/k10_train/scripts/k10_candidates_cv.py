"""K10 candidate selection — CV ONLY (GS-Key is never read here).

Basis: GS-MTG ∩ CNN-OOF (n≈1129). Candidates per pre-registration:
  ref  production-style fusion with S-KEY (baseline for comparison)
  (a)  CNN standalone (OOF predictions)
  (b)  fusion with CNN posteriors in the S-KEY slot
  (c)  fusion reranking between S-KEY and CNN
CNN OOF probs are out-of-fold wrt CNN training -> leak-free in fusion CV.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

T = Path('/Users/jhurliman/.claude/jobs/7eb03476/tmp')
spec = importlib.util.spec_from_file_location("kf", T / "key_fusion.py")
kf = importlib.util.module_from_spec(spec); sys.modules["kf"] = kf
spec.loader.exec_module(kf)  # reuse parse/mirex/skey_feats/load_split/build/etc.
NOTES, SKEY_MAP = kf.NOTES, kf.SKEY_MAP

train = kf.load_split(kf.WT / "eval/data/gsmtg/manifest.jsonl",
                      kf.ROOT / "eval/data/gsmtg/skey_gsmtg.jsonl",
                      kf.ROOT / "eval/data/gsmtg/keyfeat_gsmtg.jsonl")
oof = {json.loads(l)["track_id"]: json.loads(l) for l in open(T / "cnn_oof_train.jsonl")}
rows = [r for r in train if r["tid"] in oof]
for r in rows:
    r["cnn"] = oof[r["tid"]]
print(f"basis n={len(rows)} (gsmtg∩oof)")

def cnn_probs_skey_order(probs):
    return [probs[NOTES.index(n) * 2 + (1 if m == "minor" else 0)] for n, m in SKEY_MAP]

def fusion_cv(rows, post_of, key_of, label):
    """Full-pipeline 5-fold CV: mode head + rerank(refined-edma vs candidate)."""
    X, y, meta = [], [], []
    for r in rows:
        tonic = NOTES.index(r["feat"]["edma_tonic"])
        base = list(r["feat"]["cues"]) + [r["feat"]["edma_conf"]] + \
               kf.skey_feats(post_of(r), tonic, r["feat"]["edma_mode"])
        X.append(base); y.append(1 if kf.parse(r["ref"])[1] == "minor" else 0); meta.append(r)
    X, y = np.array(X), np.array(y)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    best = (-1, None, None)
    for mthr in (0.6, 0.7, 0.8, 0.9):
        for rthr in (0.5, 0.6, 0.7, 0.8):
            ws = []
            for tr, te in skf.split(X, y):
                sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
                # refined keys on both parts
                def refined(idx):
                    p = clf.predict_proba(sc.transform(X[idx]))[:, 1]
                    out = []
                    for pi, i in zip(p, idx):
                        tonic, mode = meta[i]["feat"]["edma_tonic"], meta[i]["feat"]["edma_mode"]
                        if pi >= mthr: mode = "minor"
                        elif pi <= 1 - mthr: mode = "major"
                        out.append(f"{tonic} {mode}")
                    return out
                rk_tr, rk_te = refined(tr), refined(te)
                def rrfeat(idx, rk):
                    F, Y = [], []
                    for i, ek in zip(idx, rk):
                        r = meta[i]
                        f = list(X[i]) + [1.0 if kf.parse(ek) == kf.parse(key_of(r)) else 0.0]
                        F.append(f)
                        Y.append(1 if kf.mirex(r["ref"], key_of(r)) > kf.mirex(r["ref"], ek) else 0)
                    return np.array(F), np.array(Y)
                Ftr, Ytr = rrfeat(tr, rk_tr); Fte, _ = rrfeat(te, rk_te)
                s2 = StandardScaler().fit(Ftr); rr = LogisticRegression(max_iter=1000).fit(s2.transform(Ftr), Ytr)
                pr = rr.predict_proba(s2.transform(Fte))[:, 1]
                preds = [key_of(meta[i]) if p >= rthr else ek for i, p, ek in zip(te, pr, rk_te)]
                ws.append(float(np.mean([kf.mirex(meta[i]["ref"], k) for i, k in zip(te, preds)])))
            m = float(np.mean(ws))
            if m > best[0]: best = (m, mthr, rthr)
    print(f"{label}: cv_weighted={best[0]:.4f} (mode_thr={best[1]}, rerank_thr={best[2]})")
    return best[0]

# (a) CNN standalone (OOF)
w_a = float(np.mean([kf.mirex(r["ref"], r["cnn"]["cnn_key"]) for r in rows]))
print(f"(a) CNN standalone OOF: weighted={w_a:.4f}")
# ref: S-KEY-based fusion on same basis
w_ref = fusion_cv(rows, lambda r: r["skey"]["posterior"], lambda r: r["skey"]["skey_key"], "ref S-KEY fusion")
# (b) CNN in the S-KEY slot
w_b = fusion_cv(rows, lambda r: cnn_probs_skey_order(r["cnn"]["probs"]), lambda r: r["cnn"]["cnn_key"], "(b) CNN-slot fusion")
# (c) rerank S-KEY vs CNN (edma cues as context; candidates are the two model keys)
def fusion_c(rows):
    X, meta = [], []
    for r in rows:
        tonic = NOTES.index(r["feat"]["edma_tonic"])
        f = list(r["feat"]["cues"]) + [r["feat"]["edma_conf"]]
        f += kf.skey_feats(r["skey"]["posterior"], tonic, r["feat"]["edma_mode"])
        f += kf.skey_feats(cnn_probs_skey_order(r["cnn"]["probs"]), tonic, r["feat"]["edma_mode"])
        f.append(1.0 if kf.parse(r["skey"]["skey_key"]) == kf.parse(r["cnn"]["cnn_key"]) else 0.0)
        X.append(f); meta.append(r)
    X = np.array(X)
    y = np.array([1 if kf.mirex(r["ref"], r["cnn"]["cnn_key"]) > kf.mirex(r["ref"], r["skey"]["skey_key"]) else 0 for r in meta])
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    best = -1
    for thr in (0.5, 0.6, 0.7, 0.8):
        ws = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr]); m = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            pr = m.predict_proba(sc.transform(X[te]))[:, 1]
            preds = [meta[i]["cnn"]["cnn_key"] if p >= thr else meta[i]["skey"]["skey_key"] for i, p in zip(te, pr)]
            ws.append(float(np.mean([kf.mirex(meta[i]["ref"], k) for i, k in zip(te, preds)])))
        if np.mean(ws) > best: best = float(np.mean(ws))
    print(f"(c) S-KEY-vs-CNN rerank: cv_weighted={best:.4f}")
    return best
w_c = fusion_c(rows)
print(json.dumps({"a": w_a, "ref": w_ref, "b": w_b, "c": w_c}))
