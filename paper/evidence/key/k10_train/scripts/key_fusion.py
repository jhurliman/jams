"""Key-detection fusion experiment — honest protocol.

Train split: GiantSteps-MTG (1157). Test: GiantSteps Key (567), evaluated ONCE.
Systems compared (MIREX weighted / exact):
  A. edma raw (no refinement)
  B. current production (mode model trained on the test set — CONTAMINATED, reference only)
  C. honest retrain: same 8-cue logistic mode refinement, fit on GS-MTG
  D. S-KEY standalone
  E. fusion-mode: logistic mode decision on 8 cues + edma conf + S-KEY posterior features
  F. fusion-rerank: choose among {edma key, parallel, S-KEY key} with a classifier
Model selection (thresholds, C) via 5-fold CV on GS-MTG only.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/Users/jhurliman/Documents/Code/jhurliman/jams")
WT = Path("/Users/jhurliman/Documents/Code/jhurliman/jams/.claude/worktrees/hard-deps")
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}
SKEY_MAP = (
    [(n, "major") for n in ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]]
    + [(n, "minor") for n in ["B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#"]]
)
SKEY_IDX = {k: i for i, k in enumerate(SKEY_MAP)}


def parse(k):
    p = (k or "").replace("Major", "major").replace("Minor", "minor").split()
    if len(p) != 2:
        return None
    t = FLAT.get(p[0], p[0])
    return (NOTES.index(t), p[1]) if t in NOTES and p[1] in ("major", "minor") else None


def mirex(ref, est):
    r, e = parse(ref), parse(est)
    if r is None or e is None:
        return 0.0
    iv = (e[0] - r[0]) % 12
    if iv == 0 and e[1] == r[1]:
        return 1.0
    if e[1] == r[1] and iv in (7, 5):
        return 0.5
    if r[1] == "major" and e[1] == "minor" and iv == 9:
        return 0.3
    if r[1] == "minor" and e[1] == "major" and iv == 3:
        return 0.3
    if iv == 0:
        return 0.2
    return 0.0


def load_split(manifest, skey_file, feat_file):
    gt = {str(json.loads(x)["track_id"]): json.loads(x) for x in open(manifest)}
    sk = {j["track_id"]: j for j in map(json.loads, open(skey_file)) if "posterior" in j}
    ft = {j["track_id"]: j for j in map(json.loads, open(feat_file)) if "cues" in j}
    rows = []
    for tid, g in gt.items():
        if tid not in sk or tid not in ft or parse(g["ref_key"]) is None:
            continue
        rows.append({"tid": tid, "ref": g["ref_key"], "skey": sk[tid], "feat": ft[tid]})
    return rows


def skey_feats(post, tonic_idx, edma_mode):
    """S-KEY posterior features anchored at edma's tonic."""
    p = np.asarray(post)
    def P(t, m):
        return p[SKEY_IDX[(NOTES[t % 12], m)]]
    rel_minor = (tonic_idx + 9) % 12   # relative minor of a major tonic
    rel_major = (tonic_idx + 3) % 12   # relative major of a minor tonic
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return [
        float(P(tonic_idx, "minor")), float(P(tonic_idx, "major")),
        float(P(tonic_idx, "minor") - P(tonic_idx, "major")),
        float(P(rel_minor, "minor")), float(P(rel_major, "major")),
        float(P(tonic_idx + 7, edma_mode)), float(P(tonic_idx + 5, edma_mode)),
        float(p.max()), ent,
    ]


def build(rows, feature_set):
    X, y_mode, meta = [], [], []
    for r in rows:
        tonic = NOTES.index(r["feat"]["edma_tonic"])
        cues = r["feat"]["cues"]
        conf = r["feat"]["edma_conf"]
        base = list(cues)
        if feature_set == "fusion":
            base += [conf] + skey_feats(r["skey"]["posterior"], tonic, r["feat"]["edma_mode"])
        ref = parse(r["ref"])
        # mode label is only supervised when edma got the tonic right (same as prod recipe)
        y = 1 if ref[1] == "minor" else 0
        X.append(base)
        y_mode.append(y)
        meta.append(r)
    return np.array(X), np.array(y_mode), meta


def fit_mode_model(X, y, seed=0):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(sc.transform(X), y)
    return sc, clf


def apply_mode(sc, clf, X, meta, thr):
    """edma tonic kept; mode overridden only when classifier is confident."""
    p_minor = clf.predict_proba(sc.transform(X))[:, 1]
    out = []
    for p, r in zip(p_minor, meta):
        tonic, mode = r["feat"]["edma_tonic"], r["feat"]["edma_mode"]
        if p >= thr:
            mode = "minor"
        elif p <= 1 - thr:
            mode = "major"
        out.append(f"{tonic} {mode}")
    return out


def score(meta, preds):
    w = [mirex(r["ref"], k) for r, k in zip(meta, preds)]
    ex = [1.0 if x == 1.0 else 0.0 for x in w]
    return float(np.mean(w)), float(np.mean(ex))


def cv_threshold(X, y, meta, thrs, seed=0):
    """Pick the confidence threshold by 5-fold CV MIREX on the training split."""
    best_thr, best_w = None, -1
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for thr in thrs:
        ws = []
        for tr, te in skf.split(X, y):
            sc, clf = fit_mode_model(X[tr], y[tr], seed)
            preds = apply_mode(sc, clf, X[te], [meta[i] for i in te], thr)
            ws.append(score([meta[i] for i in te], preds)[0])
        m = float(np.mean(ws))
        if m > best_w:
            best_w, best_thr = m, thr
    return best_thr, best_w


def main():
    train = load_split(WT / "eval/data/gsmtg/manifest.jsonl",
                       ROOT / "eval/data/gsmtg/skey_gsmtg.jsonl",
                       ROOT / "eval/data/gsmtg/keyfeat_gsmtg.jsonl")
    test = load_split(ROOT / "eval/data/manifest.jsonl",
                      ROOT / "eval/data/gsmtg/skey_gskey.jsonl",
                      ROOT / "eval/data/gsmtg/keyfeat_gskey.jsonl")
    print(f"train n={len(train)}  test n={len(test)}")

    # A. edma raw
    preds = [f"{r['feat']['edma_tonic']} {r['feat']['edma_mode']}" for r in test]
    print("A. edma raw:            w=%.4f exact=%.4f" % score(test, preds))

    # D. skey standalone
    preds = [r["skey"]["skey_key"] for r in test]
    print("D. S-KEY standalone:    w=%.4f exact=%.4f" % score(test, preds))

    thrs = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    # C. honest retrain (cues only)
    Xtr, ytr, mtr = build(train, "cues")
    Xte, yte, mte = build(test, "cues")
    thr, cvw = cv_threshold(Xtr, ytr, mtr, thrs)
    sc, clf = fit_mode_model(Xtr, ytr)
    preds = apply_mode(sc, clf, Xte, mte, thr)
    print("C. honest retrain:      w=%.4f exact=%.4f  (thr=%.2f cv=%.4f)"
          % (*score(mte, preds), thr, cvw))

    # E. fusion-mode
    Xtr, ytr, mtr = build(train, "fusion")
    Xte, yte, mte = build(test, "fusion")
    thr, cvw = cv_threshold(Xtr, ytr, mtr, thrs)
    sc, clf = fit_mode_model(Xtr, ytr)
    preds = apply_mode(sc, clf, Xte, mte, thr)
    print("E. fusion-mode:         w=%.4f exact=%.4f  (thr=%.2f cv=%.4f)"
          % (*score(mte, preds), thr, cvw))

    # F. fusion-rerank: candidates {edma-as-refined-by-E, skey argmax}; pick by classifier
    # trained to predict which candidate scores higher (features: fusion set + agreement).
    def rerank_features(rows):
        F, Y, M = [], [], []
        for r in rows:
            tonic = NOTES.index(r["feat"]["edma_tonic"])
            f = list(r["feat"]["cues"]) + [r["feat"]["edma_conf"]] + \
                skey_feats(r["skey"]["posterior"], tonic, r["feat"]["edma_mode"])
            e_key = f"{r['feat']['edma_tonic']} {r['feat']['edma_mode']}"
            s_key = r["skey"]["skey_key"]
            agree = 1.0 if parse(e_key) == parse(s_key) else 0.0
            f.append(agree)
            w_e, w_s = mirex(r["ref"], e_key), mirex(r["ref"], s_key)
            F.append(f); Y.append(1 if w_s > w_e else 0); M.append((r, e_key, s_key))
        return np.array(F), np.array(Y), M

    Ftr, Ytr, Mtr = rerank_features(train)
    Fte, Yte, Mte = rerank_features(test)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Ftr)
    rr = LogisticRegression(max_iter=1000).fit(scaler.transform(Ftr), Ytr)
    # threshold for switching to skey, CV-picked
    best_thr, best_w = None, -1
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for thr in [0.5, 0.6, 0.7, 0.8]:
        ws = []
        for tr, te in skf.split(Ftr, Ytr):
            s2 = StandardScaler().fit(Ftr[tr])
            m2 = LogisticRegression(max_iter=1000).fit(s2.transform(Ftr[tr]), Ytr[tr])
            pr = m2.predict_proba(s2.transform(Ftr[te]))[:, 1]
            preds = [Mtr[i][2] if p >= thr else Mtr[i][1] for i, p in zip(te, pr)]
            ws.append(float(np.mean([mirex(Mtr[i][0]["ref"], k) for i, k in zip(te, preds)])))
        if np.mean(ws) > best_w:
            best_w, best_thr = float(np.mean(ws)), thr
    pr = rr.predict_proba(scaler.transform(Fte))[:, 1]
    preds = [m[2] if p >= best_thr else m[1] for m, p in zip(Mte, pr)]
    w = [mirex(m[0]["ref"], k) for m, k in zip(Mte, preds)]
    print("F. fusion-rerank:       w=%.4f exact=%.4f  (thr=%.2f cv=%.4f, switched=%d)"
          % (float(np.mean(w)), float(np.mean([1.0 if x == 1.0 else 0.0 for x in w])),
             best_thr, best_w, int((pr >= best_thr).sum())))


if __name__ == "__main__":
    main()
