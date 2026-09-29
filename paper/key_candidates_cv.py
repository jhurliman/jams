# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn>=1.4"]
# ///
"""Recompute the K10 candidate-selection cross-validation from committed inputs.

The pre-registration (paper/EXPERIMENTS.md, entry K10) named three final-system candidates
to be compared by cross-validation within the training data only: (a) the CNN standalone,
(b) the existing two-head fusion with the CNN's posteriors in its S-KEY slot, and (c) the
fusion reranking between S-KEY and the CNN; a reference row runs the existing S-KEY fusion
on the same basis. The basis is the intersection of the GiantSteps-MTG feature set with the
CNN's out-of-fold posteriors (n = 1,129). This script re-implements the archived run script
(paper/evidence/key/k10_train/scripts/k10_candidates_cv.py, kept verbatim) on committed
inputs and checks the four numbers against the ledger.

Inputs (all under paper/evidence/key/):
  k10_oof_train.jsonl                          CNN out-of-fold posteriors (bp_tid, track_id)
  k10_train/selection/gsmtg_labels.jsonl       GS-MTG track ids and reference keys
  k10_train/selection/skey_gsmtg.jsonl         S-KEY posteriors on GS-MTG
  k10_train/selection/keyfeat_gsmtg.jsonl      template-matcher cues on GS-MTG
  k10_train/scripts/key_fusion.py              the fusion feature/scoring module (verbatim)

Usage: uv run paper/key_candidates_cv.py   -> paper/evidence/key/k10_candidates_cv.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

KEY = Path(__file__).resolve().parent / "evidence" / "key"
SEL = KEY / "k10_train" / "selection"
OUT = KEY / "k10_candidates_cv.json"
EXPECTED = {"a": 0.7489, "ref": 0.7308, "b": 0.7466, "c": 0.7488}


def load_fusion_module():
    spec = importlib.util.spec_from_file_location(
        "key_fusion", KEY / "k10_train" / "scripts" / "key_fusion.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["key_fusion"] = mod
    spec.loader.exec_module(mod)
    return mod


def fusion_cv(kf, rows, post_of, key_of):
    """Full-pipeline 5-fold CV: mode head + rerank (refined edma key vs candidate key)."""
    notes = kf.NOTES
    X, y, meta = [], [], []
    for r in rows:
        tonic = notes.index(r["feat"]["edma_tonic"])
        base = (
            list(r["feat"]["cues"])
            + [r["feat"]["edma_conf"]]
            + kf.skey_feats(post_of(r), tonic, r["feat"]["edma_mode"])
        )
        X.append(base)
        y.append(1 if kf.parse(r["ref"])[1] == "minor" else 0)
        meta.append(r)
    X, y = np.array(X), np.array(y)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    best = (-1.0, None, None)
    for mthr in (0.6, 0.7, 0.8, 0.9):
        for rthr in (0.5, 0.6, 0.7, 0.8):
            ws = []
            for tr, te in skf.split(X, y):
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])

                def refined(idx, clf=clf, sc=sc, mthr=mthr):
                    p = clf.predict_proba(sc.transform(X[idx]))[:, 1]
                    out = []
                    for pi, i in zip(p, idx, strict=True):
                        tonic, mode = meta[i]["feat"]["edma_tonic"], meta[i]["feat"]["edma_mode"]
                        if pi >= mthr:
                            mode = "minor"
                        elif pi <= 1 - mthr:
                            mode = "major"
                        out.append(f"{tonic} {mode}")
                    return out

                rk_tr, rk_te = refined(tr), refined(te)

                def rrfeat(idx, rk):
                    f_rows, y_rows = [], []
                    for i, ek in zip(idx, rk, strict=True):
                        r = meta[i]
                        f = list(X[i]) + [1.0 if kf.parse(ek) == kf.parse(key_of(r)) else 0.0]
                        f_rows.append(f)
                        y_rows.append(
                            1 if kf.mirex(r["ref"], key_of(r)) > kf.mirex(r["ref"], ek) else 0
                        )
                    return np.array(f_rows), np.array(y_rows)

                f_tr, y_tr = rrfeat(tr, rk_tr)
                f_te, _ = rrfeat(te, rk_te)
                s2 = StandardScaler().fit(f_tr)
                rr = LogisticRegression(max_iter=1000).fit(s2.transform(f_tr), y_tr)
                pr = rr.predict_proba(s2.transform(f_te))[:, 1]
                preds = [
                    key_of(meta[i]) if p >= rthr else ek
                    for i, p, ek in zip(te, pr, rk_te, strict=True)
                ]
                ws.append(
                    float(
                        np.mean(
                            [kf.mirex(meta[i]["ref"], k) for i, k in zip(te, preds, strict=True)]
                        )
                    )
                )
            m = float(np.mean(ws))
            if m > best[0]:
                best = (m, mthr, rthr)
    return best


def rerank_skey_vs_cnn(kf, rows, cnn_probs_skey_order):
    notes = kf.NOTES
    X, meta = [], []
    for r in rows:
        tonic = notes.index(r["feat"]["edma_tonic"])
        f = list(r["feat"]["cues"]) + [r["feat"]["edma_conf"]]
        f += kf.skey_feats(r["skey"]["posterior"], tonic, r["feat"]["edma_mode"])
        f += kf.skey_feats(cnn_probs_skey_order(r["cnn"]["probs"]), tonic, r["feat"]["edma_mode"])
        f.append(1.0 if kf.parse(r["skey"]["skey_key"]) == kf.parse(r["cnn"]["cnn_key"]) else 0.0)
        X.append(f)
        meta.append(r)
    X = np.array(X)
    y = np.array(
        [
            1
            if kf.mirex(r["ref"], r["cnn"]["cnn_key"]) > kf.mirex(r["ref"], r["skey"]["skey_key"])
            else 0
            for r in meta
        ]
    )
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    best = -1.0
    for thr in (0.5, 0.6, 0.7, 0.8):
        ws = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            pr = m.predict_proba(sc.transform(X[te]))[:, 1]
            preds = [
                meta[i]["cnn"]["cnn_key"] if p >= thr else meta[i]["skey"]["skey_key"]
                for i, p in zip(te, pr, strict=True)
            ]
            ws.append(
                float(
                    np.mean([kf.mirex(meta[i]["ref"], k) for i, k in zip(te, preds, strict=True)])
                )
            )
        best = max(best, float(np.mean(ws)))
    return best


def main() -> int:
    kf = load_fusion_module()
    train = kf.load_split(
        SEL / "gsmtg_labels.jsonl", SEL / "skey_gsmtg.jsonl", SEL / "keyfeat_gsmtg.jsonl"
    )
    oof = {}
    for line in (KEY / "k10_oof_train.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            oof[str(r["track_id"])] = r
    rows = [r for r in train if r["tid"] in oof]
    for r in rows:
        r["cnn"] = oof[r["tid"]]
    notes, skey_map = kf.NOTES, kf.SKEY_MAP

    def cnn_probs_skey_order(probs):
        return [probs[notes.index(n) * 2 + (1 if m == "minor" else 0)] for n, m in skey_map]

    w_a = float(np.mean([kf.mirex(r["ref"], r["cnn"]["cnn_key"]) for r in rows]))
    ref = fusion_cv(kf, rows, lambda r: r["skey"]["posterior"], lambda r: r["skey"]["skey_key"])
    b = fusion_cv(
        kf,
        rows,
        lambda r: cnn_probs_skey_order(r["cnn"]["probs"]),
        lambda r: r["cnn"]["cnn_key"],
    )
    w_c = rerank_skey_vs_cnn(kf, rows, cnn_probs_skey_order)
    result = {
        "basis_n": len(rows),
        "a_cnn_standalone_oof": round(w_a, 4),
        "ref_skey_fusion": {
            "cv_weighted": round(ref[0], 4),
            "mode_thr": ref[1],
            "rerank_thr": ref[2],
        },
        "b_cnn_slot_fusion": {"cv_weighted": round(b[0], 4), "mode_thr": b[1], "rerank_thr": b[2]},
        "c_skey_vs_cnn_rerank": round(w_c, 4),
        "selected": "a (simplest within noise)",
        "ledger_expected": EXPECTED,
    }
    got = {
        "a": result["a_cnn_standalone_oof"],
        "ref": ref and round(ref[0], 4),
        "b": round(b[0], 4),
        "c": round(w_c, 4),
    }
    result["matches_ledger"] = got == EXPECTED
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    if len(rows) != 1129 or not result["matches_ledger"]:
        print("MISMATCH against the ledger values", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
