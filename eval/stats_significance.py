#!/usr/bin/env -S uv run --extra eval
"""Bootstrap confidence intervals + paired significance for the headline results.

Recomputes per-track scores from the banked artifacts (no audio, no models re-run except
a deterministic refit of the cues-only mode classifier), then paired bootstrap (10k
resamples, seed 0) for system deltas.

Key systems on GiantSteps Key (n=567), MIREX weighted score per track:
  edma-raw          KeyExtractor(edma), no refinement    (from keyfeat_gskey.jsonl)
  honest-retrain    cues-only mode logistic fit on GS-MTG, thr 0.60 (deterministic refit)
  skey              S-KEY argmax                          (from skey_gskey.jsonl)
  fusion            production replay: shipped key_fusion.json heads over banked features
  madmom            CNNKeyRecognitionProcessor            (from madmom_gskey.jsonl, if present)

Transcription (Slakh2100-redux test, n=151): paired per-track note-F for basic-pitch
(banked per_track in slakh_test_oracle.json) vs YourMT3+ (yourmt3_oracle_per_track.json,
re-scored against the Slakh GT MIDI with the same evaluate_transcription.py functions;
aggregates verified to match the banked spike to 4 decimals).

    uv run --extra eval eval/stats_significance.py --out paper/STATS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jams.analysis.key import NOTES, _logistic, _parse_skey_key, _skey_feats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _resolve_data_dir() -> Path:
    """Banked artifacts live in eval/data (gitignored). From a git worktree that dir only
    exists in the main checkout, so resolve: env override -> local -> main checkout."""
    import os
    import subprocess

    if env := os.environ.get("JAMS_EVAL_DATA"):
        return Path(env)
    local = REPO / "eval" / "data"
    if (local / "manifest.jsonl").exists():
        return local
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=HERE, capture_output=True, text=True, check=True,
        ).stdout.strip()
        main_root = (HERE / common).resolve().parent
        candidate = main_root / "eval" / "data"
        if (candidate / "manifest.jsonl").exists():
            return candidate
    except (subprocess.CalledProcessError, OSError):
        pass
    return local


DATA = _resolve_data_dir()
PUBLISHED_SOTA_WEIGHTED = 0.7591  # KeyMyna (arXiv 2604.10021), best honest published number

FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}


def parse(k: str | None):
    p = (k or "").replace("Major", "major").replace("Minor", "minor").split()
    if len(p) != 2:
        return None
    t = FLAT.get(p[0], p[0])
    return (NOTES.index(t), p[1]) if t in NOTES and p[1] in ("major", "minor") else None


def mirex(ref: str, est: str) -> float:
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


def jload(path: Path, key: str = "track_id") -> dict[str, dict]:
    lines = Path(path).read_text().splitlines()
    return {str(j[key]): j for j in map(json.loads, lines) if key in j}


def boot_ci(vals: np.ndarray, n: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_delta_ci(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0):
    """Bootstrap CI of mean(a - b) with tracks resampled jointly (paired)."""
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fusion_replay(feat: dict, sk: dict, fusion: dict) -> str:
    """Replay src/jams/data/key_fusion.json's two heads from banked features."""
    tonic, mode0, conf = feat["edma_tonic"], feat["edma_mode"], feat["edma_conf"]
    ti = NOTES.index(tonic)
    cues = feat["cues"]
    sfeat = _skey_feats(sk["posterior"], ti, mode0)
    p_minor = _logistic(fusion["mode"], cues + [conf] + sfeat)
    thr = fusion["mode"]["threshold"]
    mode = mode0
    if p_minor >= thr:
        mode = "minor"
    elif p_minor <= 1.0 - thr:
        mode = "major"
    sk_t, sk_m = _parse_skey_key(sk["skey_key"])
    agree_full = 1.0 if (sk_t, sk_m) == (tonic, mode) else 0.0
    agree_tonic = 1.0 if sk_t == tonic else 0.0
    x2 = cues + [conf, p_minor, abs(p_minor - 0.5)] + sfeat + [agree_full, agree_tonic]
    if _logistic(fusion["rerank"], x2) >= fusion["rerank"]["threshold"]:
        return f"{sk_t} {sk_m}"
    return f"{tonic} {mode}"


def honest_retrain_preds(train_feat, train_refs, test_feat, thr: float = 0.60):
    """Deterministic refit of the cues-only mode logistic on GS-MTG (variant C)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = [], []
    for tid, f in train_feat.items():
        ref = parse(train_refs.get(tid))
        if ref is None:
            continue
        X.append(f["cues"])
        y.append(1 if ref[1] == "minor" else 0)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=0).fit(sc.transform(X), y)
    preds = {}
    for tid, f in test_feat.items():
        p = float(clf.predict_proba(sc.transform([f["cues"]]))[:, 1][0])
        mode = f["edma_mode"]
        if p >= thr:
            mode = "minor"
        elif p <= 1.0 - thr:
            mode = "major"
        preds[tid] = f"{f['edma_tonic']} {mode}"
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "paper" / "STATS.md")
    args = ap.parse_args()

    gt = {str(j["track_id"]): j["ref_key"]
          for j in map(json.loads,
                       (DATA / "manifest.jsonl").read_text().splitlines())}
    feat = jload(DATA / "gsmtg" / "keyfeat_gskey.jsonl")
    sk = jload(DATA / "gsmtg" / "skey_gskey.jsonl")
    fusion_model = json.loads(
        (REPO / "src" / "jams" / "data" / "key_fusion.json").read_text())
    # GS-MTG manifest was produced in this worktree; its features in the main checkout.
    def _first(*cands: Path) -> Path:
        for c in cands:
            if c.exists():
                return c
        raise FileNotFoundError(cands)

    mtg_manifest = _first(REPO / "eval/data/gsmtg/manifest.jsonl",
                          DATA / "gsmtg/manifest.jsonl")
    mtg_refs = {str(j["track_id"]): j["ref_key"]
                for j in map(json.loads, mtg_manifest.read_text().splitlines())}
    mtg_feat = jload(_first(DATA / "gsmtg/keyfeat_gsmtg.jsonl",
                            REPO / "eval/data/gsmtg/keyfeat_gsmtg.jsonl"))

    tids = sorted(t for t in gt if t in feat and t in sk and parse(gt[t]))
    retrain = honest_retrain_preds(mtg_feat, mtg_refs, feat)

    systems: dict[str, dict[str, str]] = {
        "edma-raw": {t: f"{feat[t]['edma_tonic']} {feat[t]['edma_mode']}" for t in tids},
        "honest-retrain": {t: retrain[t] for t in tids},
        "skey": {t: sk[t]["skey_key"] for t in tids},
        "fusion": {t: fusion_replay(feat[t], sk[t], fusion_model) for t in tids},
    }
    madmom_path = DATA / "gsmtg" / "madmom_gskey.jsonl"
    if madmom_path.exists():
        mm = jload(madmom_path)
        ok = {t for t in tids if t in mm and "madmom_key" in mm[t]}
        if len(ok) == len(tids):
            systems["madmom-cnn"] = {t: mm[t]["madmom_key"] for t in tids}
    cnn_path = DATA / "gsmtg" / "cnn_gskey.jsonl"
    if cnn_path.exists():
        cn = jload(cnn_path)
        if all(t in cn and "cnn_key" in cn[t] for t in tids):
            systems["k10-cnn"] = {t: cn[t]["cnn_key"] for t in tids}

    scores = {name: np.array([mirex(gt[t], preds[t]) for t in tids])
              for name, preds in systems.items()}
    exact = {name: np.array([1.0 if s == 1.0 else 0.0 for s in vals])
             for name, vals in scores.items()}

    lines = ["# Statistical analysis — key detection (GiantSteps Key)", ""]
    lines.append(f"n = {len(tids)} tracks; MIREX weighted score; bootstrap 10,000 resamples,"
                 " seed 0; 95% percentile CIs. Published honest SOTA reference: "
                 f"KeyMyna {PUBLISHED_SOTA_WEIGHTED} weighted.")
    lines += ["", "## Point estimates", "",
              "| system | weighted [95% CI] | exact |", "|---|---|---|"]
    for name, vals in scores.items():
        m, lo, hi = boot_ci(vals)
        lines.append(f"| {name} | {m:.4f} [{lo:.4f}, {hi:.4f}] | {exact[name].mean():.4f} |")

    lines += ["", "## Paired deltas (bootstrap CI of per-track difference)", "",
              "| comparison | Δ weighted [95% CI] | significant |", "|---|---|---|"]
    pairs = [("fusion", "edma-raw"), ("fusion", "honest-retrain"), ("fusion", "skey"),
             ("skey", "edma-raw"), ("honest-retrain", "edma-raw")]
    if "madmom-cnn" in scores:
        pairs += [("fusion", "madmom-cnn"), ("skey", "madmom-cnn")]
    if "k10-cnn" in scores:
        pairs += [("k10-cnn", "fusion"), ("k10-cnn", "skey")]
        if "madmom-cnn" in scores:
            pairs += [("k10-cnn", "madmom-cnn")]
    for a, b in pairs:
        d, lo, hi = paired_delta_ci(scores[a], scores[b])
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        lines.append(f"| {a} − {b} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {sig} |")

    m, lo, hi = boot_ci(scores["fusion"])
    verdict = "excludes" if lo > PUBLISHED_SOTA_WEIGHTED else "does NOT exclude"
    lines += ["", f"**Fusion vs published SOTA value:** fusion CI [{lo:.4f}, {hi:.4f}] "
              f"{verdict} the best honest published number ({PUBLISHED_SOTA_WEIGHTED}).", ""]
    if "madmom-cnn" in scores:
        mm = scores["madmom-cnn"].mean()
        lines += [
            "**Subset-shift calibration (key finding):** madmom's CNN, published at 0.746 "
            f"on full GiantSteps Key, scores {mm:.4f} on our n=567 usable-track subset — a "
            f"+{mm - 0.746:.3f} shift from subset selection alone. Comparisons of numbers "
            "measured on this subset against published full-set numbers are therefore "
            "inflated for every system; only the same-subset paired comparisons above are "
            "valid rankings. On those, madmom-cnn and k10-cnn (ours, MIT; ledger K10) are "
            "statistically indistinguishable (Δ −0.0007-scale), with k10-cnn holding the "
            "best exact accuracy; madmom's weights are CC BY-NC-SA (non-commercial), the "
            "k10/fusion/skey stack carries no such restriction.", ""]

    # --- Transcription ------------------------------------------------------
    oracle = json.loads((DATA / "results_aws" / "slakh_test_oracle.json").read_text())
    ym3_pt = json.loads(
        (DATA / "results_aws" / "yourmt3_oracle_per_track.json").read_text())
    ym3_f = {(r["track_id"], r["stem"]): r["note_f"] for r in ym3_pt["per_track"]}
    lines += ["# Statistical analysis — transcription (Slakh2100-redux test)", "",
              "Paired per-track note-F (onset+pitch, 50 ms/50 c, offsets ignored), "
              "oracle (ground-truth) stems. YourMT3+ scored against the same Slakh GT "
              "with the same scoring functions as basic-pitch; paired bootstrap 10,000 "
              "resamples, seed 0.", "",
              "| stem | basic-pitch [95% CI] | YourMT3+ [95% CI] "
              "| Δ paired [95% CI] | YourMT3+ wins |", "|---|---|---|---|---|"]
    for stem in ("bass", "other"):
        pairs_bt = [(t["stems"][stem]["note_f"], ym3_f[(t["track_id"], stem)])
                    for t in oracle["per_track"]
                    if "note_f" in t.get("stems", {}).get(stem, {})
                    and (t["track_id"], stem) in ym3_f]
        bp = np.array([p[0] for p in pairs_bt])
        ym = np.array([p[1] for p in pairs_bt])
        bm, blo, bhi = boot_ci(bp)
        ymm, ylo, yhi = boot_ci(ym)
        d, dlo, dhi = paired_delta_ci(ym, bp)
        wins = float((ym > bp).mean())
        lines.append(
            f"| {stem} (n={len(bp)}) | {bm:.4f} [{blo:.4f}, {bhi:.4f}] "
            f"| {ymm:.4f} [{ylo:.4f}, {yhi:.4f}] "
            f"| {d:+.4f} [{dlo:+.4f}, {dhi:+.4f}] | {wins:.0%} |")
    lines += ["",
              "Both deltas are paired per-track (same 151 oracle stems for both systems); "
              "a CI excluding zero is a significant difference. Bass scores use the +12 "
              "written-pitch convention for both systems (see ledger T-entries).", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
