"""K10 — THE single pre-registered test evaluation on GiantSteps Key (n=567).

Running this script spends the one shot (ledger K10, commit e9ab795). It:
  1. runs the LOCKED model (path+width passed in) on every manifest track,
  2. banks per-track predictions (cnn_gskey.jsonl),
  3. scores weighted/exact + 95% bootstrap CI,
  4. paired bootstrap vs banked madmom (K9) and production fusion (K6 replay),
  5. prints the gate verdict per the pre-registered decision rule.

Usage: uv run --extra eval python k10_test_shot.py --model final.pt --width 1.0
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

T = Path("/Users/jhurliman/.claude/jobs/7eb03476/tmp")
W = Path("/Users/jhurliman/Documents/Code/jhurliman/jams/.claude/worktrees/hard-deps")
DATA = Path("/Users/jhurliman/Documents/Code/jhurliman/jams/eval/data")

spec2 = importlib.util.spec_from_file_location("tkc", W / "eval/train_key_cnn.py")
tkc = importlib.util.module_from_spec(spec2)
sys.modules["tkc"] = tkc
spec2.loader.exec_module(tkc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--stage", choices=("infer", "score"), required=True)
    ap.add_argument("--out", type=Path, default=DATA / "gsmtg/cnn_gskey.jsonl")
    args = ap.parse_args()
    man = [json.loads(l) for l in open(DATA / "manifest.jsonl")]

    if args.stage == "infer":
        do_infer(args, man)
        return
    do_score(args, man)


def do_infer(args, man) -> None:
    import librosa
    import torch

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = tkc.build_model(args.width)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(dev).eval()

    n_bins = (tkc.N_OCT * 12 + 2 * tkc.PAD_SEMI) * (tkc.BINS_PER_OCT // 12)
    fmin = librosa.note_to_hz("C1") * 2 ** (-tkc.PAD_SEMI / 12)
    per = tkc.BINS_PER_OCT // 12
    core = tkc.N_OCT * tkc.BINS_PER_OCT

    with open(args.out, "w") as out:
        for i, r in enumerate(man):
            y, _ = librosa.load(r["audio_path"], sr=tkc.SR, mono=True)
            C = np.log1p(np.abs(librosa.cqt(y, sr=tkc.SR, hop_length=tkc.HOP, fmin=fmin,
                                            n_bins=n_bins, bins_per_octave=tkc.BINS_PER_OCT)))
            X = C[tkc.PAD_SEMI * per: tkc.PAD_SEMI * per + core].astype(np.float32)
            with torch.no_grad():
                probs = torch.softmax(model(torch.from_numpy(X)[None, None].to(dev)), -1)[0].cpu()
            cls = int(probs.argmax())
            key = f"{tkc.NOTES[cls // 2]} {'minor' if cls % 2 else 'major'}"
            out.write(json.dumps({"track_id": str(r["track_id"]), "cnn_key": key,
                                  "probs": [round(float(v), 6) for v in probs]}) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(man)}")


def do_score(args, man) -> None:
    spec = importlib.util.spec_from_file_location("ss", W / "eval/stats_significance.py")
    ss = importlib.util.module_from_spec(spec)
    sys.modules["ss"] = ss
    spec.loader.exec_module(ss)
    preds = {j["track_id"]: j["cnn_key"] for j in map(json.loads, open(args.out))}
    # ---- scoring (same replay/scoring path as STATS.md) ----
    feat = ss.jload(DATA / "gsmtg/keyfeat_gskey.jsonl")
    sk = ss.jload(DATA / "gsmtg/skey_gskey.jsonl")
    mm = ss.jload(DATA / "gsmtg/madmom_gskey.jsonl")
    man_by = {str(r["track_id"]): r for r in man}
    fusion = json.load(open(W / "src/jams/data/key_fusion.json"))
    tids = sorted(set(feat) & set(sk) & set(mm) & set(man_by) & set(preds))
    print(f"scored n = {len(tids)}")

    refs = {t: man_by[t]["ref_key"] for t in tids}
    s_cnn = np.array([ss.mirex(refs[t], preds[t]) for t in tids])
    s_mm = np.array([ss.mirex(refs[t], mm[t]["madmom_key"]) for t in tids])
    s_fus = np.array([ss.mirex(refs[t], ss.fusion_replay(feat[t], sk[t], fusion)) for t in tids])

    m, lo, hi = ss.boot_ci(s_cnn)
    print(f"\nK10 CNN:    weighted {m:.4f} [{lo:.4f}, {hi:.4f}]  exact {(s_cnn == 1).mean():.4f}")
    print(f"madmom K9:  weighted {s_mm.mean():.4f}  exact {(s_mm == 1).mean():.4f}")
    print(f"fusion K6:  weighted {s_fus.mean():.4f}  exact {(s_fus == 1).mean():.4f}")

    for name, other in (("madmom", s_mm), ("fusion", s_fus)):
        d, dlo, dhi = ss.paired_delta_ci(s_cnn, other)
        wins = int((s_cnn > other).sum()); losses = int((s_cnn < other).sum())
        sig = "SIGNIFICANT" if (dlo > 0 or dhi < 0) else "ns"
        print(f"CNN − {name}: Δ {d:+.4f} [{dlo:+.4f}, {dhi:+.4f}] {sig}  "
              f"(wins {wins} / losses {losses} / ties {len(tids) - wins - losses})")

    d, dlo, dhi = ss.paired_delta_ci(s_cnn, s_mm)
    if dlo > 0:
        verdict = "GATE: SIGNIFICANT SUPERIORITY over madmom — claim allowed."
    elif d > 0:
        verdict = "GATE: point lead over madmom, not significant — report as point lead."
    else:
        verdict = "GATE: no lead over madmom — negative result, report as-is."
    print("\n" + verdict)


if __name__ == "__main__":
    main()
