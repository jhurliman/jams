# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Held-out weighted score per epoch for the five K10 cross-validation folds, from the
run's own per-epoch histories (paper/evidence/key/k10_train/fold{k}_hist.json).

Usage: uv run eval/mirex2026_key/abstract/make_fold_figure.py  -> fold_curves.pdf
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REC = HERE.parents[2] / "paper" / "evidence" / "key" / "k10_train"
cv = json.loads((REC / "cv_summary.json").read_text())

fig, ax = plt.subplots(figsize=(6.6, 2.1))
for k in range(5):
    hist = json.loads((REC / f"fold{k}_hist.json").read_text())
    ep = [h["epoch"] for h in hist]
    val = [h["val_weighted"] for h in hist]
    best = cv["folds"][k]
    (line,) = ax.plot(ep, val, lw=1.1, label=f"fold {k}")
    ax.plot(best["best_epoch"], best["best_weighted"], "o", ms=4, color=line.get_color())
ax.axvline(cv["median_best_epoch"], color="k", ls="--", lw=0.8)
ax.text(
    cv["median_best_epoch"] + 0.6,
    0.12,
    f"median best epoch = {cv['median_best_epoch']}",
    fontsize=7,
)
ax.set_xlabel("epoch", fontsize=8)
ax.set_ylabel("held-out weighted score", fontsize=8)
ax.set_ylim(0.1, 0.8)
ax.tick_params(labelsize=7)
ax.grid(alpha=0.3, lw=0.5)
ax.legend(fontsize=7, ncol=5, loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(HERE / "fold_curves.pdf")
print("wrote", HERE / "fold_curves.pdf")
