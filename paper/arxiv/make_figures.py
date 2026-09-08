#!/usr/bin/env python3
"""Render tables and a figure from the reported summary, NOT raw predictions.

Run from any directory: python paper/arxiv/make_figures.py
Requires matplotlib. For archive validation/recomputation see paper/verify_results.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main() -> None:
    data = json.loads((HERE.parent / "results_snapshot.json").read_text())
    rows = data["transcription"]
    if data["status"] not in ("reported_not_recomputed", "stored_scores_recomputed"):
        raise ValueError("Review provenance labeling before changing the snapshot status")
    tables = HERE / "tables"
    figures = HERE / "figs"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    lines = [r"\begin{tabular}{llll}", r"\toprule",
             r"Input & Transcriber & Mean F1 [95\% CI] & Run \\", r"\midrule"]
    for r in rows:
        score = f"{r['mean']:.4f}"
        if r["ci"]:
            score += f" [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"
        lines.append(f"{r['input']} & {r['model']} & {score} & {r['id']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "transcription.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{SI-SDR (dB)} & \multicolumn{3}{c}{Downstream F1} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Separator & Drums & Bass & Other & Drums & Bass & Other \\", r"\midrule"]
    for r in data["separation"]:
        values = [f"{r[g + '_si_sdr']:.2f}" for g in ("drums", "bass", "other")]
        values += [f"{r[g + '_f1']:.4f}" for g in ("drums", "bass", "other")]
        lines.append(" & ".join([r["model"], *values]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "separation.tex").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.2, 2.7), layout="constrained")
    colors = {"basic-pitch": "#286090", "YourMT3+": "#AB4B13"}
    for model, offset in (("basic-pitch", 0.10), ("YourMT3+", -0.10)):
        selected = [r for r in rows if r["model"] == model]
        for i, r in enumerate(selected):
            y = i + offset
            ax.plot(r["mean"], y, "o", color=colors[model], label=model if i == 0 else None)
            if r["ci"]:
                ax.errorbar(r["mean"], y,
                            xerr=[[r["mean"] - r["ci"][0]], [r["ci"][1] - r["mean"]]],
                            fmt="none", color=colors[model], capsize=4)
            ax.text(r["mean"], y - 0.17, f"{r['mean']:.4f}", ha="center", fontsize=9,
                    color=colors[model])
    ax.set_yticks([0, 1], ["Reference other", "SCNet other"])
    ax.set_ylim(1.55, -0.55)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean onset-and-pitch F1")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=9)
    fig.savefig(figures / "fig_transcription.pdf", metadata={"CreationDate": None,
        "Title": "Reported Slakh2100 accompaniment transcription results"})
    plt.close(fig)
    print(f"Rendered 2 tables and 1 figure from the {data['status']} snapshot; no experiments rerun.")


if __name__ == "__main__":
    main()
