#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Generate the paper's figures into paper/arxiv/figs/.

Every number is transcribed from paper/EXPERIMENTS.md (ledger IDs in comments) or
paper/STATS.md; the structure per-class CIs regenerate via the st3/st4_cis bootstrap
(track-level resampling, 10k, seed 0) documented in the ledger's ST-v3/ST-v4 entries.
Run: uv run paper/arxiv/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

BLUE, ORANGE, GREEN, RED, GRAY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C"


# ---------------------------------------------------------------- Figure 1: key forest
def fig_key_forest() -> None:
    # STATS.md point estimates + 95% CI (n=567, MIREX weighted, bootstrap 10k seed 0)
    systems = [
        ("edma (template)", 0.7589, 0.7277, 0.7885, GRAY),          # K1
        ("mode retrain", 0.8095, 0.7799, 0.8372, BLUE),              # K3
        ("S-KEY", 0.8168, 0.7887, 0.8434, BLUE),                     # K4
        ("fusion (prior)", 0.8123, 0.7831, 0.8402, BLUE),            # K6
        ("key CNN (ours)", 0.8321, 0.8039, 0.8586, GREEN),           # K10
        ("madmom CNN\n(CC BY-NC-SA)", 0.8328, 0.8063, 0.8580, ORANGE),  # K9
    ]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ys = range(len(systems))
    for y, (name, m, lo, hi, c) in zip(ys, systems):
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=c,
                    capsize=3, markersize=5, elinewidth=1.2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([s[0] for s in systems])
    ax.invert_yaxis()

    # Published full-set numbers (ledger K9 note): best honest published = KeyMyna 0.7591;
    # madmom's own published number = 0.746.
    ax.axvline(0.7591, color=GRAY, ls="--", lw=1)
    ax.text(0.7591, -0.55, "best published\n(full set): 0.759", ha="center", va="bottom",
            fontsize=7.5, color=GRAY)
    # Subset-calibration arrow: madmom published 0.746 -> 0.8328 on our n=567 subset (K9)
    ax.annotate("", xy=(0.8328, 5.42), xytext=(0.746, 5.42),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.plot([0.746], [5.42], marker="s", color=ORANGE, ms=4, clip_on=False)
    ax.text(0.789, 5.62, "+0.087 subset shift (same system)", ha="center", va="top",
            fontsize=7.5, color=ORANGE)
    ax.set_xlim(0.70, 0.875)
    ax.set_ylim(6.15, -1.65)
    ax.set_xlabel("MIREX weighted score, GiantSteps Key (n=567), 95% CI")
    fig.savefig(OUT / "fig_key_forest.pdf")
    plt.close(fig)


# ------------------------------------------------- Figure 2: separation cascade scatter
def fig_cascade() -> None:
    # Ledger S1/S3/S4/S5: SI-SDR (dB) and through-separation note-F with the transcriber
    # fixed (basic-pitch pitched stems, ADTOF drums). htdemucs_ft on the 50-track subset
    # (S3), BS-RoFormer on its 47 matched tracks (S5) -- open markers.
    #            name        marker filled  drums(sdr,f)     bass(sdr,f)     other(sdr,f)
    data = [
        ("HT-Demucs", "o", True, (11.61, 0.5845), (4.57, 0.5957), (10.13, 0.4585)),  # S1
        ("HT-Demucs-ft (n=50)", "^", False, (11.00, 0.6049), (4.83, 0.6070), (10.90, 0.4545)),  # S3
        ("SCNet XL (shipped)", "s", True, (14.31, 0.5741), (5.98, 0.6448), (11.77, 0.4733)),  # S4
        ("BS-RoFormer (n=47)", "D", False, (13.11, 0.5964), (5.74, 0.6283), (8.57, 0.4682)),  # S5
    ]
    colors = [BLUE, BLUE, GREEN, ORANGE]
    panels = [("drums (onset-F)", 3), ("bass (note-F)", 4), ("other (note-F)", 5)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ax, (title, idx) in zip(axes, panels):
        for (name, mk, filled, *stems), c in zip(data, colors):
            sdr, f = stems[idx - 3]
            ax.plot(sdr, f, marker=mk, color=c, ms=6,
                    mfc=c if filled else "white", mew=1.2, ls="none",
                    label=name if idx == 3 else None)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("SI-SDR (dB)")
    axes[0].set_ylabel("transcription F through separation")
    axes[0].annotate("+2.7 dB,\nworse F", xy=(14.2, 0.5745), xytext=(12.55, 0.5775),
                     fontsize=7, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=False,
               fontsize=7.5)
    fig.savefig(OUT / "fig_cascade.pdf")
    plt.close(fig)


# ---------------------------------------------- Figure 3: structure per-class trade-off
def fig_structure_trade() -> None:
    # ST-v3 / ST-v4 per-class paired deltas vs stock, fold-2 held-out (165 tracks);
    # per-track GT-duration coverage, track-level bootstrap 10k seed 0 (st3/st4_cis).
    #      class      n    v3 (d, lo, hi)              v4 (d, lo, hi)
    rows = [
        ("cooldown", 113, (+0.233, +0.157, +0.312), (+0.076, -0.000, +0.151)),
        ("buildup", 115, (-0.158, -0.219, -0.098), (-0.240, -0.308, -0.175)),
        ("end", 140, (+0.014, +0.000, +0.036), (-0.355, -0.435, -0.278)),
        ("outro", 148, (-0.092, -0.143, -0.044), (+0.081, +0.027, +0.135)),
        ("intro", 160, (+0.047, +0.006, +0.091), (+0.078, +0.031, +0.128)),
        ("altoutro", 26, (+0.036, -0.079, +0.181), (+0.123, +0.010, +0.256)),
        ("altintro", 22, (+0.083, +0.000, +0.211), (+0.083, +0.000, +0.211)),
        ("bridge", 4, (+0.000, +0.000, +0.000), (+0.125, +0.000, +0.375)),
        ("breakdown", 164, (+0.007, +0.000, +0.016), (+0.016, +0.002, +0.032)),
        ("drop", 165, (+0.003, -0.007, +0.014), (-0.004, -0.018, +0.010)),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    h = 0.36
    for i, (name, n, v3, v4) in enumerate(rows):
        for off, (d, lo, hi), c, lab in ((-h / 2, v3, BLUE, "ST-v3"),
                                         (+h / 2, v4, ORANGE, "ST-v4")):
            ax.barh(i + off, d, height=h, color=c, alpha=0.85,
                    label=lab if i == 0 else None)
            ax.plot([lo, hi], [i + off, i + off], color="black", lw=0.9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]} (n={r[1]})" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("paired Δ class coverage vs. stock (95% CI)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(OUT / "fig_structure_trade.pdf")
    plt.close(fig)


# ------------------------------------------------------------- Figure 4: key by genre
def fig_key_genre() -> None:
    # K10 CNN genre slice (n>=15 per genre; descriptive post-hoc on cnn_gskey.jsonl)
    genres = [
        ("Dubstep", 0.900), ("Electro House", 0.894), ("Techno", 0.854),
        ("Deep House", 0.846), ("Tech House", 0.812), ("Drum & Bass", 0.804),
        ("House", 0.800), ("Prog. House", 0.772), ("Trance", 0.763),
        ("Electronica", 0.739),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    xs = range(len(genres))
    ax.bar(xs, [g[1] for g in genres], color=BLUE, width=0.65)
    ax.axhline(0.8321, color=GREEN, ls="--", lw=1)
    ax.text(len(genres) - 0.4, 0.8321, "overall 0.832", ha="right", va="bottom",
            fontsize=7.5, color=GREEN)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([g[0] for g in genres], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0.65, 0.90)
    ax.set_ylabel("MIREX weighted")
    fig.savefig(OUT / "fig_key_genre.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_key_forest()
    fig_cascade()
    fig_structure_trade()
    fig_key_genre()
    print(f"wrote 4 figures to {OUT}")
