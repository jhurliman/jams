# Paper draft skeleton

**Working titles**
1. "jams: An Open, Reproducible MIR Stack for Electronic Dance Music, at or beyond
   Published State of the Art"
2. "Honest Key Detection: Test-Set Hygiene, Self-Supervised Fusion, and a New GiantSteps
   State of the Art"
3. (LBD framing) "An Open EDM Analysis Service: Key, Structure, Stems and MIDI with
   Auditable Benchmarks"

**Format targets**: arXiv preprint (full, ~8 pp) → condensed 4-page ISMIR LBD; blog post
adapted from the same material.

---

## Abstract (draft — revised after the madmom calibration finding)

We present jams, an open-source music-information-retrieval service for electronic dance
music that couples every analysis capability — key, tempo, beats, structural segmentation,
source separation, and per-stem MIDI transcription — to a reproducible evaluation harness,
and we report two evaluation-hygiene findings that reframe how key-detection results are
compared. First, auditing our own pipeline surfaced test-set contamination (a learned
refinement fit on GiantSteps Key itself); re-training under a strict protocol (fit on
GiantSteps-MTG-Keys, evaluate once on GiantSteps Key) showed the contaminated number had
in fact been honest — but only the audit could tell. Second, and more consequentially:
re-running the strongest published system (madmom's CNN, published 0.746 weighted) on our
n=567 usable-track subset yields 0.833 — a +0.087 shift from subset selection alone —
demonstrating that the field's routine comparison of new results against published
GiantSteps numbers is miscalibrated by more than the total claimed progress of recent
systems. Under same-subset paired bootstrap comparisons, our fusion of an EDM-tuned
template method with a self-supervised tonality estimator (S-KEY) is statistically
indistinguishable from the madmom CNN (Δ −0.021 [−0.044, +0.003]) while being fully
permissively licensed, and achieves the best exact accuracy (0.757). On transcription, we
show that feeding separated stems to a multi-instrument transformer (YourMT3+) yields
0.849 note-F on both bass and dense polyphonic accompaniment on the Slakh2100 test split —
a 36-point improvement over a widely-used lightweight transcriber — and we quantify the
end-to-end cost of separation (SCNet XL) at each stage: the full mix→MIDI system reaches
0.788 note-F on accompaniment — above the lightweight transcriber's *ground-truth-stem*
ceiling (0.490) — including a case where +2.7 dB SI-SDR *reduces* downstream
drum-transcription accuracy. [TODO: structure fine-tune sentence.] All code, evaluation scripts, per-track artifacts, and statistical analyses are
public.

## 1 Introduction

- DJ/EDM-focused MIR service; production constraints (on-device, cross-platform,
  no silent quality degradation).
- Core theme: *auditable* SOTA — every number regenerable from committed scripts; the
  contamination we found in our own pipeline as motivation for protocol rigor.
- Contributions: (1) honest-protocol key results beating published SOTA, with a
  contamination case study + significance analysis; (2) stem-input evaluation of
  multi-instrument transcription with separation-cascade costs; (3) [pending] EDM structure
  fine-tune with class-balanced function loss; (4) the open system + harness itself.

## 2 Related Work

- **Key**: template methods (Krumhansl; edma profile — Faraldo et al.); CNNs (Korzeniowski
  & Widmer ISMIR 2018 = madmom; InceptionKeyNet ISMIR 2021); SSL tonality (STONE, S-KEY —
  Kong et al., ICASSP 2025); masked-contrastive (KeyMyna, arXiv 2604.10021); datasets:
  GiantSteps Key (Knees et al. 2015), GiantSteps-MTG-Keys (Faraldo).
- **Transcription**: basic-pitch (Bittner et al., ICASSP 2022); MT3 (Gardner et al.);
  PerceiverTF; YourMT3+ (Chang et al., MLSP 2024); Slakh2100 (Manilow et al.).
- **Separation**: HT-Demucs (Rouard et al.); BS-RoFormer (Lu et al.); Mel-RoFormer; SCNet
  (Tong et al.); MSST zoo (ZFTurbo).
- **Structure**: All-In-One (Kim & Nam, WASPAA 2023); Harmonix Set (Nieto et al.);
  [Raveform citation].
- **Evaluation hygiene**: train/test leakage discussions in MIR; mir_eval (Raffel et al.).

## 3 System

- Architecture: Python 3.14 core (essentia-tensorflow) + isolated uv workers per heavy
  model (torch), JSONL IPC; no-silent-fallback contract (failures raise, quality never
  silently varies with environment).
- Capabilities table with method per lane (edma+S-KEY fusion / TempoCNN / All-In-One /
  SCNet XL / YourMT3+ + ADTOF drums w/ measured velocities / GM MIDI assembly).
- Web annotator as demo + label-editing tool (LBD asset).

## 4 Experimental setup

Condensed from paper/EXPERIMENTS.md: datasets, splits, metrics, tolerances, the
GS-MTG→GS-Key protocol, pre-registered structure gate. Statistical method: 10k-resample
paired bootstrap, 95% percentile CIs (paper/STATS.md).

## 5 Results

- **5.1 Key** (Table 1 = EXPERIMENTS.md K-table + STATS.md CIs). Findings, in order of
  importance: (a) **calibration**: the strongest published system re-run on our usable
  subset gains +0.087 over its published number — published-number comparisons are
  miscalibrated by more than recent claimed progress; only same-subset paired tests rank
  systems. (b) Under those tests, retrain/S-KEY/fusion/madmom are pairwise
  indistinguishable and all significantly above edma-raw; fusion is the best
  permissively-licensed system and best exact accuracy. (c) Contamination case study: the
  contaminated 0.801 turned out to be honestly reproducible (0.8095) — leakage masked a
  real result; only the audit could tell. (d) Genre analysis: weakest on Trance/Techno
  (tonal ambiguity), no genre collapse.
- **5.2 Transcription** (Table 2 — oracle AND full-system e2e, from EXPERIMENTS.md
  T1/T2/T10). Stem-input YourMT3+ ≈ its mixture-input published F; 36-pt jump on
  polyphonic accompaniment vs basic-pitch (paired Δ +0.359 [+0.347, +0.371], 100% of
  tracks); bass written-octave convention (+12) as a scoring pitfall (0.04→0.80 under
  50-cent tolerance). Full mix→MIDI system (SCNet → YourMT3+/ADTOF): other 0.788 /
  bass 0.661 / drums onset 0.574 — separation costs YourMT3+ 6.1 pt on other but 18.7 pt
  on bass (low-frequency bleed); **e2e other (0.788) exceeds basic-pitch's oracle
  (0.490)**. Cells measured: both transcribers oracle; e2e for the shipped system and for
  SCNet+basic-pitch (S4); htdemucs+YourMT3 e2e n/a (not run — separator already selected).
- **5.3 Separation cascade** (Table 3). SCNet XL wins SI-SDR on every stem AND
  through-separation note-F on pitched stems; drums onset-F dips 1 pt (transcriber
  sensitivity to separator transients) — metric divergence SDR vs downstream-task.
- **5.4 Structure** [TODO — held-out CV table vs all-all baseline, gate outcomes,
  buildup/cooldown confusion recovery].
- **5.5 Ablations**: quantization costs accuracy (stylistic transform only); babyslakh
  16 kHz bandwidth artifact on drum scores; threshold sweep.

## 6 Limitations

- GS Key n=567 usable subset (vs ~600 in literature) — same dataset, not identical tracks;
  published comparisons are point values, not paired tests.
- S-KEY in-repo checkpoint outperforms its paper (72.1 → 81.7 here): likely newer weights
  + subset differences; we cannot attribute precisely.
- Slakh is synthetic (sample-rendered); vocals unmeasured pending MedleyDB full audio
  (n=1 smoke: 0.58 note-F); drums transcriber (ADTOF) evaluated out-of-domain on E-GMD
  e-kits (0.645).
- Structure training: single run per fold, no seed variance; Raveform-only in-domain eval.
- madmom baseline weights are CC BY-NC-SA (evaluation-only use here).

## 7 Conclusion

Auditable, protocol-strict evaluation is cheap relative to model work and changes
conclusions; open EDM MIR can match or beat published SOTA with composed open components.

## Assets checklist

- [x] paper/EXPERIMENTS.md (ledger) · [x] paper/STATS.md (CIs) · [x] structure results
  (negative-results arc v1→v4, per-class CI addendum)
- [x] madmom baseline row · [ ] MedleyDB vocals (data access — documented limitation)
- [x] figures: genre bars, key-system CI forest, cascade SDR-vs-noteF scatter, structure
  class-trade chart (`paper/arxiv/make_figures.py`)
- [x] bibtex file · [ ] LBD 4-page cut (call opens ~Aug) · [ ] blog post draft

## Full draft

**The complete arXiv draft lives in `paper/arxiv/`** (main.tex + references.bib +
make_figures.py; build with `latexmk -pdf main.tex`). This file remains the skeleton /
planning record; the tex is the source of truth for prose.
