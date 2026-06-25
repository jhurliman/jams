# Eval harness — reproduce & push the accuracy

Benchmarks the production `jams.detect_key` / `jams.detect_tempo` against expert-labeled
ground truth in the electronic/DJ domain. All scripts run in the project env (so they
import `jams` directly) with the `eval` extra:

```sh
uv run --extra eval eval/acquire_dataset.py     # download GiantSteps Key -> eval/data/manifest.jsonl
uv run --extra eval eval/evaluate.py            # score detect_key / detect_tempo (applies corrections)
uv run --extra eval eval/analyze_errors.py      # domain error breakdown (mode, octave, per-genre)
uv run --extra eval eval/benchmark_methods.py   # compare essentia key profiles + tempo algos
uv run --extra eval eval/benchmark_final.py     # final shoot-out incl. TempoCNN + ensembles
uv run --extra eval eval/build_corrections.py   # regenerate tempo_corrections.csv
```

## Dataset

[**GiantSteps Key**](https://github.com/GiantSteps/giantsteps-key-dataset) — 600 EDM
Beatport previews with expert key labels (+ Beatport tempo), loaded via `mirdata`. Audio
is freely downloadable from Zenodo. 567 usable after dropping 33 atonal/ambiguous labels;
458 have tempo. `eval/data/` is regenerable and gitignored.

## Headline results (full set)

| Metric | librosa baseline | **jams (SOTA)** |
|--------|------------------|-----------------|
| Key MIREX | 0.614 | **0.801** |
| Key exact | 0.529 | **0.743** |
| Tempo Acc1 (raw labels) | 0.830 | 0.921 |
| **Tempo Acc1 (corrected labels + full-tempo)** | 0.830 | **0.965** |

Methods: key = Essentia `edma`; tempo = pretrained TempoCNN `deepsquare` + genre-aware
octave resolution. Chosen by `benchmark_*`; both beat librosa, RhythmExtractor2013,
Percival, and madmom (see git history / the comparison scripts).

## Label corrections

The GiantSteps **Key** tempo labels are *wrong* for half-time genres (D&B labeled ~87 when
the true tempo is ~174). `tempo_corrections.csv` (committed, curated) fixes them and is
applied by `evaluate.py --corrections` (on by default). Built by `build_corrections.py`:

1. **Authoritative** — GiantSteps **Tempo** v2 (Schreiber's expert re-annotation of the
   same Beatport tracks), joined by Beatport ID. On the 23-track overlap our model matched
   v2 (not the Key label) on 15/19 disagreements — the model was right, the labels wrong.
2. **Convention** — D&B/dubstep tracks labeled in clean half-time, doubled to full tempo.
   `needs_review=yes`; confirm by ear.

## Where the remaining errors are (`analyze_errors.py`)

- **Key — mode confusion (addressed).** Was 11.8% parallel errors, 65/67 minor→major.
  Fixed by a learned major/minor refinement (`train_mode_model.py` → `mode_model.json`):
  a logistic classifier over chroma cues (third / 6th / 7th / bass-third) that overrides
  edma's mode only when confident. CV MIREX 0.759→0.801, exact 0.688→0.743.
  Further levers: more features (edma strength, beat-synchronous chroma), or a deep key model.
- **Tempo — half/double-time**, fixed by octave resolution + label corrections (above).
  Residual D&B (~0.79) is tracks not covered by v2; extend `tempo_corrections.csv` by ear.

## Song structure (Harmonix, per-fold CV)

`acquire_harmonix.py` + `evaluate_structure.py` benchmark `jams.analysis.structure`
(the local All-In-One backend) against the **Harmonix Set** — the standard
beats/downbeats/segments benchmark, and the set All-In-One was trained on.

```sh
uv run --extra eval eval/acquire_harmonix.py                  # annotations + YouTube audio → manifest
uv run --extra eval eval/evaluate_structure.py                # per-fold CV, mir_eval metrics
uv run --extra eval eval/evaluate_structure.py --target none  # target_bpm ablation (off)
```

**Honest cross-validation.** All-In-One ships 8 fold models. The split is positional —
`fold(track_i) = i % 8` over the sorted track list — so each track is scored *only* by the
held-out model `harmonix-fold{i%8}` it never trained on. `acquire_harmonix.py` reproduces
that exact split and records each track's fold; `evaluate_structure.py` uses it.

**Audio caveat (important).** Harmonix annotations are public; the **audio is not**
(copyright). We source each track from its YouTube URL (`yt-dlp` → m4a) and keep only tracks
with alignment ≥ 0.95. Even so, YouTube uploads are different masters/edits than Harmonix's
originals, so **absolute beat/boundary timing drifts** track-by-track — and no single shift
reconciles beats *and* segments (`--align` is a diagnostic only). Treat beat-F / boundary-HR
on this audio as a **lower bound**; the **segment-labeling** metrics (pairwise-F, V-measure)
are the robust cross-domain signal. Paper-comparable timing needs Harmonix's own audio.

**Metrics** (`mir_eval`): beats/downbeats F (70 ms); segment boundaries Hit-Rate F @0.5 s /
@3 s; segment labeling pairwise-F + V-measure. `--target {jams,ref,none}` sets the
beat-tracking BPM constraint (jams' tempo / Harmonix BPM / none) for the `target_bpm` ablation.

Validation (7-track YouTube sample, per-fold CV, raw timings): pairwise-F **0.57**,
V-measure **0.57** (up to **0.90 / 0.87** on cleanly-matched tracks); beat/boundary timing
depressed by the audio caveat above. Harmonix is Western **pop** — an in-domain EDM structure
set is future work.

## Files

| File | Purpose |
|------|---------|
| `acquire_dataset.py` | Download GiantSteps Key → `data/manifest.jsonl` |
| `acquire_harmonix.py` | Download Harmonix annotations + YouTube audio → `data/harmonix/manifest.jsonl` |
| `evaluate_structure.py` | Per-fold-CV structure scoring (`mir_eval`) + `target_bpm` ablation |
| `evaluate.py` | Score production `jams.detect_key` / `detect_tempo` |
| `benchmark_methods.py`, `benchmark_final.py` | Method comparisons |
| `analyze_errors.py` | Domain error taxonomy (mode, octave, per-genre) |
| `build_corrections.py` | Regenerate `tempo_corrections.csv` from GiantSteps-Tempo v2 |
| `train_mode_model.py` | Train/export the major-minor refinement (`src/jams/data/mode_model.json`) |
| `tempo_corrections.csv` | Curated tempo-label fixes (committed) |
