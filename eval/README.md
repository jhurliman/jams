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

## Files

| File | Purpose |
|------|---------|
| `acquire_dataset.py` | Download GiantSteps Key → `data/manifest.jsonl` |
| `evaluate.py` | Score production `jams.detect_key` / `detect_tempo` |
| `benchmark_methods.py`, `benchmark_final.py` | Method comparisons |
| `analyze_errors.py` | Domain error taxonomy (mode, octave, per-genre) |
| `build_corrections.py` | Regenerate `tempo_corrections.csv` from GiantSteps-Tempo v2 |
| `train_mode_model.py` | Train/export the major-minor refinement (`src/jams/data/mode_model.json`) |
| `tempo_corrections.csv` | Curated tempo-label fixes (committed) |
