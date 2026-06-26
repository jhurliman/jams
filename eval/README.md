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

## Song structure (multi-dataset)

`evaluate_structure.py` scores `jams.analysis.structure` (the local All-In-One backend)
against any dataset via a common manifest built by a per-dataset `acquire_*` script. Each
manifest row carries the `model` to score with and a `format` for loading its annotations;
the evaluator computes the same `mir_eval` metrics across all of them.

| Dataset | Domain | Tracks | Annotations | Model | Audio |
|---------|--------|--------|-------------|-------|-------|
| **Raveform** *(primary)* | EDM / DJ | 1,423 | beats, downbeats, functional segments | `harmonix-all` (out-of-domain) | YouTube id — **annotations made on the same video → native alignment** |
| Harmonix | Western pop | 912 | beats, downbeats, segments | per-fold CV `harmonix-fold{i%8}` (held out) | YouTube — **different master → needs alignment** |
| EDM-98 | EDM | 98 | segments only | `harmonix-all` | *not publicly released yet* |

```sh
uv run --extra eval eval/acquire_raveform.py     # primary: MIT annotations + YouTube audio
uv run --extra eval eval/evaluate_structure.py --manifest eval/data/raveform/manifest.jsonl
uv run --extra eval eval/acquire_harmonix.py     # pop cross-domain (per-fold CV)
uv run --extra eval eval/evaluate_structure.py   # defaults to the Harmonix manifest
```

**Metrics** (`mir_eval`): beats/downbeats F (70 ms); segment boundaries Hit-Rate F @0.5 s /
@3 s; segment labeling pairwise-F + V-measure. Segments-only datasets (EDM-98) skip the beat
metrics automatically. `--target {jams,ref,none}` sets the beat-tracking BPM constraint
(jams' tempo / dataset BPM / none) — the `target_bpm` ablation.

**Model selection.** Harmonix trained All-In-One, so it uses **per-fold CV** — the positional
split `fold(track_i)=i%8` means each track is scored only by the held-out `harmonix-fold{i%8}`.
Raveform/EDM-98 are **out-of-domain test sets** for that model, so they use `harmonix-all`
directly (no leakage; the numbers read as "how well the model generalizes to EDM").

**Harmonix audio alignment.** Harmonix annotations are public but its audio isn't, so we
source YouTube uploads — which are different masters/edits and drift in time. `align_harmonix.py`
fits a per-track affine map `t_audio = a·t_anno + b` + a chance-corrected confidence, classes
each track case1 (offset) / case2 (speed-changed) / case3 (different edit), and writes
`alignment.jsonl`; `evaluate_structure.py` applies the warp and drops case3. **Raveform needs
none of this** — its annotations were made on the same YouTube audio we download.

Preliminary Raveform (4-track, `harmonix-all`, target=ref, native alignment): beat-F **0.63**,
boundary-HR@0.5 **0.53**, pairwise-F **0.70**, V-measure **0.77** — trustworthy timings, and a
baseline for pushing in-domain EDM structure accuracy.

## Files

| File | Purpose |
|------|---------|
| `acquire_dataset.py` | Download GiantSteps Key → `data/manifest.jsonl` |
| `acquire_raveform.py` | Download Raveform (primary EDM structure set) → `data/raveform/manifest.jsonl` |
| `acquire_harmonix.py` | Download Harmonix annotations + YouTube audio → `data/harmonix/manifest.jsonl` |
| `align_harmonix.py` | Fit per-track YouTube↔annotation affine warp + confidence → `alignment.jsonl` |
| `evaluate_structure.py` | Multi-dataset structure scoring (`mir_eval`) + `target_bpm` ablation |
| `evaluate.py` | Score production `jams.detect_key` / `detect_tempo` |
| `benchmark_methods.py`, `benchmark_final.py` | Method comparisons |
| `analyze_errors.py` | Domain error taxonomy (mode, octave, per-genre) |
| `build_corrections.py` | Regenerate `tempo_corrections.csv` from GiantSteps-Tempo v2 |
| `train_mode_model.py` | Train/export the major-minor refinement (`src/jams/data/mode_model.json`) |
| `tempo_corrections.csv` | Curated tempo-label fixes (committed) |
