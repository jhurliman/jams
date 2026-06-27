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

**Raveform is the structure benchmark.** Harmonix is supported in code but **disabled by
default** — see *Harmonix status* below for why.

| Dataset | Domain | Tracks | Annotations | Model | Audio | Default |
|---------|--------|--------|-------------|-------|-------|---------|
| **Raveform** | EDM / DJ | 1,423 | beats, downbeats, functional segments | `harmonix-all` (out-of-domain) | YouTube id — **annotations made on the same video → native alignment** | ✅ |
| Harmonix | Western pop | 912 | beats, downbeats, segments | per-fold CV `harmonix-fold{i%8}` | YouTube — **different master → misaligned** | ❌ |
| EDM-98 | EDM | 98 | segments only | `harmonix-all` | *not publicly released* | — |

```sh
uv run --extra eval eval/acquire_raveform.py    # MIT annotations + YouTube audio → manifest
uv run --extra eval eval/evaluate_structure.py  # scores Raveform by default
```

**Metrics** (`mir_eval`): beats/downbeats F (70 ms); segment boundaries Hit-Rate F @0.5 s /
@3 s; segment labeling pairwise-F + V-measure. Segments-only datasets (EDM-98) skip the beat
metrics automatically. `--target {jams,ref,none}` sets the beat-tracking BPM constraint
(jams' tempo / dataset BPM / none) — the `target_bpm` ablation. `harmonix-all` is used for
out-of-domain sets (no fold/CV leakage; the numbers read as "how well the model generalizes
to EDM").

### Harmonix status — disabled by default

Harmonix's annotations are public but **its audio is not**, so we source YouTube uploads —
which are *different masters/edits* than the audio the annotations were made on. We built a
per-track affine aligner (`align_harmonix.py`: `t_audio = a·t_anno + b` + a chance-corrected
confidence, classing tracks case1/case2/case3 and dropping case3), and the evaluator applies
it. But an affine map **cannot fix a discrete downbeat-phase shift** introduced when a YouTube
edit has a different intro length — so the precision-sensitive metrics stay corrupted. Full
runs make this unambiguous:

| Metric | **Raveform** (native) | Harmonix (YouTube-aligned) |
|--------|----------------------:|---------------------------:|
| Downbeats F | **~0.50** | 0.20 |
| Boundary HR@0.5s | **~0.46** | 0.18 |
| Beats F | 0.58 | 0.68 |

On Harmonix, **468/728 tracks score exactly 0 on downbeats** (302 of them with beats-F > 0.7) —
a bimodal phase artifact, not model behaviour. Conclusion: Harmonix-on-YouTube is **not a
usable target** for the metrics we care about. The scripts (`acquire_harmonix.py`,
`align_harmonix.py`) are kept for reference / cross-domain curiosity; pass
`--manifest eval/data/harmonix/manifest.jsonl` to run it, but don't optimize against it.

Raveform (native alignment) is the trustworthy benchmark and points at **EDM beat tracking**
(beats-F 0.58, *below* pop) as the lever for improvement — consistent with the Raveform paper's
finding that pop-trained models degrade on EDM.

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
