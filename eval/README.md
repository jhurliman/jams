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
| **Raveform** | EDM / DJ | 1,423 | beats, downbeats, functional segments | honest 8-fold CV `all-fold{fold}` (EDM-trained; `manifest_foldcv.jsonl`) | YouTube id — **annotations made on the same video → native alignment** | ✅ |
| Harmonix | Western pop | 912 | beats, downbeats, segments | per-fold CV `harmonix-fold{i%8}` | YouTube — **different master → misaligned** | ❌ |
| EDM-98 | EDM | 98 | segments only | `all-all` | *not publicly released* | — |

```sh
uv run --extra eval eval/acquire_raveform.py    # MIT annotations + YouTube audio → manifest
uv run --extra eval eval/evaluate_structure.py  # scores Raveform by default
```

**Metrics** (`mir_eval`): beats/downbeats F (70 ms); segment boundaries Hit-Rate F @0.5 s /
@3 s; segment labeling pairwise-F + V-measure. Segments-only datasets (EDM-98) skip the beat
metrics automatically. `--target {jams,genre,ref,none}` sets the beat-tracking BPM prior:
`none` (model-native, the paper's protocol), `jams` (jams' TempoCNN), `genre` (TempoCNN folded
into the track's genre octave — D&B etc.), or `ref` (dataset BPM, the octave-correct ceiling).

### SOTA reproduced — Raveform held-out 8-fold CV

The EDM-trained ensemble loads locally via a state-dict remap (no training; see
`structure_worker.py`). Scored with each track's **held-out** `all-fold{fold}`:

| Metric | jams (104-track CV) | paper "v2" (EDM-trained) |
|--------|--------------------:|-------------------------:|
| Beats F | **0.978** | 0.991 |
| Downbeats F | **0.964** | 0.965 |
| Boundary HR@0.5 s | **0.755** | 0.835 |
| Pairwise F | **0.825** | 0.847 |
| V-measure | **0.877** | (Sf 0.890) |

Two fixes were needed to get here (each a large jump): the boundary peak threshold was hard-coded
to `> 0.0` (2–3× over-segmentation → HR 0.53; now a tunable default 0.2), and segments were scored
against the coarse beat-CSV `section` column instead of canonical `segments.json` (which preserves
same-label phrase boundaries — embed as `row["sections"]`).

**Held-out CV understates production.** Each CV track is scored by a single held-out fold; the
shipped model is the 8-fold `all-all` ensemble, which is more robust (e.g. D&B track 0098: 0.228
under its held-out fold → **0.964** under `all-all`). Raveform can't honestly eval `all-all`
(contamination), so production D&B is better than the CV row suggests — measure it on an external set.

### Harmonix status — disabled by default

Harmonix's annotations are public but **its audio is not**, so we source YouTube uploads —
which are *different masters/edits* than the audio the annotations were made on. We built a
per-track affine aligner (`align_harmonix.py`: `t_audio = a·t_anno + b` + a chance-corrected
confidence, classing tracks case1/case2/case3 and dropping case3), and the evaluator applies
it. But an affine map **cannot fix a discrete downbeat-phase shift** introduced when a YouTube
edit has a different intro length — so the precision-sensitive metrics stay corrupted. Full
runs make this unambiguous:

On Harmonix, **468/728 tracks score exactly 0 on downbeats** (302 of them with beats-F > 0.7) —
a bimodal phase artifact, not model behaviour. Conclusion: Harmonix-on-YouTube is **not a
usable target** for the metrics we care about. The scripts (`acquire_harmonix.py`,
`align_harmonix.py`) are kept for reference / cross-domain curiosity; pass
`--manifest eval/data/harmonix/manifest.jsonl` to run it, but don't optimize against it.

### Where the remaining error is (error analysis)

Raveform (native alignment) is the trustworthy benchmark. With the EDM model the headroom is no
longer beat tracking (0.978, near-ceiling) but:
- **Boundary HR** (0.755, weakest metric, 29/104 tracks < 0.7) — threshold 0.2 is the optimum;
  per-genre tuning adds only ~0.006. D&B boundaries stay ~0.60 (genuinely ambiguous sections).
- **Label confusion** (pairwise 0.825): the model over-predicts *drop* — buildup (acc 0.49) and
  cooldown (0.48) are lost to it. A positional relabel heuristic was tried and **hurt** (−0.024
  pairwise); the fix needs training (class weighting), not postprocessing.
- **Downbeat phase**: not a separate bug — bar offsets are already correct (0 tracks improvable),
  downbeat just tracks beat.
- **D&B beat**: mostly a held-out single-fold artifact (the `all-all` ensemble recovers it).

See `TRAINING.md` for the D&B-oversampling + tempo-augmentation training plan that targets the
genuine remainders.

## Stems → MIDI transcription (multi-dataset)

Scores the stems pipeline (`jams.analysis.stems.analyze_stems`) with `mir_eval`. Two modes
**decouple transcription from separation** so we can measure quality before separation is
polished:

- `--mode oracle` — transcribe the dataset's **ground-truth stems** (separation skipped).
  Isolates the transcribers (basic-pitch / OaF-drums). The headline number.
- `--mode e2e` — separate the mix with Demucs, then transcribe; also scores separation SI-SDR.

| Dataset | Domain | What it scores | Ground truth | Acquire |
|---------|--------|----------------|--------------|---------|
| **Slakh2100** | synth multitrack | note-F (bass/other) + drum onset-F + SI-SDR | stems **and** aligned per-stem MIDI (CC-BY-4.0, mirdata) | `acquire_slakh.py --data-home <babyslakh\|slakh2100_flac_redux>` |
| **E-GMD** | isolated drums | per-GM-instrument drum onset-F | audio↔MIDI (Roland TD-17, GM-native) | `acquire_egmd.py --data-home <extracted e-gmd>` |
| **MedleyDB** | real multitrack | melodic note-F (f0→notes) | gated audio + pitch annotations (mirdata) | `acquire_medleydb.py --data-home <medleydb_pitch>` |

Metrics: pitched stems use `transcription.precision_recall_f1_overlap` (onset+pitch F, offsets
ignored); drums use per-class `onset.f_measure` (50 ms) macro-averaged in the standard
**5-class ADT vocabulary** (kick/snare/hats/toms/cymbals — `--drum-classes gm10` scores the
full GM set instead; both sides canonicalised via `jams.analysis.gm`); e2e adds SI-SDR per
stem. `--fresh` discards a stale `--out` checkpoint after pipeline changes.

```sh
uv run --extra eval eval/acquire_slakh.py --data-home /data/babyslakh_16k --subset babyslakh
uv run --extra eval eval/evaluate_transcription.py --manifest eval/data/slakh/manifest.jsonl --mode oracle
```

**Dataset notes.** Slakh full is 100 GB+ — start with `babyslakh` (`--subset babyslakh`) or the
`2100-redux` set; the acquire script never triggers the giant download (point `--data-home` at
a local copy). E-GMD's ~100 GB audio is served only as one zip → download+extract it, then use
`--data-home` (individual-file HTTP fetch 404s). MedleyDB audio is gated → obtain it manually,
place under `--data-home`; the script drops-all-and-exits with instructions otherwise.

**Drum model.** `drum_worker.py` uses **ADTOF-pytorch** (torch port of the ADTOF Frame_RNN,
parity-validated against the original: F 88.5 vs 88.7 on MDBDrums++) — torch/librosa only, so
drum transcription runs on Apple Silicon, Linux, and CI identically. It emits the 5-class
vocabulary above with fixed velocity. (An earlier Magenta/OaF E-GMD integration was dropped:
its pinned `tensorflow==2.9.1` has no arm64 wheel.)

## Files

| File | Purpose |
|------|---------|
| `acquire_dataset.py` | Download GiantSteps Key → `data/manifest.jsonl` |
| `acquire_raveform.py` | Download Raveform (primary EDM structure set) → `data/raveform/manifest.jsonl` |
| `acquire_harmonix.py` | Download Harmonix annotations + YouTube audio → `data/harmonix/manifest.jsonl` |
| `align_harmonix.py` | Fit per-track YouTube↔annotation affine warp + confidence → `alignment.jsonl` |
| `evaluate_structure.py` | Multi-dataset structure scoring (`mir_eval`); `--target {none,jams,genre,ref}` |
| `acquire_slakh.py` | Slakh2100 (stems + per-stem MIDI) → `data/slakh/manifest.jsonl` |
| `acquire_egmd.py` | E-GMD drums (audio↔MIDI) → `data/egmd/manifest.jsonl` |
| `acquire_medleydb.py` | MedleyDB (real melodic, gated audio) → `data/medleydb/manifest.jsonl` |
| `evaluate_transcription.py` | Stems→MIDI scoring (`mir_eval`); `--mode {oracle,e2e}` |
| `prepare_raveform_training.py` | Raveform → Harmonix-shaped training set (for `TRAINING.md`) |
| `TRAINING.md` | Ready-to-run D&B fine-tune recipe (v1 trainer, oversampling + tempo aug) |
| `evaluate.py` | Score production `jams.detect_key` / `detect_tempo` |
| `benchmark_methods.py`, `benchmark_final.py` | Method comparisons |
| `analyze_errors.py` | Domain error taxonomy (mode, octave, per-genre) |
| `build_corrections.py` | Regenerate `tempo_corrections.csv` from GiantSteps-Tempo v2 |
| `train_mode_model.py` | Train/export the major-minor refinement (`src/jams/data/mode_model.json`) |
| `tempo_corrections.csv` | Curated tempo-label fixes (committed) |
