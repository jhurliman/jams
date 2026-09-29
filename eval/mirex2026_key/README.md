# MIREX 2026 Audio Key Detection — K10 submission package

**Eligibility: confirmed by the task captains (28 September 2026).** The organizers
replied that a submission is not disqualified solely because GiantSteps Key was used for
evaluation during development, provided the usage is described accurately in the
extended abstract; K10 participates in the ranked results with that disclosure. They
will report both the symmetric and the asymmetric (ascending-only) fifth-credit
conventions; previous MIREX evaluations used the symmetric one. **Deadline extended to
7 October 2026 (AoE).** Upload at <http://futuremirex.com/submission>. The required 2–4
page extended abstract is `abstract/main.pdf` (ISMIR 2026 LBD template, source alongside).

## Authors / contact

- John Hurliman, independent researcher — <jhurliman@jhurliman.org> (sole contact)

## Method description

A single small convolutional network (109,128 parameters, 24 output logits over
{12 tonics} x {major, minor}) in the Korzeniowski & Widmer (ISMIR 2018) family:

1. Decode input audio, downmix to mono, resample to 22050 Hz.
2. Log-magnitude constant-Q transform: 24 bins/octave, hop 4096 (~5.4 fps),
   8 octaves from C1, computed with a 4-semitone margin on each side (208 bins)
   and cropped to the central 192 bins; `log1p` compression.
3. One forward pass over the full-track CQT: three conv blocks
   (Conv3x3-BN-ELU x2 + MaxPool2, channels 16/32/64), Dropout2d 0.2, average
   pooling over TIME ONLY (the 24 remaining frequency positions are kept for the
   dense readout), linear layer to 24 logits.
4. Argmax -> (tonic, mode). Deterministic; no test-time augmentation.

Training (done previously; this submission is inference-only): mirdata
`beatport_key` (Faraldo's revised v1.0 annotations, 1,486 tracks; 1,363 retained
by the label rule below) with ±4-semitone shift augmentation (a shifted 192-bin
window inside the 208-bin padded CQT, label transposed to match). Cross-entropy,
AdamW (lr 1e-3, weight decay 1e-4, cosine schedule), batch 32, 320-frame random
crops, 5 folds (seed-0 shuffle of the sorted ids, round-robin), symmetric-fifth
weighted score on the held-out fold as the selection metric, at most 60 epochs with
patience 10. Fold results 0.734/0.699/0.696/0.731/0.723 (mean 0.716; best epochs
53/23/15/17/36, fold 0 censored at the 60-epoch cap; the per-epoch histories confirm the
default cap and patience); final model trained on all 1,363 tracks for 23 epochs (median
best epoch). A width-2.0 variant with patience 15 (fold mean 0.710) was rejected. Invocation: `train_key_cnn.py train --workers 12`,
all other arguments at their defaults, one A10 GPU, about 2 h. Records:
`paper/evidence/key/k10_train/` (fold summaries, per-epoch histories, training log,
run scripts, selection scripts), `paper/evidence/key/k10_oof_train.jsonl`
(out-of-fold posteriors), `paper/evidence/key/k10_train_labels.jsonl` and
`k10_train_drops.json` (reconstructed by `paper/key_train_manifest.py`).

Training label rule: a beatport_key track is retained when its first listed
annotation is exactly `<tonic> major|minor` (flats mapped to sharps). 1,275 retained
tracks have a single-key annotation; 88 have a two-key annotation (`E minor | E major`)
and use the first listed key. The 123 drops are 115 single-key annotations (75 atonal
`X`, 12 `other`, 28 modally qualified) and 8 two-key annotations whose first entry failed
the major/minor parser. No audio was dropped; the confidence field is not used.

Reference: John Hurliman, *Evaluating a Modular Mix-to-MIDI Pipeline on Slakh2100*, 2026
(repository `paper/arxiv/`; key results are appendix material; arXiv deposit pending).

Development-time result (one evaluation of the locked model, 567 usable GiantSteps
Key excerpts): weighted **0.8321** [0.8042, 0.8584] with symmetric fifth credit (interval
from `paper/key_scores.py` on the committed predictions; the ledger's original endpoints
differ in the fourth decimal from bootstrap row order),
**0.8145** with mir_eval 0.8.2's ascending-only fifth credit, exact **0.7795**. madmom's
default processor on the same tracks: 0.8328 / 0.8134 / 0.7725; the paired difference
crosses zero under both conventions and for exact accuracy. These are subset numbers on
a benchmark the project had used before and do not predict the MIREX full-set score.

Evaluation reference labels: `eval/acquire_dataset.py` reduces each GiantSteps Key
annotation to one major/minor key (modal qualifiers collapsed by mode family; two-key
annotations collapsed to the first tonic and to minor if any listed key is minor-family).
Of the 567 references, 378 are plain major/minor (incl. ionian/aeolian), 98 carry another
modal qualifier, 91 are collapsed two-key annotations (15 match neither listed key); 33
excerpts with `other`/`X`/malformed labels were excluded. Excluding only the 15
neither-key references gives K10 0.8493 and madmom 0.8484; on the 476 single-key
references 0.8668 and 0.8687; no subset establishes an advantage for either system
(`paper/key_scores.py`).

Package check: running this package on the same 567 audio files reproduces all 567
committed K10 predictions, and the training pipeline's float16 feature cache changes no
argmax (max softmax difference 7.5e-5): `paper/key_rerun_check.py`,
`paper/evidence/key/k10_rerun_check.json`.

## Training and development disclosure

**GiantSteps Key was previously used in this project's development.** Earlier
systems (template matcher, S-KEY, madmom, two logistic refinements fit on GS-MTG) were
scored on the 567 usable excerpts and their errors analyzed; one older mode classifier
had been fit on GiantSteps Key itself and was retired as contaminated before any
comparison was reported. The analysis of those banked predictions motivated building a
stronger base model (K10), but did not inform K10's architecture, hyperparameters, or
weights, which were selected by cross-validation within `beatport_key` under a
pre-registered decision rule (`paper/EXPERIMENTS.md`, entry K10). The locked K10
checkpoint was evaluated once on GiantSteps Key; descriptive breakdowns followed.
`uv run eval/verify_key_disjoint.py` checks that the two mirdata indexes share no
Beatport track id; it does not check audio duplicates. This history was disclosed to the
task captains, who confirmed eligibility on 2026-09-28 (see the abstract, Section 2).

## Calling format

MIREX per-file contract — one audio file in, one text file out:

    predict_key.py %input %output

Example:

    ./predict_key.py /path/to/track.wav /path/to/track.key.txt

- `%input`: path to the audio file (MIREX format: 44.1 kHz, 16-bit, mono WAV;
  any libsndfile/audioread-decodable file works — audio is resampled to
  22050 Hz mono internally, the same CQT configuration as training).
- `%output`: path to write the result.

Output is a single line of tab-delimited ASCII, tonic TAB mode, terminated by
one `\n` (LF only, no CR):

    C	major

Tonic vocabulary: `C C# D D# E F F# G G# A A# B` (sharps only). Mode: `major`
or `minor`. Exit code 0 on success, nonzero with a message on stderr on
failure (no output file is written on failure).

## Files

- `predict_key.py` — self-contained CLI wrapper (PEP 723 inline metadata; run
  via `uv run` or any environment with the dependencies below).
- `predict_key.py.lock` — uv lockfile for the script: the exact environment the
  package was tested in (`uv run --locked predict_key.py ...` refuses to run with
  anything else).
- `key_cnn_v1.pt` — bundled model weights, 450,107 bytes
  (md5 `6141daf25376c16a7bc4326b742e4a3c`, sha256
  `18c24e61cf779f399014c2deaaa156d1010c56dac8e463809830bfd354a0b4a4`). This is the
  frozen K10 `final.pt`; no download step is required.
- `README.md` — this file.
- `abstract/main.pdf` — the extended abstract (LaTeX source alongside).

## Runtime environment

- Language: Python; declared range >=3.10, <3.13. Tested with CPython 3.10.20 (macOS
  arm64) and 3.11.14 (Linux x86-64).
- Dependencies: declared `torch==2.8.*`, `librosa>=0.10,<1.0`, `numpy>=1.26,<2.3`;
  locked to torch 2.8.0, librosa 0.11.0, numpy 2.2.6, soundfile 0.14.0.
- Recommended invocation: install [uv](https://docs.astral.sh/uv/) and run
  `uv run --locked predict_key.py in.wav out.txt` from this directory — uv builds
  and caches the locked environment on first run. Alternatively
  `pip install "torch==2.8.*" "librosa>=0.10,<1.0" "numpy>=1.26,<2.3"` and
  `python predict_key.py ...`.
- On Linux the default index resolves the CUDA build of torch (about 1.5 GB of
  wheels); the script never uses the GPU. A CPU-only torch wheel works equally.
- No network access is needed at prediction time (after the one-time
  environment install).

## Measured resource use (2026-09-28)

End-to-end `uv run predict_key.py` on one 120 s, 44.1 kHz, 16-bit mono WAV, including
interpreter start and model load; `torch.set_num_threads(1)`.

| host | wall (median) | peak RSS | first-run install |
|---|---|---|---|
| Apple M2 Max, macOS 26.5, 5 runs | 1.6 s | 525 MB | (cached) |
| Intel i9-14900K, Ubuntu on WSL2, 3 runs | 1.7 s | 900 MB | 46 s |

- **Threads/cores:** single-threaded torch (one process, one core). Safe to run
  many instances in parallel.
- **Memory:** scales with track length through the full-track CQT; the Linux
  figure includes the CUDA torch build's overhead.
- **Scratch disk:** none used by the program itself (output file only). The
  one-time uv environment is about 1.5 GB on Linux (CUDA torch wheel) and about
  0.5 GB on macOS.

## Special notices

- The model predicts a single global key per track (24 classes); no key
  changes are reported, per the task definition.
- Decoding of the input file uses libsndfile via librosa/soundfile; standard
  PCM WAV requires no extra system packages.
