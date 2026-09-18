# MIREX 2026 Audio Key Detection — K10 candidate package

**Eligibility unresolved (8 September 2026).** The [2026 task rules](https://music-ir.org/mirex/wiki/2026:Audio_Key_Detection) prohibit GiantSteps Key use for any development purpose. Prior project test-error analysis informed K10. Do not represent this package as eligible without organizer clarification using the full history below. The October 1 deadline does not remove this issue. See [publication plan](../../paper/VENUE.md); a required task extended abstract is not yet included.

## Authors / contact

- John Hurliman, independent researcher — <jhurliman@jhurliman.org> (sole contact)

## Method description

A single small convolutional network (~0.1 M parameters, 24-way softmax over
{12 tonics} x {major, minor}) in the Korzeniowski & Widmer (ISMIR 2018) family:

1. Decode input audio, downmix to mono, resample to 22050 Hz.
2. Log-magnitude constant-Q transform: 24 bins/octave, hop 4096 (~5.4 fps),
   8 octaves from C1, computed with a 4-semitone margin on each side (208 bins)
   and cropped to the central 192 bins; `log1p` compression.
3. One forward pass over the full-track CQT: three conv blocks
   (Conv3x3-BN-ELU x2 + MaxPool2, channels 16/32/64), adaptive average pooling
   over TIME ONLY (frequency structure is preserved to the readout, since key
   identity lives in absolute frequency position), linear layer to 24 classes.
4. Argmax -> (tonic, mode). Deterministic; no test-time augmentation.

Training (done previously; this submission is inference-only): mirdata
`beatport_key` (Faraldo's revised v1.0 annotations, 1,486 tracks; 1,363 usable
after label/audio filtering) with ±4-semitone pitch-shift augmentation via CQT
bin-roll and label transposition. All model selection (width, budget,
candidates) by 5-fold cross-validation within the training corpus; the
submitted weights are the final model trained on the full corpus at the
CV-selected epoch budget.

Reference (in preparation): John Hurliman, *Evaluating a Modular Mix-to-MIDI Pipeline on Slakh2100*, 2026. Key results and qualifications are secondary appendix material; there is no arXiv identifier yet.

The historical K10 summary reports a **legacy symmetric-fifth weighted score** of
0.8321 [95% CI 0.8039, 0.8586] and exact accuracy 0.7795 on 567 GiantSteps Key
examples. The legacy scorer gives fifth credit in both directions; mir_eval 0.8.2
uses +7 only. Confirm the organizers' implementation before claiming protocol
compatibility. These intervals are exploratory and the raw archives still require
release; they do not demonstrate equivalence to madmom or predict a future result.

## Training and development disclosure

The ledger records final K10 training on mirdata `beatport_key` (1,363 usable
examples), with architecture/epoch selection by cross-validation within that corpus.
It reports all 1,157 usable GiantSteps-MTG IDs included and zero track-ID overlap
with GiantSteps Key. `uv run eval/verify_key_disjoint.py` checks dataset index IDs;
it does not check audio duplicates or development history.

**GiantSteps Key was previously used in this project's development.** Earlier
systems were scored and their errors analyzed; an older mode classifier was fit
on that test set. K10's motivation explicitly uses the banked GiantSteps error and
oracle analysis to justify a stronger base model. The ledger says the frozen K10
checkpoint received one final test evaluation, but the project as a whole was not
blind to that benchmark. This broader history is material under the 2026 rule
against any development use. Seek an organizer ruling with this disclosure before
competitive submission; do not claim that disjoint final training IDs resolve it.

## Calling format

MIREX per-file contract — one audio file in, one text file out:

    predict_key.py %input %output

Example:

    ./predict_key.py /path/to/track.wav /path/to/track.key.txt

- `%input`: path to the audio file (MIREX format: 44.1 kHz, 16-bit, mono WAV;
  any libsndfile/audioread-decodable file works — audio is resampled to
  22050 Hz mono internally, matching the training featurization).
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
- `key_cnn_v1.pt` — bundled model weights, 450,107 bytes
  (md5 `6141daf25376c16a7bc4326b742e4a3c`). This is the frozen K10 `final.pt`;
  no download step is required.
- `README.md` — this file.

## Historically recorded runtime environment

The measurements below come from the original package documentation and were not rerun during the publication review. Dependency ranges are not a lockfile; capture an exact environment and verify the target platform before submission.

- Language: Python. Tested with CPython 3.12 (declared range: >=3.10, <3.13).
- Dependencies (exact versions used in verification): `torch` 2.8.0 (CPU),
  `librosa` 0.11.0, `numpy` 2.2.6. Declared constraints: `torch==2.8.*`,
  `librosa>=0.10`, `numpy>=1.26,<2.3`.
- Recommended invocation: install [uv](https://docs.astral.sh/uv/) and run
  `./predict_key.py in.wav out.txt` (or `uv run predict_key.py in.wav
  out.txt`) — uv resolves and caches the environment from the script's inline
  metadata on first run. Alternatively `pip install "torch==2.8.*"
  "librosa>=0.10" "numpy>=1.26,<2.3"` and run `python predict_key.py ...`.
- OS: any platform with CPU torch wheels (historically verified on macOS arm64; Linux
  x86-64 needs a packaging smoke test).
- No network access is needed at prediction time (after the one-time
  environment install).

## Resource requirements

- **Threads/cores:** single-threaded (`torch.set_num_threads(1)` is set
  explicitly; one process, one core). Safe to run many instances in parallel.
- **Memory:** ~390 MB peak RSS per invocation (measured on a 32 s input;
  scales mildly with track length via the full-track CQT — longer-input memory was not verified in this review).
- **Runtime:** ~1.7 s wall per 2-minute track on one Apple M-series CPU core,
  including interpreter start and model load (x86-64 timing not measured). GPU is not used. The very first invocation on a fresh
  machine additionally pays a one-time dependency install (~40 s with a warm
  network; please run one warm-up invocation before batch timing).
- **Scratch disk:** none used by the program itself (output file only). The
  one-time uv/pip environment is ~1.5 GB (dominated by the torch wheel).

## Special notices

- The model predicts a single global key per track (24 classes); no key
  changes are reported, per the task definition.
- Decoding of the input file uses libsndfile via librosa/soundfile; standard
  PCM WAV requires no extra system packages.
