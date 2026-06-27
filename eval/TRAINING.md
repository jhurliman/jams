# Training a Raveform model to push structure analysis past SOTA on D&B

Ready-to-run recipe for a GPU box. Goal: train a **v1-architecture, 11-class Raveform** model
that loads as a **drop-in** in `structure_worker.py` (no remap), with **D&B oversampling +
tempo augmentation** — the two things the paper explicitly did *not* use ("Data augmentation and
additional datasets are not utilized in this study"), which is where the headroom is.

## Why v1, not v2
The published `all-fold*` (v2) weights have a multi-dataset architecture (`dataset_classifier` +
per-dataset function heads) whose **training code is unpublished**. But the published **v1 trainer**
(`mir-aidj/all-in-one`, `src/allin1/training/`) produces exactly the single-dataset, flat-head
checkpoint our port already loads (`structure_worker._remap_v2_to_v1` treats `raveform-fold3` as a
no-op). So a v1-trained Raveform checkpoint is a drop-in. **Don't** try to fine-tune `all-fold*` —
that needs the unpublished v2 trainer.

## Calibrate expectations first (important)
The held-out 8-fold CV understates production. The catastrophic D&B case (track 0098, beat-F
**0.228** under its held-out single fold) is **0.964** under the production `all-all` ensemble — the
0.228 was a single-fold weak-intro artifact, not a model failure. Across the D&B set the ensemble
already tracks the correct global tempo on 63/64 tracks. So:
- **Beat/downbeat headroom on D&B is small** (the ensemble largely solves it).
- The real wins to chase: **boundary HR** (weakest metric, 0.755; D&B 0.60 — genuinely hard) and
  the **buildup/cooldown→drop label confusion** (buildup acc 0.49, cooldown 0.48 — both lost to
  "drop"). Tempo augmentation + class weighting target these.
- Measure on an **external** D&B set too (Raveform can't honestly eval the ensemble — contamination).

## Steps

### 0. Environment (~15 min, needs CUDA)
```bash
git clone https://github.com/mir-aidj/all-in-one.git && cd all-in-one
git rev-parse HEAD                       # pin the commit
pip install -e '.[train]'                # torch, lightning, hydra, timm, demucs, natten, madmom, wandb
wandb login
```

### 1. Data prep (~30–60 min, CPU)
From this repo:
```bash
unzip -p eval/data/raveform/raveform.zip raveform/structures/segments.json > eval/data/raveform/segments.json
uv run eval/prepare_raveform_training.py --out data/raveform_train
sh data/raveform_train/transcode.sh      # m4a -> tracks/*.mp3 (the loader globs *.mp3)
```
Produces `metadata.csv` (File, BPM, **true fold**, genre), `beats/*.txt` (`time<TAB>downbeat`),
`segments/*.txt` (`start<TAB>end<TAB>label`, 11-class), `labels.txt`. 1372 tracks, ~21 D&B/fold.

### 2. Patch the trainer (~1–2 h, the only real code work)
1. **Register a `raveform` DataModule** — `train.py` raises on non-harmonix. Copy the Harmonix
   DataModule/dataset and point it at `data/raveform_train`; read beats/segments from the txt files
   above (Harmonix uses the same shape). Set `num_labels=11`, label order = `labels.txt`.
2. **True folds** — replace `folds = np.arange(len(ids)) % total_folds` with the `fold` column from
   `metadata.csv` (our folds match the paper's; index%8 would not).
3. **D&B oversampling** — swap the train DataLoader's `shuffle=True` for a `WeightedRandomSampler`
   with per-track weight ∝ inverse genre frequency, **capped ≤4×** for D&B (Techno dominates). Mild,
   to protect generalization.
4. **Tempo augmentation** — in the 5-min-chunk loader, time-stretch ±8–12% on the fly (resample the
   mel-frame axis, rescale beat/downbeat/section *times* by the same factor). This directly attacks
   local half-time drift. Optionally inject occasional half/double-time hard negatives on D&B chunks.
   (Pitch-shift is low value here; tempo is the lever.)

Keep v1 defaults otherwise: `lr=0.005` radam, SWA, `batch_size=1`, `segment_size=300`,
early-stopping patience 10; loss weights beat 1 / downbeat 3 / section 15 / function 0.1.

### 3. Preprocess once (GPU, ~1–3 h, demucs-bound)
```bash
allin1-preprocess data.name=raveform data.path_track_dir=data/raveform_train/tracks \
  data.path_feature_dir=data/raveform_train/features
```

### 4. Train per-fold CV (GPU)
```bash
allin1-train data.name=raveform fold=0 data.num_labels=11 \
  data.path_feature_dir=data/raveform_train/features      # fold-0 smoke first
wandb sweep sweep.yaml && CUDA_VISIBLE_DEVICES=0 wandb agent <SWEEP_ID>   # all 8 folds
```
~2–5 h/fold on a 24 GB GPU (300K-param model, IO-bound). 8 folds → parallelize across GPUs.

### 5. Evaluate against our numbers
Score each fold's tracks with its **held-out** `raveform-fold{i}` (the harness does this for Harmonix
with `harmonix-fold{i%8}`; replicate using the true fold). Then:
```bash
uv run --extra eval eval/evaluate_structure.py \
  --manifest eval/data/raveform/manifest_foldcv.jsonl --target jams --out runs/raveform_trained.jsonl
```
**Gate (no-regression):** D&B beat-F > 0.962 (held-out) AND overall beat ≥ 0.978, downbeat ≥ 0.964,
boundary HR@0.5 ≥ 0.755. Slice metrics by `genre == "Drum & Bass"`. Use `--target jams` so the gain
is the *model*, not the post-hoc octave-correct.

### 6. Deploy a winner
Drop the 8 `raveform-fold{0..7}.pth` into `structure_worker.py` `_EXTRA_FILES` + an
`_EXTRA_ENSEMBLES` entry (already v1 form → no remap), and point the default model at the new
ensemble.

## Compute summary
Preprocess ~1–3 h (once). Train ~2–5 h/fold; 8 folds = one fold wall-clock if 8 GPUs, else stage
them. **Pragmatic day plan:** preprocess in the morning → fold-0 smoke (~3–5 h) → check D&B lift +
no-regression on the harness → launch the other 7 folds overnight.

## Blockers / risks
- No published `raveform` DataModule → reuse the Harmonix path (step 2.1). Main code task.
- Oversampling/augmentation aren't in the published trainer → you add them (step 2.3–2.4); this *is*
  the headroom.
- Don't over-oversample D&B (cap ≤4×); gate on the no-regression check before shipping.
- Augmentation that forces metronomic grids can hurt expressive material — keep stretch mild and
  validate non-D&B metrics.
