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

## Calibrate expectations first (important — measured 2026-06-27)
The held-out 8-fold CV understates production. Scored with the production `all-all` ensemble on a
90-track genre-balanced sample (vs each track's held-out single fold):

| metric (D&B, n=20) | held-out CV | **all-all ensemble** |
|---|---|---|
| beat-F | 0.941 | **0.980** |
| downbeat-F | 0.906 | **0.974** |

The catastrophic 0098 (held-out 0.228 → **0.964** ensemble) drove most of the gap. **D&B beat/
downbeat are near-solved in production** (~0.98), ~0.004 off the overall — so:
- **Do NOT spend tomorrow on D&B beat/tempo.** Tempo augmentation chases a nearly-closed gap.
- The genuine remaining headroom is the **section head (boundary HR: 0.755 overall, D&B ~0.60)**
  and the **function head (buildup acc 0.49, cooldown 0.48 — both lost to "drop")**. These need
  D&B oversampling + **function-class balancing** + boundary supervision, NOT tempo aug.
- Measure on an **external** D&B set too (Raveform can't honestly eval the ensemble — contamination).
- If beats matter for an external library, the ensemble already ships ~0.98; verify there first.

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
4. **Target the section + function heads, not beats** (production beats are already ~0.98 — see
   expectations). Two levers:
   - **Function-class balancing.** buildup/cooldown are systematically lost to "drop". Up-weight the
     function-head cross-entropy for the minority classes (buildup, cooldown, altintro, altoutro,
     bridge) — inverse-frequency class weights, or bump `loss_weight_function` (default 0.1 is very
     low) so the head is actually optimized. This directly attacks pairwise/V-measure.
   - **Boundary supervision.** `loss_weight_section` is already 15; the gap is harder D&B sections.
     D&B oversampling (step 3) plus mild tempo augmentation (±8–12% time-stretch in the loader,
     rescaling annotation times) mainly helps the section head generalize — keep tempo-aug *mild*
     and secondary, since its beat benefit is now marginal.

Keep v1 defaults otherwise: `lr=0.005` radam, SWA, `batch_size=1`, `segment_size=300`,
early-stopping patience 10; loss weights beat 1 / downbeat 3 / section 15 / function 0.1 (consider
raising the function weight given the label-confusion target).

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
**Gate (the target is segments, not beats):** improve held-out boundary HR@0.5 (> 0.755 overall,
> 0.60 D&B) and pairwise-F (> 0.825; especially recover buildup/cooldown accuracy from ~0.49) with
NO regression on beat (≥ 0.974 held-out) / downbeat (≥ 0.96). Slice by `genre == "Drum & Bass"`.
Use `--target none` (model-native). Note: held-out CV understates production, so also compare the
new ensemble to the current `all-all` numbers above before declaring a win.

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
