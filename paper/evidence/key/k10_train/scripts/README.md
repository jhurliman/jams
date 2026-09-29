# K10 run scripts (verbatim records)

These files are the scripts as they ran on 2026-07-12; they are kept byte-for-byte as
provenance and are **not** meant to be executed from a checkout: they reference the
author's job scratch directory and the training box's home directory.

- `k10_runner.sh`, `k10_retrain.sh` (one level up): the Lambda-instance invocations
  (`train_key_cnn.py acquire / features / train --workers 12`, everything else default).
- `k10_oof_infer.py`: produced the out-of-fold posteriors (`../../k10_oof_train.jsonl`)
  from the five fold checkpoints.
- `k10_candidates_cv.py`: the pre-registered candidate comparison, cross-validation only.
  It imports `key_fusion.py` (the fusion feature/scoring module of the earlier key work,
  also kept here verbatim) and the GS-MTG feature files.
  `paper/key_candidates_cv.py` re-implements it on the committed inputs in
  `../selection/` and reproduces its four numbers. Its fusion candidates (b) and (c)
  cross-validate their meta-models over one global out-of-fold table, not nested inside
  the CNN folds; see that script's docstring for why this is stated rather than redone.
- `k10_test_shot.py`: the single pre-registered GiantSteps Key evaluation of `final.pt`.

Inputs committed for reproduction (`../selection/`): `gsmtg_labels.jsonl` (GS-MTG track
ids, reference keys, and annotation confidence, reduced from the local manifest),
`skey_gsmtg.jsonl` (S-KEY posteriors), `keyfeat_gsmtg.jsonl` (template-matcher cues).
