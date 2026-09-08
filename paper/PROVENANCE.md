# Provenance record — principal Slakh2100-redux transcription results

Written 2026-09-08 for step 3 of [REVIEW.md](REVIEW.md) ("Pin provenance"). Scope: the four
principal rows of [results_snapshot.json](results_snapshot.json) — T1 (basic-pitch, reference
"other"), T2b (YourMT3+, reference "other"), S4 (SCNet + basic-pitch), T10 (SCNet + YourMT3+).

Every statement below carries one of these statuses:

- **verified** — computed or read directly from a file, hash, URL or API response inspected
  while writing this record (the source is named);
- **ledger-quoted** — copied from `paper/EXPERIMENTS.md`, `paper/STATS.md`, `eval/README.md`,
  a git commit message, or a log stored in S3, with the section/line cited; not independently
  reproduced;
- **inferred** — a conclusion that follows from verified facts (the reasoning is stated);
- **UNKNOWN / NOT RECOVERABLE** — no file, hash or record was found.

Local inspection was done in the worktree at commit `b22d085868643c6f0b4606eeeabaf38e59b7c1ce`
(branch `paper-rigorous-revision`); the main checkout `/Users/jhurliman/Documents/Code/jhurliman/jams`
and `~/.cache` were read but not modified. The author workstation "aleph0" could **not** be
inspected in this session (see §7). Nothing was re-run; no model inference was performed.

---

## 1. Result archives (the four principal files)

The four archive files were recovered from the author machine's ignored
`eval/data/results_aws/` (REVIEW.md's "absent from the checkout and its history" referred to
the repository history) and are **committed under `paper/evidence/results_aws/`**, with the
SI-SDR aggregate, the compressed raw YourMT3+ predictions, and `SHA256SUMS`. Their SHA-256
values equal those recorded in `paper/evidence/publication_verified.json`
(`status: "stored_scores_recomputed"`) and in `paper/results_snapshot.json`.

| id | file (`paper/evidence/results_aws/`) | bytes | mtime (original copy, local) | SHA-256 | status |
|---|---|---:|---|---|---|
| T1 | `slakh_test_oracle.json` | 94,745 | 2026-07-02 04:31 | `03c9566fd53a78cdb16099e49f7ecb3077d5ca0205c444ddf8344592a700edb8` | verified (shasum; = publication_verified.json) |
| T2b | `yourmt3_oracle_per_track.json` | 50,575 | 2026-07-05 18:31 | `da4e13d00c301eb8a15fc9eaacdcd6cba5b790ef533e377e62c28d7658eb0061` | verified |
| S4 | `sep_scnet_notes.json` | 94,598 | 2026-07-02 16:42 | `19397c11e7d3d6023e624db9056fce47bb8c498d1c261ba24c19aebaf7765e1b` | verified |
| T10 | `slakh_test_e2e_scnet_yourmt3.json` | 108,954 | 2026-07-05 22:46 | `7c12670cf69f75e389609ce2caf77826080984741aad284d2cd7a5c8745112cd` | verified |

Companion files (`paper/evidence/results_aws/`; gate files under `paper/evidence/structure/`), hashed here:

| file | bytes | SHA-256 | content (verified by reading) |
|---|---:|---|---|
| `sep_scnet_sdr.json` | 147 | `34290c03c917fd688e97ca599a6061a6d1604bb29a51b730ab8b991a275bba9e` | `{"model":"scnet","n":151,"missing":0,"si_sdr":{"bass":5.98,"drums":14.31,"other":11.765,"vocals":null}}` |
| `yourmt3_scores.json` | 110 | `4e8288d22bfd05b6a0cb6866826c4a7502c14b6cb4e1b5034d67bda47a279037` | `{"yourmt3_oracle":{"bass":{"n":143,"note_f":0.1191},"other":{"n":151,"note_f":0.8488}},"errors":0}` |
| `yourmt3_final.txt` | 339 | `1a84811fc267ab3259faf0278c69f1682df4b0cc0216820f5afc2171d0e64dc3` | same numbers as above plus a uv warning naming `VIRTUAL_ENV=/home/ubuntu/mt3_venv` |
| `yourmt3_notes.jsonl` | 37,544,453 | `7a0c95f25142e63b7de470ac3a39c98a0a8baf9d66360c45d33f0f74060ce9c1` | 294 rows `{track_id, stem, notes:[{onset,offset,pitch,velocity}]}` — the raw YourMT3+ note predictions behind T2/T2b (151 "other" + 143 "bass"; velocity 100 throughout the rows inspected; pitches at sounding pitch, i.e. before the +12 bass shift) |

Archive headers (verified by reading the files):

- T1: `"mode": "oracle", "n": 151, "failed": 0`, aggregate `note_f` bass 0.7889 / other 0.4897 / overall 0.6352, `drum_onset_f` 0.6383. Per-track rows carry `stems.{drums,bass,other}` with `note_f/note_p/note_r` and per-class drum F.
- S4: `"mode": "oracle", "n": 151, "failed": 0`, bass 0.6448 / other 0.4733 / overall 0.5567, `drum_onset_f` 0.5741. **The S4 archive is an oracle-mode run** (the SCNet-separated stems were supplied to the evaluator as if they were reference stems); the SI-SDR half of S4 is the separate `sep_scnet_sdr.json`.
- T10: `"mode": "e2e", "n": 151, "failed": 0`, bass 0.6613 / other 0.7877 / overall 0.7262, `drum_onset_f` 0.5741, `sdr` drums 14.3098 / bass 5.9794 / other 11.7645.
- T2b: `"method": "yourmt3-oracle-rescore", "n_rows": 294, "missing": 0`, `aggregate_note_f` bass 0.8486 / other 0.8488; flat rows `{track_id, stem, note_f, note_p, note_r}`.

Observation (verified from the two files): the T2 archive `yourmt3_scores.json` records **bass note-F 0.1191**, whereas the ledger row T2 and the T2b rescore record 0.8486. The ledger row T2 says "+12 bass convention"; `src/jams/analysis/gm.py` (HEAD) comments "YourMT3+ 0.12 -> 0.85 note-F" for the +12 shift. The archived T2 bass figure is therefore the unshifted score, and the 0.8486 figure first appears in the T2b rescore. The "other" figure (0.8488) is identical in both.

Copies of T2b, T10 and `yourmt3_notes.jsonl` also exist in S3 (verified with `aws s3 ls`):
`s3://jams-mir-eval-usw2/lambda/e2eval/` holds `yourmt3_notes.jsonl` (37,544,453 B, 2026-07-05 17:32),
`yourmt3_oracle_per_track.json` (50,575 B, 18:46), `slakh_test_e2e_scnet_yourmt3.json` (108,954 B, 22:46),
plus `e2e.log` (107,201 B), `e2eval.log` (344 B), `rescore.log` (75 B). Sizes match the local files
byte-for-byte; S3 object hashes were not computed (nothing was downloaded except the three small logs).
No S3 copy of the T1 or S4 archives was found in the prefixes listed (`checkpoints/ d1/ gates/ k10/ lambda/ ov1/ ov2/ raveform_train/ s7/ tp1/`).

Per-track predictions for T1, S4 and T10 (MIDI or note lists): **NOT RECOVERABLE** — the
archives store scores only; no prediction files were found locally or in the S3 prefixes listed.
Per-track predictions for T2/T2b: recoverable from `yourmt3_notes.jsonl` (above).

---

## 2. Model checkpoints

### 2a. SCNet XL IHF (S4, T10 separator)

| item | value | source | status |
|---|---|---|---|
| Download base | `https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download` | `src/jams/data/stems_worker.py:180` (`_SCNET_BASE`), identical at S4 commit `8f14ff4` (line 153) | verified |
| Release tag | `v1.0.15` — GitHub release named "SCNet XL IHF Weights", `published_at` 2025-06-18T09:25:22Z | worker `_SCNET_FILES`; GitHub API `releases/tags/v1.0.15` | verified |
| Config asset | `config_musdb18_scnet_xl_more_wide_v5.yaml`, 5,906 B on GitHub | GitHub API; HEAD request Content-Length 5906 | verified |
| Weights asset | `model_scnet_ep_36_sdr_10.0891.ckpt`, 214,063,778 B on GitHub | GitHub API; HEAD request Content-Length 214063778 | verified |
| Local cache | `~/.cache/jams/scnet/` (path from `_scnet_cache_dir()`: `$XDG_CACHE_HOME` or `~/.cache`, then `jams/scnet`), files dated 2026-07-02 16:51 | `ls -la` | verified |
| SHA-256 weights (local cache) | `ac25975f0f5704f3d1a3c3c251505b7a0f417a22eafe82773440ee4f7e14b74f` (214,063,778 B — size equals the GitHub asset) | `shasum -a 256` | verified locally; GitHub publishes no digest for the asset, so identity with the file the AWS/Lambda boxes downloaded rests on the URL + size only |
| SHA-256 config (local cache) | `54ddcb8aeeae85c8d0e148741e7d87b0fbd38bd21cffa5d59ba124fe1e672f40` (5,906 B) | `shasum -a 256` | verified locally |
| Integrity check in code | size-only (`min_bytes` 1,000 / 200,000,000); no digest check | `stems_worker.py:208-225` | verified |
| Vendored model code | `src/jams/data/scnet/` "Vendored from ZFTurbo/Music-Source-Separation-Training … commit `ccc011abf7f89dd7922bb2888d48493b575c0289` (2026-06-09), models/scnet/ … Unmodified except this header" | `src/jams/data/scnet/__init__.py`; `git diff 8f14ff4 HEAD -- src/jams/data/scnet/` is empty | verified (no drift since S4) |
| Checkpoint loading | `torch.load(..., map_location="cpu", weights_only=False)`; unwraps `state` / `state_dict` keys | `stems_worker.py:256-261` | verified |
| Inference config | `cfg["inference"]["batch_size"]` forced to 1 off-CUDA; otherwise the YAML's values (not transcribed here) | `stems_worker.py:293-296` | verified |

The S3 log for T10 (`lambda/e2eval/e2e.log`) prints `[stems] SCNet ready on cuda` from
`/home/ubuntu/jams/src/jams/data/scnet/scnet.py` — the vendored code ran on CUDA on that box
(ledger-quoted from the log). The S4 box's log is not in S3 (UNKNOWN).

### 2b. YourMT3+ (T2/T2b, T10 pitched transcriber)

| item | value | source | status |
|---|---|---|---|
| Toolkit | `mt3-infer` (PyPI), worker pin `mt3-infer>=0.1.3` at T2 commit `4d466e2` and at HEAD | `src/jams/data/yourmt3_worker.py` header | verified |
| Model key | `yourmt3` → mt3-infer key `yptf_moe_nops`, display name **"YPTF.MoE+Multi (noPS)"** | `mt3_infer/config/checkpoints.yaml` and `adapters/yourmt3.py` in the 0.1.3 wheel (`~/.cache/uv/archive-v0/4kn2pRToSIhTshsQqBFHd`); T10 and S7 logs print `Loading YourMT3 model: YPTF.MoE+Multi (noPS)` | verified. Note the ledger/paper name it "YPTF.MoE+Multi" without the "(noPS)" qualifier |
| Source repository | Hugging Face **Space** `mimbres/YourMT3` (git-LFS clone, branch `main`, depth 1) — a Space, not a model repo; `cardData.license: apache-2.0` | `checkpoints.yaml` (`source_type: git_lfs`, `source_url: https://huggingface.co/spaces/mimbres/YourMT3`); HF API `/api/spaces/mimbres/YourMT3` | verified |
| File in repo | `amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt` | `checkpoints.yaml` `target_path` | verified |
| Local path used by the worker | `<repo>/.mt3_checkpoints/yourmt3/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/last.ckpt` (gitignored) | `checkpoints.yaml` `path`; T10 log `Checkpoint: /home/ubuntu/jams/.mt3_checkpoints/yourmt3/.../last.ckpt`; commit `4d466e2` message | verified |
| Upstream LFS SHA-256 | `ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25`, size 561,544,628 B; git blob oid `3b0997b1d3b7078cf4fe7fb2b9b08d405a196665` | HF API `/api/spaces/mimbres/YourMT3/tree/main/amt/logs/2024/.../checkpoints` (`lfs.oid`) | verified |
| Space revision | HEAD `5e66c1ea173a8186e0d20432b841d3180cc015b5`, `lastModified` 2025-01-31T14:38:17Z | HF API | verified. Because the Space was last modified before any run in the ledger (July 2026), every depth-1 clone of `main` during the experiments received this same commit and file (inferred) |
| Local clone, main checkout | `/Users/jhurliman/Documents/Code/jhurliman/jams/.mt3_checkpoints/.../last.ckpt`, 561,544,628 B, mtime 2026-07-14 13:35, SHA-256 `ae38e415…fc25` (= upstream LFS oid) | `shasum -a 256` | verified |
| Local clone, this worktree | same path, 561,544,628 B, mtime 2026-07-02 18:42; hash not computed in this session (command denied) | `ls -la` | size verified only |
| OV1 box record | `ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25  /home/ubuntu/ov1/scripts/.mt3_checkpoints/yourmt3/.../last.ckpt` (2026-07-14) and the archived file `s3://jams-mir-eval-usw2/ov1/scripts/.mt3_checkpoints/.../last.ckpt` (561,544,628 B) | `s3://jams-mir-eval-usw2/ov1/logs/yourmt3_ckpt_hashes.txt` | ledger-quoted (S3 log); equals the upstream digest |
| Checkpoint on the T2 / T10 boxes | not hashed on those boxes; paths only (`/home/ubuntu/jams/.mt3_checkpoints/...` in the T10 log) | `lambda/e2eval/e2e.log` | UNKNOWN hash; inferred identical (same clone mechanism, unchanged upstream) |
| Git commit of the clone | depth-1 clone; upstream HEAD `5e66c1ea…` (above). A `.git` directory for the clone was not looked for locally | — | upstream verified; local clone metadata UNKNOWN |
| mt3-infer version actually installed on the boxes | not logged. PyPI releases: 0.1.3 uploaded 2026-01-14 (wheel SHA-256 `974e1d5a4809c63079fa7e2c699add3671bb2970ba1ff3ea138923e2a8922b78`), **0.2.0 uploaded 2026-07-11** (`95209657fafab7eda0e5187f5fb9ae3438a18e2a4e30acd4786f3cda25389ff4`). T2 (Jul 2/3) and T10 (Jul 5) predate 0.2.0, so `>=0.1.3` could only resolve to 0.1.3 from PyPI at that time | PyPI JSON API | inferred |

### 2c. basic-pitch (T1, S4 transcriber)

| item | value | source | status |
|---|---|---|---|
| Version pin | `basic-pitch[onnx]>=0.4` in the stems worker manifest, identical at `cf28158` (T1), `8f14ff4` (S4) and HEAD; not in `uv.lock` (worker env is separate) | `src/jams/data/stems_worker.py` header; `git show` at those commits | verified |
| Resolvable version | PyPI lists releases up to **0.4.0** (no newer release as of 2026-09-08), so `>=0.4` resolves to 0.4.0 | PyPI JSON API | inferred |
| Wheel | `basic_pitch-0.4.0-py2.py3-none-any.whl`, SHA-256 `738adb503aae7fdfc7d1e1511aa0ce35052315f260a19531ef4c356708425db0`, uploaded 2024-08-16 | PyPI | verified |
| Installed locally | 0.4.0 in every local worker env (`~/.cache/uv/environments-v2/stems-worker-*`), earliest env dated 2026-06-30 | `ls` of dist-info | verified |
| Model | `ICASSP_2022_MODEL_PATH` = `basic_pitch/saved_models/icassp_2022/<variant>`; the variant is chosen at import time: TF SavedModel `nmp/` if `tensorflow` imports, else CoreML `nmp.mlpackage` if `coremltools` imports, else `nmp.tflite`, else `nmp.onnx` | `basic_pitch/__init__.py:78-95` (0.4.0) | verified |
| Which variant ran on the Linux boxes | UNKNOWN. On Linux, 0.4.0 declares `tensorflow<2.15.1,>=2.4.1` for Python ≥3.11 and `tflite-runtime` for Python <3.11 (`METADATA` Requires-Dist); the worker capped Python at <3.12 and installed the `[onnx]` extra. The worker docstring says "inference still runs via ONNX; TF is just an install-time dependency", but by the import-order rule above ONNX is used only if neither TF, coremltools nor tflite-runtime imports. Not resolvable without the box's env | `basic_pitch/__init__.py`; `stems_worker.py` docstring | UNKNOWN |
| SHA-256 of bundled model files (0.4.0 wheel contents, local env) | `nmp.onnx` `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec` (230,444 B); `nmp.tflite` `3db297d54af8e01c6e5618245c956b1d71b6a2b978cb2dedb527173186552676` (204,448 B); `nmp/saved_model.pb` `eaa25c91c431c91100c416a2c018663f4c635f28fa19529c4ff5e14c18aa29c9`; `nmp/variables/variables.data-00000-of-00001` `f5d12cd7245fecea0c956c963751f3519c263615ea954b5948d5e8c9c3376f9b`; `nmp/variables/variables.index` `356aa1a00095cf2dba17386144e8b289cb04195ae090aa7f324312b08220115e`; `nmp.mlpackage/Data/com.apple.CoreML/model.mlmodel` `af7bf7d49bc167e0bf0c30aa2ca6b432c3e10df048d2dd4173ff3a738c020858`; `nmp.mlpackage/Data/com.apple.CoreML/weights/weight.bin` `691a6b63c7ddcdde0ee131ff3986dcb1250df47cd738612efde966ba9b4c99cd`; `nmp.mlpackage/Manifest.json` `c1fa5ef8acc34703edcd4e90e9a8640bd4673d9f3a68753c3b9d1ca0365e2928` | `shasum -a 256` over `~/.cache/uv/environments-v2/stems-worker-b624957fa85a6c78/.../saved_models/icassp_2022/` | verified (all variants ship in the same wheel, so the wheel hash above pins all of them) |

### 2d. Drum transcriber in the S4/T10 archives (ADTOF-pytorch port)

| item | value | source | status |
|---|---|---|---|
| Declared dependency at T1/S4/T10 time | `adtof-pytorch @ git+https://github.com/xavriley/ADTOF-pytorch` (no ref pinned); "ADTOF Frame_RNN (PyTorch port) … no declared license (it ports GPL'd ADTOF)" | `git show cf28158:src/jams/data/drum_worker.py` header | verified |
| Commit resolved on the T10 box | `85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9` ("Updated https://github.com/xavriley/ADTOF-pytorch (85c192e7…)" / "Built adtof-pytorch @ git+…@85c192e7…"); upstream commit date 2025-11-11T15:46:47Z, message "Allow for model activations to be returned" | `s3://jams-mir-eval-usw2/lambda/e2eval/e2e.log` lines 56-59; GitHub API `commits/85c192e7…` | ledger-quoted (log) + commit existence verified |
| Weights file loaded on the T10 box | `.../site-packages/adtof_pytorch/data/adtof_frame_rnn_pytorch_weights.pth` (repeated per track in the log) | `e2e.log` | ledger-quoted; **SHA-256 UNKNOWN** |
| Commit resolved for T1 and S4 | UNKNOWN (no log; the dependency was unpinned). The ledger notes for D1 external diagnostics: "ADTOF-pytorch … upstream commit unpinned — provenance caveat if the paper needs it" (`EXPERIMENTS.md` D1, line ~366) | — | UNKNOWN |
| Evidence the same drum path served S4 and T10 | S4 `drum_onset_f` 0.5741 and T10 `drum_onset_f` 0.5741 (both archives), and ledger T10 note (a): "drums onset-F 0.5741 matches S4 exactly (drums path unchanged by the transcriber swap)" | archives; `EXPERIMENTS.md` T10 notes | verified equality; causal reading is ledger-quoted |
| License note | ledger D1: weights "mechanically converted from ADTOF originals that are CC BY-NC-SA 4.0 … the port is unlicensed" | `EXPERIMENTS.md` D1 motivation | ledger-quoted |
| Velocity | port emits flat velocity 100; the T1-era worker replaced it with a 30 ms RMS estimate (T8); onset scoring unaffected | `cf28158:drum_worker.py` `_velocities_from_audio`; ledger T8 | verified/ledger-quoted |

The current `src/jams/data/drum_worker.py` runs the in-house D1 CRNN (`drum_cnn_v1.pt`) and is
**not** the model behind any of the four rows (ledger D1, `eval/README.md` "Drum model").

### 2e. In-house bundled models (`src/jams/data/models/`)

| file | SHA-256 | note | status |
|---|---|---|---|
| `key_cnn_v1.pt` | `18c24e61cf779f399014c2deaaa156d1010c56dac8e463809830bfd354a0b4a4` | not used by the four rows | verified |
| `tempo_cnn_v1.pt` | `6f3a21c5b7a8f8e3f1326f2953416cfc92c71b113aaf9327edb318529f305a69` | not used by the four rows | verified |
| `drum_cnn_v1.pt` | `08436085a4651532b77f2c542520de876e592939bd43dfa18a63a8f5a1d4afa0`; MD5 `51c693eb0195ac0b369108a560996a32` (the ledger's "md5 51c693eb…" matches) | replaced ADTOF after T10 (PR #19 per ledger S7) | verified |

---

## 3. Environment

### 3a. Where the runs happened (ledger-quoted)

| run | host | evidence |
|---|---|---|
| T1 (and S1, T6) | AWS `g6.2xlarge` (NVIDIA L4): commit `cf28158` message "Verified end-to-end on a g6.2xlarge (L4)"; archives named `results_aws/` | commit message; ledger |
| T2 | an Ubuntu box with a venv at `/home/ubuntu/mt3_venv` (uv warning captured in `yourmt3_final.txt`); the ledger gives no instance name | archive file |
| S4 | UNKNOWN box; archive mtimes 2026-07-02 16:42 (local) | archive mtimes |
| T2b, T10 | Lambda box "jams-e2eval" (ledger T10: "~4dd3f9e (box jams-e2eval)"); log paths `/home/ubuntu/jams/...`; artifacts under `s3://jams-mir-eval-usw2/lambda/e2eval/` and `lambda/instance-jams-e2eval/ALL_DONE` (2026-07-05 22:46) | ledger; S3 |

Ledger dates vs commit timestamps (verified): T2 commit `4d466e2` = 2026-07-02 18:51:42 −0700
and S4 commit `8f14ff4` = 2026-07-02 18:31:49 −0700, while the ledger dates both rows "Jul 3"
(consistent with UTC). T10's ledger commit `1d32b7a` = 2026-07-05 22:55:36 −0700.

### 3b. Lockfiles

- **Scorer/orchestrator environment** (`uv run --extra eval eval/evaluate_transcription.py`
  runs in the project env): `uv.lock` is committed. Versions extracted from `uv.lock` at the T1
  commit `cf28158`, at the T10 ledger commit `1d32b7a`, and at HEAD:

  | package | `cf28158` (T1) | `1d32b7a` (T10) | HEAD `b22d085` |
  |---|---|---|---|
  | librosa | 0.11.0 | 0.11.0 | 0.11.0 |
  | mir-eval | 0.8.2 | 0.8.2 | 0.8.2 |
  | mirdata | 1.0.0 | 1.0.0 | 1.0.0 |
  | numpy | two entries: 2.2.6 and 2.4.6 (marker-dependent) | 2.2.6 and 2.4.6 | 2.2.6 |
  | pretty-midi | 0.2.11 | 0.2.11 | 0.2.11 |
  | soundfile | 0.14.0 | 0.14.0 | 0.14.0 |
  | torch | not in lock (project depended on `essentia-tensorflow`, not torch) | not in lock | 2.8.0 |

  Status: verified from the lock text. Which numpy entry applied on the boxes' interpreter is
  not determined here (UNKNOWN). Project Python pin at `cf28158`: `.python-version` = `3.14`;
  `requires-python = ">=3.10"` (verified).

- **Worker environments** (PEP 723 inline manifests; each worker resolves its own env at first
  run): **no lockfile exists** in the repository for any worker at any of the ledger commits
  (no `*.lock` next to the scripts), and none was found in S3. The inline manifests were:

  | worker | commit | manifest (verbatim) |
  |---|---|---|
  | `stems_worker.py` | `cf28158` (T1) and `8f14ff4` (S4) | `requires-python = ">=3.10,<3.12"`; `demucs>=4.0`, `basic-pitch[onnx]>=0.4`, `soundfile>=0.12`, `numpy>=1.23,<2`, `librosa>=0.10` (+ `pyyaml>=6.0` at `8f14ff4`). **torch unpinned** (pulled in by demucs) |
  | `stems_worker.py` | HEAD (T10-era pins per `4dd3f9e`) | adds `torch==2.8.*`, `torchaudio==2.8.*` |
  | `yourmt3_worker.py` | `4d466e2` (T2) | `requires-python = ">=3.10,<3.13"`; `mt3-infer>=0.1.3`, `pytorch-lightning>=2.2`, `transformers==4.45.1`, `pretty_midi>=0.2.10`, `librosa>=0.10`, `soundfile>=0.12`; docstring: "torch is deliberately NOT pinned" |
  | `yourmt3_worker.py` | HEAD (`4dd3f9e`) | adds `torch==2.8.*`, `torchaudio==2.8.*` |
  | `drum_worker.py` | `cf28158` | `requires-python = ">=3.10,<3.13"`; `adtof-pytorch @ git+https://github.com/xavriley/ADTOF-pytorch`, `torch>=2.0`, `librosa>=0.10`, `pretty_midi>=0.2.10`, `numpy` |

  Commit `4dd3f9e` (2026-07-05 19:01 −0700) states the torch pins were "fixes [that] previously
  existed only on the now-terminated instance" and that unpinned resolution had "pulled a cu130
  torch newer than the box's CUDA driver and every worker silently ran on CPU" — i.e. the exact
  worker code on the T10 box was a local modification of the tree, hence the ledger's "~4dd3f9e"
  (ledger-quoted). The T10 log shows `[stems] SCNet ready on cuda` and `Device: cuda` for
  YourMT3 (ledger-quoted from `e2e.log`).

- **Interpreter versions on the T10 box** (from env paths in `e2e.log`, verified by reading the
  log): stems worker `python3.10`, YourMT3 worker `python3.11`, drum worker `python3.11`.
  Package versions inside those envs: **UNKNOWN** (not logged; the boxes were terminated).

- **Local macOS worker envs** (not the historical boxes; listed only to show what the same
  manifests resolve to here): Jul 2 `yourmt3-worker-f83dfec3487ab0c4` → mt3_infer 0.1.3, torch 2.8.0,
  torchaudio 2.8.0, numpy 2.2.6, librosa 0.11.0, pretty_midi 0.2.11, transformers 4.45.1,
  pytorch_lightning 2.6.5; Jul 2 `stems-worker-7248b3818dcafd84` → basic_pitch 0.4.0, demucs 4.0.1,
  torch 2.8.0, numpy 1.26.4, librosa 0.11.0, onnxruntime 1.23.2, coremltools 9.0, pyyaml 6.0.3
  (verified from dist-info names).

- `s3://jams-mir-eval-usw2/lambda/requirements.txt` (3,061 B, 2026-07-03) is the pip freeze for
  the **Raveform structure-training** boxes (`provision.sh` installs it into `~/venv` for
  `all-in-one`), not for the transcription runs; `lambda/code.tar.gz` (197,130 B, 2026-07-05
  20:31) contains the `all-in-one/` trainer tree (104 entries), not the jams repo (verified by
  listing). No environment freeze for the transcription boxes exists in S3.

---

## 4. Run configurations

Evaluator flags (verified in `eval/evaluate_transcription.py` at `cf28158` and HEAD; the diff
between them is 11 lines and does not change scoring): `--mode {oracle,e2e}` default `oracle`;
`--drum-classes {adtof5,gm10}` default `adtof5`; `--bass-octave-shift N` default `0`;
`--quantize` default off; `--no-drums`; `--limit`; `--out`; `--fresh`. Scoring:
`mir_eval.transcription.precision_recall_f1_overlap(onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)`;
drums `mir_eval.onset.f_measure(window=0.05)` per class, macro over classes present; SI-SDR via
`librosa.load(sr=None, mono=True)` on the reference, estimate resampled to it, truncated to the
shorter length.

Where notes are post-processed (verified):

- at `cf28158` (T1): `stems_worker.py` applied `BASS_OCTAVE_SHIFT = 12` **inside the worker**
  to basic-pitch bass output and applied the monophonic filter to bass/vocals; so T1's bass score
  includes +12 even though `--bass-octave-shift` defaults to 0;
- from `4d466e2` (T2) onward: the +12 shift and the monophonic filter moved to the orchestrator
  (`jams.analysis.gm.shift_bass_notes`, `monophonic_filter`; `jams/analysis/stems.py`) and apply
  to both transcribers; the workers emit sounding pitch. The T2 archive (`yourmt3_notes.jsonl`)
  contains sounding-pitch notes.

Exact command lines: **not recorded** for T1, T2, S4 or T10 in the repository, the commit
messages, or the S3 logs (the S3 `e2e.log` starts at the evaluator's own stderr line
`=> Scoring 151 slakh tracks (mode=e2e)`). What is recorded:

| row | ledger row (`EXPERIMENTS.md` "Transcription"/"Separation" tables) | additional recorded facts | status |
|---|---|---|---|
| T1 | "Jul 2 · cf28158 · basic-pitch (tuned: other onset 0.6/frame 0.25) + ADTOF drums · 0.7889 · 0.4897 · 0.6383 · results_aws/slakh_test_oracle.json" | commit `cf28158` message: "Slakh2100-redux test split (151 tracks, 44.1 kHz, 0 failures both modes): oracle bass 0.789 / other 0.490 note-F, drums 0.638 onset-F"; archive `mode: oracle`. Separator irrelevant (oracle). Drum classes: archive per-class keys 36/38/42/47/49 ⇒ `adtof5` | ledger-quoted; archive verified |
| T2 | "Jul 3 · 4d466e2 · YourMT3+ (YPTF.MoE+Multi via mt3-infer), +12 bass convention · 0.8486 · 0.8488 · (drums stay ADTOF) · results_aws/yourmt3_notes.jsonl, yourmt3_scores.json" | the script that produced `yourmt3_notes.jsonl` / `yourmt3_scores.json` is not in the repository history (`git log -S` finds only ledger/backfill references) — **NOT RECOVERABLE**; archive bass figure is 0.1191 (see §1) | ledger-quoted |
| T2b | "Jul 5 · (this) · T2 re-scored per-track against Slakh GT (same scoring fns; aggregates match T2 to 4 dp) → paired bootstrap vs T1 …" | commit `57840dd` (2026-07-05 18:56 −0700): "re-scored on the e2e eval box against Slakh GT with the same scoring functions"; S3 `rescore.log`: `aggregate note_f: {'bass': 0.8486, 'other': 0.8488}  (rows=294, missing=0)`; the rescore script is not in the repository or in `lambda/e2eval/` — **NOT RECOVERABLE**. `eval/stats_significance.py` only reads the per-track file | ledger-quoted |
| S4 | "Jul 3 · 8f14ff4 · SCNet XL IHF (ZFTurbo MSST zoo) · 14.31 / 11.77 / 5.98 · 0.6448 · 0.4733 · 0.5741 · sep_scnet_{sdr,notes}.json" | `eval/README.md` "Separation-model A/B": "Candidates from the MSST zoo scored through the full pipeline (SI-SDR vs GT stems + note-F of transcription run on the separated stems, same oracle-mode protocol for all)"; archive `mode: oracle`; `sep_scnet_sdr.json` other = 11.765 (ledger rounds to 11.77). The A/B driver script is not in the repository — **NOT RECOVERABLE** | ledger-quoted; archive verified |
| T10 | "Jul 5 · ~4dd3f9e (box jams-e2eval) · shipped system, e2e: mix → SCNet XL IHF → YourMT3+ (pitched) + ADTOF (drums); n=151, 0 failures · 0.6613 · 0.7877 · 0.5741 · results_aws/slakh_test_e2e_scnet_yourmt3.json" | S3 `e2e.log` tail: `tracks: 151 scored, 0 failed`, `note_f … {'bass': 0.6613, 'other': 0.7877, 'vocals': None, 'overall': 0.7262}`, `drum_onset_f (macro): 0.5741`, `SI-SDR dB per stem: {'drums': 14.3098, 'bass': 5.9794, 'other': 11.7645}`, `=> Wrote eval/data/results_aws/slakh_test_e2e_scnet_yourmt3.json`. Separator: `config.stems_model` default `scnet_xl_ihf` (HEAD; the S4 commit made it default); transcriber default `yourmt3`; quantize: evaluator default off; bass +12 in orchestrator; drum classes `adtof5` (archive per-class keys 36/38/42/47/49) | ledger-quoted; archive verified |

Two later runs re-executed the T10 configuration and are recorded with their commands:

- **S7 arm A** (2026-07-14, aleph0): `s3://jams-mir-eval-usw2/s7/run_slakh_arms.sh` runs
  `JAMS_STEMS_OUT_DIR=$HOME/s7/out/slakh_armA uv run --extra eval eval/evaluate_transcription.py --manifest eval/data/slakh/manifest.jsonl --mode e2e --out $HOME/s7/results/slakh_armA.json`
  from `~/s7/jams`; `s7_summary.json` records `repo_commit: 73640599f896bf7045a5e6abe41c44e6e781fe44 (feat/two-pass-separation, PR #24)`,
  box "aleph0 (WSL2 x64, RTX 4090, cuda)", arm-A means bass 0.66259 / other 0.78808 / SI-SDR
  drums 14.30983 / bass 5.97949 / other 11.76455 (verified by reading the S3 files). The
  ledger (S7 verdict) calls this a reproduction of T10's separation numbers "exactly"; the
  per-track note-F files are `s7/slakh/slakh_armA.json` and `slakh_armA_pertrack.jsonl` in S3.
  Drums in S7 use the D1 CNN, not ADTOF.
- **OV1 confirmatory** (2026-07-14, aleph0): pooled note-F on "other" 0.8488 with the same
  checkpoint, "independent implementations, identical number" (`EXPERIMENTS.md` OV1
  confirmatory paragraph; ledger-quoted).

### Note-decoding parameters

**basic-pitch** (verified in `stems_worker.py` at `cf28158` lines 176-185 and at HEAD lines 361-370;
defaults from `basic_pitch/inference.py:414-425` in 0.4.0):

| parameter | basic-pitch 0.4.0 default | worker value | note |
|---|---|---|---|
| `onset_threshold` / `frame_threshold` | 0.5 / 0.3 | **other: 0.6 / 0.25**; bass, vocals: 0.5 / 0.3 | `ONSET_FRAME = {"other": (0.6, 0.25)}`; worker comment: "(0.6, 0.25) scored 0.468 vs the default (0.5, 0.3)'s 0.445 note-F in a sweep on babyslakh ground-truth stems" (= ledger T4, Jul 1, commit `6941991`, "tmp sweep logs") |
| `minimum_note_length` (ms) | 127.70 | 90.0 for bass/vocals, 58.0 for other | |
| `minimum_frequency` / `maximum_frequency` (Hz) | None / None | bass 30–400; vocals 65–2100; other None | |
| `multiple_pitch_bends` | False | False | |
| `melodia_trick` | True | default (True) | not overridden |
| `midi_tempo` | 120 | default | affects only the discarded MIDI object |
| post-filter | — | monophonic filter on bass/vocals (loudest-first greedy overlap removal) | in-worker at `cf28158`; orchestrator from `4d466e2` |

The T4 sweep itself (grid, seeds, per-setting scores beyond the two quoted numbers) is
**NOT RECOVERABLE** ("tmp sweep logs"); local env dirs `bp-sweep-855f3c9c8c8956b2` and
`bp-sweep2-53cb9844eb5e621f` (2026-07-01) show that a sweep script ran on this Mac, but the
script is not in the repository.

**YourMT3+** (verified in `yourmt3_worker.py` at `4d466e2` and HEAD, and in mt3-infer 0.1.3
`api.py` / `adapters/yourmt3.py`): the worker loads the stem with
`librosa.load(wav, sr=16000, mono=True)` and calls `mt3_infer.transcribe(y, model="yourmt3", sr=16000)`
with no other keyword arguments, i.e. mt3-infer defaults: `device="auto"`, `adaptive=False`
(no time-stretch retries). The adapter resamples to the checkpoint's `audio_cfg.sample_rate`
if needed, slices the signal into non-overlapping segments of `audio_cfg.input_frames`, runs
`model.inference_file(bsz=8, …)`, detokenizes every decoding channel and merges them with
`mix_notes`, and writes a MIDI via `write_model_output_as_midi`. The worker then reads the MIDI
with `pretty_midi` and **discards every instrument with `is_drum`**, flattening the rest into one
instrument-agnostic note list (onset/offset rounded to 4 decimals). No thresholds are exposed or
set. The T10 log's "Running inference on 127 segments… Got 16 prediction batches" is consistent
with `bsz=8` (ledger-quoted from `e2e.log`).

---

## 5. Data identities

### Slakh2100-redux test split

| item | value | source | status |
|---|---|---|---|
| Zenodo record | `10.5281/zenodo.4599666` (concept DOI `10.5281/zenodo.4599665`), title "Slakh2100", version string `slack2100-redux` (sic), publication date 2019-10-20 | Zenodo API `records/4599666` | verified |
| Archive | `slakh2100_flac_redux.tar.gz`, 104,322,767,708 B, **MD5 `f4b71b6c45ac9b506f59788456b3f0c4`** | Zenodo API | verified (Zenodo publishes MD5 only) |
| mirdata index | Zenodo `10.5281/zenodo.14009687`, "mirdata-slakh_index_2100-redux", file `slakh_index_2100-redux.json` 11,010,681 B, MD5 `7eaefceadb16f1d3621b5dce4b7867c3` (the ledger's "canonical CC BY index (Zenodo 14009687)" is this per-file MD5 index; a copy of the same size sits at `s3://jams-mir-eval-usw2/ov1/scripts/slakh_index_redux.json`) | Zenodo API; S3 listing | verified |
| Loader | `mirdata.initialize("slakh", version="2100-redux")`, mirdata 1.0.0 in `uv.lock` at all three commits | `eval/acquire_slakh.py:205-207`; `uv.lock` | verified |
| Split selection | `--subset full --split test`; rows kept where `mtrack.split == "test"`; `audio_exists` rows only | `acquire_slakh.py:252, 296` | verified |
| Track IDs (151) | `Track01876 … Track02098` (151 IDs; the full sorted list is in `paper/evidence/publication_verified.json` under each archive's `track_ids`; all four archives carry the identical list) | `publication_verified.json` | verified |
| Per-file verification on the boxes | T1/S4/T10 boxes: **UNKNOWN** (no md5 log in S3 for those runs). Later runs recorded verification: D1 gate — "Slakh test audio fetched per-file from the gated HF mirror and md5-verified bit-identical to the canonical CC BY index (Zenodo 14009687)" (`EXPERIMENTS.md` D1 execution notes); OV1 — `ov1_fetch.py` verifies every audio/MIDI/metadata file's MD5 against the index and exits non-zero on mismatch (script text read from S3); S7 — "mixes from the md5-verified d1gate copy, GT stems+MIDI from the OV1 conf-test staging on aleph0" (`EXPERIMENTS.md` S7 data) | ledger-quoted |
| Local copy of the redux test set | none on this Mac (`eval/data/slakh/redux/` absent; `oracle_redux.partial.jsonl` is an 80-byte stub) | `ls` | verified absent |

### babyslakh (16 kHz preview) — development subset

| item | value | source | status |
|---|---|---|---|
| Source | `https://zenodo.org/record/4599666/files/babyslakh_16k.tar.gz` (same Zenodo record as above; file not in that record's current file list, which shows only the redux tarball — **the babyslakh URL was not re-verified**) | `acquire_slakh.py:65` `_BABYSLAKH_HELP` | ledger-quoted (script text) |
| Local copy | `/Users/jhurliman/Documents/Code/jhurliman/jams/eval/data/slakh_home/babyslakh_16k.tar.gz` and extracted `babyslakh_16k/Track00001…`; tarball hash not computed | `ls` | verified present |
| Loader | `mirdata.initialize("slakh", version="baby")`; `split == None` | `acquire_slakh.py:205, 249` | verified |
| Dev IDs | 20 multitracks `Track00001`–`Track00020` in `eval/data/slakh/manifest.jsonl` (main checkout, 2026-07-01); 19 have a bass stem/MIDI, 20 have other/drums | manifest read | verified |
| Dev results | `eval/data/slakh/oracle_babyslakh.json`: n=20, failed=0, bass 0.7987 (19 tracks), other 0.4681, drums 0.4548 — matches `eval/README.md` "Dev subset (babyslakh, 20 tracks, 16 kHz …): oracle bass 0.799 / other 0.468 / drums 0.455" and ledger T3 "all 19 tracks improved" | file read; README | verified/ledger-quoted |
| What was tuned on it | T3 (+12 bass octave, commit `6941991`), T4 (basic-pitch "other" thresholds 0.6/0.25), T5 (quantize ablation), T7 (bandwidth finding) — all "Jul 1 · 6941991" | `EXPERIMENTS.md` Transcription table | ledger-quoted |
| Relationship to the test split | Multitrack IDs `Track00001–00020` do not appear in the 151 test IDs (ID ranges are disjoint). The ledger states babyslakh tracks are from the **train** split ("babyslakh Track00001, TRAIN split", OV1; "Tier-0 null on babyslakh TRAIN (Tracks 1/2/6 …)", OV2). Source-MIDI (Lakh) overlap between the babyslakh tracks and the test split: **UNKNOWN** — not checked here (REVIEW.md step 3 asks for this) | manifest; ledger | verified (ID disjointness) / UNKNOWN (MIDI overlap) |

### Reference-grouping rule (`eval/acquire_slakh.py`, unchanged in substance since `cf28158`; verified)

- `classify(track)`: `drums` if `track.is_drum`; `bass` if `track.program_number` is not None
  and `32 <= program_number <= 39`, **or** `track.instrument == "Bass"`; else `other`. (Whether
  mirdata's `program_number` is zero-indexed is not verified here; the paper says "zero-indexed".)
- `sum_audio(paths, out)`: reads each member stem with `soundfile.read(always_2d=True)`; a stem
  that fails to read is **skipped with a warning** (not fatal); `n = min(frames)` — the group
  audio is **truncated to the shortest member**; channel count = max over members, mono members
  duplicated to stereo; members are **summed in float64 with no gain compensation or peak
  limiting**; written with `sf.write(out_path, mix.astype(np.float32), sr)` — **no `subtype`
  argument**.
- WAV subtype actually written: `soundfile.default_subtype("WAV")` returns **`PCM_16`** in the
  pinned soundfile 0.14.0 (verified by calling it in the worktree venv). libsndfile converts the
  float32 buffer to 16-bit PCM, so any summed sample outside [−1, 1) is **clipped**. Verified on
  the local babyslakh groups: `Track00001/{drums,bass,other}.wav` are `PCM_16`, 16,000 Hz, mono,
  3,864,916 frames each, peak |x| = 0.7397 / 0.3978 / 0.5273 (no clipping in that track). Peaks
  of the 151 redux test groups: **UNKNOWN** (files not available locally).
- `merge_midi(paths, is_drum_group, out)`: one `pretty_midi.Instrument` per source instrument,
  `is_drum = is_drum_group or inst.is_drum`, notes/control changes/pitch bends copied; an
  unreadable MIDI is skipped with a warning; returns False only if no instrument was added.
  MIDI merging is independent of audio readability, so a group can have MIDI for a stem whose
  audio was skipped (the REVIEW.md step 4 risk).
- Reference notes at scoring time (`evaluate_transcription._midi_to_notes`): every note of every
  instrument in the group MIDI, instrument labels discarded; drums canonicalised via
  `gm.canon_drum_pitch` then reduced to 5 classes for `adtof5`.
- Between `cf28158` and HEAD the only change to `acquire_slakh.py` is the data-presence probe
  (scan all IDs instead of a stride sample) and a docstring note (`git diff`, 13 lines).

---

## 6. Summary table

| artifact | identity (URL / repo / file) | hash | version / revision | source of the fact | status |
|---|---|---|---|---|---|
| T1 archive | `paper/evidence/results_aws/slakh_test_oracle.json` | SHA-256 `03c9566f…edb8` | mtime 2026-07-02 04:31 | shasum; publication_verified.json | verified |
| T2b archive | `…/yourmt3_oracle_per_track.json` | `da4e13d0…0061` | 2026-07-05 18:31 | shasum | verified |
| S4 archive | `…/sep_scnet_notes.json` (+ `sep_scnet_sdr.json` `34290c03…ba9e`) | `19397c11…5e1b` | 2026-07-02 16:42 | shasum | verified |
| T10 archive | `…/slakh_test_e2e_scnet_yourmt3.json` | `7c12670c…12cd` | 2026-07-05 22:46 | shasum; S3 `lambda/e2eval/` | verified |
| T2 raw predictions | `…/yourmt3_notes.jsonl` (294 rows) | `7a0c95f2…e9c1` | 2026-07-02 18:33 | shasum | verified |
| SCNet weights | `github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt` | SHA-256 `ac25975f…b74f` (local cache; size = GitHub asset 214,063,778 B) | release `v1.0.15` (2025-06-18) | worker code; GitHub API; shasum | verified (no upstream digest exists) |
| SCNet config | `…/v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml` | `54ddcb8a…2f40` | `v1.0.15` | same | verified |
| SCNet code | `src/jams/data/scnet/` vendored | — | upstream commit `ccc011abf7f89dd7922bb2888d48493b575c0289` | file header; git diff | verified |
| YourMT3+ checkpoint | HF Space `mimbres/YourMT3`, `amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt` | SHA-256 `ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25` (HF LFS oid = local clone = OV1 record), 561,544,628 B | Space HEAD `5e66c1ea…15b5` (2025-01-31); variant "YPTF.MoE+Multi (noPS)" | HF API; shasum; S3 log | verified |
| mt3-infer | PyPI `mt3-infer` | wheel `974e1d5a…2b78` (0.1.3) | pin `>=0.1.3`; 0.1.3 inferred for Jul 2–5 runs (0.2.0 released 2026-07-11) | worker header; PyPI | pin verified; box version inferred |
| basic-pitch | PyPI `basic-pitch` | wheel `738adb50…5db0` (0.4.0); model files hashed in §2c | pin `>=0.4` → 0.4.0 (latest release) | worker header; PyPI; shasum | pin verified; box version inferred |
| basic-pitch model variant used on Linux | `saved_models/icassp_2022/{nmp, nmp.onnx, nmp.tflite, nmp.mlpackage}` | all in wheel | selected by import availability | `basic_pitch/__init__.py` | UNKNOWN which variant ran |
| ADTOF-pytorch (drums in S4/T10 rows) | `git+https://github.com/xavriley/ADTOF-pytorch`, weights `adtof_frame_rnn_pytorch_weights.pth` | weights hash UNKNOWN | T10 box: commit `85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9`; T1/S4: UNKNOWN (unpinned) | S3 `e2e.log`; GitHub API | ledger-quoted / UNKNOWN |
| Slakh2100-redux | Zenodo `10.5281/zenodo.4599666`, `slakh2100_flac_redux.tar.gz` | MD5 `f4b71b6c45ac9b506f59788456b3f0c4` (104,322,767,708 B) | version `slack2100-redux` | Zenodo API | verified |
| mirdata redux index | Zenodo `10.5281/zenodo.14009687`, `slakh_index_2100-redux.json` | MD5 `7eaefceadb16f1d3621b5dce4b7867c3` | — | Zenodo API | verified |
| Test IDs | 151 IDs `Track01876…Track02098` | — | — | publication_verified.json | verified |
| babyslakh dev set | `babyslakh_16k` (20 tracks `Track00001–00020`; 19 with bass) | tarball not hashed | — | local manifest | verified (IDs) |
| Scorer env | `uv.lock` @ `cf28158` / `1d32b7a` | — | mir_eval 0.8.2, pretty_midi 0.2.11, librosa 0.11.0, mirdata 1.0.0, soundfile 0.14.0, numpy 2.2.6 or 2.4.6 (marker) | lock text | verified |
| Worker envs on the boxes | PEP 723 manifests (no lockfile) | — | torch unpinned at T1/T2/S4; `2.8.*` at T10 (per `4dd3f9e`); exact versions UNKNOWN | worker headers; commit msg | ledger-quoted / UNKNOWN |
| Run commands | — | — | T1/T2/S4/T10 not recorded; S7 arm A command recorded (S3) | — | NOT RECOVERABLE / verified |
| `key_cnn_v1.pt` / `tempo_cnn_v1.pt` / `drum_cnn_v1.pt` | `src/jams/data/models/` | `18c24e61…b4a4` / `6f3a21c5…5a69` / `08436085…4afa0` (md5 `51c693eb…`) | in-repo | shasum | verified (not used by the four rows) |

---

## 7. Author workstation (aleph0) — inspected 2026-09-08 (read-only, over ssh)

Host `aleph0` (WSL2 Linux, 1 TB root volume, ~287 GB free; `/mnt/d/jams` is a 423 GB
Windows volume holding `checkpoints/`, `demix/`, `features/`, `mlflow/`, `rbma13/`,
`stems_json/`, `stems_scratch/`). All digests below were computed on the host with `sha256sum`.

| artifact on aleph0 | SHA-256 | agrees with |
|---|---|---|
| `~/.cache/jams/scnet/model_scnet_ep_36_sdr_10.0891.ckpt` | `ac25975f0f5704f3d1a3c3c251505b7a0f417a22eafe82773440ee4f7e14b74f` | Mac cache (§2a): identical |
| `~/.cache/jams/scnet/config_musdb18_scnet_xl_more_wide_v5.yaml` | `54ddcb8aeeae85c8d0e148741e7d87b0fbd38bd21cffa5d59ba124fe1e672f40` | Mac cache (§2a): identical |
| `~/s7/jams/.mt3_checkpoints/yourmt3/mc13_…_b36_nops/last.ckpt` (561,544,628 B, 2026-07-14) | `ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25` | HF LFS oid, Mac clone, OV1 record (§2b): identical |
| `~/.cache/jams/melroformer/MelBandRoformer.ckpt` | `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e` | ledger S7 arm-B vocals model: identical |
| `~/.cache/jams/melroformer/config_vocals_mel_band_roformer_kj.yaml` | `f63f38eb1e6e40a7db0dade714a5ae257555dd8748f4e774eae8679275a81926` | no prior record |
| `~/s7/jams` HEAD (`git rev-parse`) | `73640599f896bf7045a5e6abe41c44e6e781fe44` | `s7/s7_summary.json`: identical. **Working tree not clean:** `eval/evaluate_transcription.py` and `src/jams/data/stems_worker.py` are modified and `.python-version` deleted; those local edits are not in git and were not diffed here. |
| `/mnt/d/jams/checkpoints/` | `fold1_epoch35.ckpt`, `fold2_epoch20.ckpt`, `fold2_st3.pth`, `fold2_st4.pth`, `fold2_v2_epoch5.ckpt` (same names and sizes as `s3://…/checkpoints/`), plus `fold1_v2.pth`, `fold2_stv2.pth`, `fold2_v2.pth`, `st3_roundtrip.pth`, `stock_fold2_init.pth` | structure-experiment checkpoints (Appendix C); not used by the four transcription rows |
| `~/s7/slakh_home/slakh2100_flac_redux/` | full redux install; test split = 151 tracks with `MIDI/`, `all_src.mid`, `metadata.yaml`, `mix.flac`, `stems/` | source of the T2b rescore references (`paper/rescore_t2b.py`) |
| `~/d1ext/` | ADTOF/SCNet batch scripts, MDB/IDMT manifests and predictions (`pred_adtof_*.jsonl`), `scnet_stems/` | D1 acoustic-kit external evaluations |
| `~/ov1/` | `data/`, `gt_test/`, `gt_val_smoke/`, `out_conf/`, `out_smoke_conf/` | OV1 confirmatory run |
| `~/.cache/huggingface/hub` | `models--adefossez--HTDemucs`, `models--taejunkim--allinone` (plus unrelated image models) | HT-Demucs and All-In-One weights; not hashed |

Grouped-reference audit (2026-09-08, `paper/audit_grouped_refs.py` run on this host against
`~/s7/jams/eval/data/slakh/redux`, report `paper/evidence/grouped_reference_audit.json`):
445 groups; no phantom-MIDI or audio-without-MIDI stems, no truncation, no clipping
(max summed peak 0.927), all WAVs PCM_16 and equal to recomputed sums to 1 LSB, no notes
past audio end. Onset-envelope/MIDI cross-correlation: 417/445 groups have their global
maximum within 0–70 ms after note-on (12–58 ms by class: attack time + 1-frame estimator
bias); for the other 28 (22 drums, 3 bass, 3 other; one-beat aliases on periodic parts) and
for 6 low-correlation dense groups the method cannot exclude an offset, though none shows a
consistent shift.
This closes open item 8 for the references on this host; it does not prove the AWS/Lambda
boxes built identical files, although the script and inputs are the same.

Structure gate arms (2026-09-08): the raw per-track predictions of every gate arm
(`gate_st3.jsonl`, `gate_st4.jsonl`, `gate_ft1_fixed.jsonl`, `gate_ft2_fixed.jsonl` from
`s3://jams-mir-eval-usw2/gates/`; `gate_stock.jsonl` and `gate_stock1.jsonl` from this
host's home directory) and the four scored JSONs are committed under
`paper/evidence/structure/`. `eval/structure_class_cis.py` recomputes the Appendix C
intervals from them (`cis_*.json` alongside): 165/165 fold-2 and 162/162 fold-1
coverage, zero error rows, and the scored artifacts equal the recomputed per-track
metrics. No stock-arm scored JSON was ever produced; the stock metrics are recomputed
from `gate_stock*.jsonl`. The held-out sets are fixed by the committed
`raveform_eval_manifest.jsonl` (folds 1–2, section references inline, `eligible` = audio
retrievable at gate time; Raveform annotations are MIT-licensed), so the recomputation
needs only the Raveform beat CSVs, not the audio.

No YourMT3+/mt3-infer cache exists on the host outside `~/s7/jams/.mt3_checkpoints`. The
SCNet-separated stems of S4/T10 were not found (`/mnt/d/jams/stems_scratch` and `stems_json`
were listed, not inspected). Open item 10 in §8 is closed; the remaining items stand.

## 8. Open items (UNKNOWN / NOT RECOVERABLE) — collected

1. Exact command lines for T1, T2, S4, T10.
2. The scripts that produced `yourmt3_notes.jsonl`/`yourmt3_scores.json` (T2), the T2b rescore,
   the S4 separation A/B driver, and the T4 threshold sweep.
3. Per-track predictions for T1, S4, T10 (scores only survive); the SCNet-separated stems of S4/T10.
4. Package versions inside the worker envs on the AWS/Lambda boxes (torch, numpy, librosa,
   pretty_midi, transformers, mt3-infer, basic-pitch runtime backend); which basic-pitch model
   variant (TF SavedModel vs ONNX vs tflite) executed on Linux.
5. ADTOF-pytorch commit for T1/S4 and the SHA-256 of `adtof_frame_rnn_pytorch_weights.pth`.
6. SHA-256 of the SCNet checkpoint as downloaded on the boxes (only size checked by the code);
   upstream publishes no digest.
7. MD5 verification logs of the Slakh test audio/MIDI for the T1/S4/T10 boxes (only later runs
   recorded verification).
8. ~~Peak amplitudes / clipping status and durations of the 151 grouped `PCM_16` reference WAVs.~~ Audited 2026-09-08 on aleph0 (§7); box-built copies not separately verified.
9. Lakh source-MIDI overlap between the babyslakh dev tracks and the test split.
10. ~~Everything on aleph0 (§7).~~ Inspected 2026-09-08; see §7.
