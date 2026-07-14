# src/jams/data — inventory

Bundled model weights and self-contained `uv` workers, plus a few artifacts that are
**eval-only**: retired from production but deliberately preserved because
`eval/stats_significance.py` replays the paper's historical baselines from them
(ledger rows in `paper/EXPERIMENTS.md` — the paper claims every CI regenerates from
committed code). Do not delete eval-only files while the paper reports those rows.

| File | Status | Consumer | Deletion constraint |
|------|--------|----------|---------------------|
| `structure_worker.py` | production | `analysis/structure.py` (local All-In-One backend) | — |
| `stems_worker.py` | production | `analysis/stems.py` (separation orchestrator) | — |
| `drum_worker.py` | production | `analysis/stems.py` (drum transcription) | — |
| `yourmt3_worker.py` | production | `analysis/stems.py` (default pitched transcriber) | — |
| `scnet/` | production | `stems_worker.py` (default separation backend) | vendored MIT code, kept byte-faithful to upstream (ruff-excluded in pyproject) |
| `models/key_cnn_v1.pt` | production | `analysis/key_cnn.py` | — |
| `models/tempo_cnn_v1.pt` | production | `analysis/tempo_cnn.py` | — |
| `models/drum_cnn_v1.pt` | production | `drum_worker.py` | — |
| `skey_worker.py` | **eval-only** | banks `eval/data/gsmtg/skey_gskey.jsonl`, replayed by `eval/stats_significance.py` | keep while the paper reports K4/K6 |
| `key_fusion.json` | **eval-only** | `eval/stats_significance.py` (replays the retired fusion heads) | keep while the paper reports K6 |
| `mode_model.json` | **eval-only** (archived) | no runtime reader; ledger row K2 artifact (contaminated — reference only); written by `eval/train_mode_model.py` | keep while the paper reports K2 |

The pure replay helpers for the retired fusion system (`_SKEY_ORDER`,
`_parse_skey_key`, `_skey_feats`, `_logistic`, plus `NOTES`/`FLAT_TO_SHARP`/
`_normalize`) live in `analysis/key.py` under an `EVAL-REPLAY ONLY` banner and are
guarded by `tests/test_key_replay_helpers.py`.

**Packaging note:** the three eval-only files currently ship in the wheel —
`skey_worker.py` via hatch's default package-source inclusion, the two JSONs via the
`src/jams/data/*.json` artifacts glob in `pyproject.toml`. Harmless (~8 KB total) but
unnecessary; excluding them is deliberately not done here to keep this change
annotation-only.
