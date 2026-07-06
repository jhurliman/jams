#!/usr/bin/env -S uv run --extra eval
"""Backfill the paper/EXPERIMENTS.md ledger into MLflow (system of record).

Creates experiments "key-detection", "transcription", "separation" with one run
per ledger entry (K1-K9, T1-T9, S1-S6): params = dataset/split/system, metrics =
the verified headline numbers (95% bootstrap CIs from paper/STATS.md as
*_ci_lo/_ci_hi where computed), tags = ledger_id/commit/date/artifacts.

The numbers here are the ledger's, transcribed verbatim — paper/EXPERIMENTS.md
stays the human-readable narrative; MLflow becomes the queryable DB. Idempotent:
an entry is skipped when a run tagged with its ledger_id already exists.

Tracking server: MLflow on aleph0 (see README "Experiment tracking"). From this
Mac that's the SSH tunnel — default http://localhost:5566, override with
MLFLOW_TRACKING_URI.

    uv run --extra eval eval/mlflow_backfill.py
"""

from __future__ import annotations

import os
import sys

TRACKING = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5566")

# (ledger_id, experiment, run_name, params, metrics, extra_tags)
E = "key-detection"
KEY_COMMON = {"dataset": "GiantSteps Key", "n": 567, "split": "test (single-shot)",
              "metric": "MIREX weighted / exact"}
T = "transcription"
TR_COMMON = {"dataset": "Slakh2100-redux test", "n": 151, "split": "test",
             "stems": "ground-truth (oracle)",
             "metric": "note-F onset+pitch 50ms/50c (drums: onset-F)"}
S = "separation"
SEP_COMMON = {"dataset": "Slakh2100-redux test", "n": 151, "split": "test",
              "metric": "SI-SDR (dB) + downstream note-F"}

LEDGER = [
    ("K1", E, "edma-raw", {**KEY_COMMON, "system": "essentia edma, no refinement"},
     {"weighted": 0.7589, "weighted_ci_lo": 0.7277, "weighted_ci_hi": 0.7885,
      "exact": 0.6878},
     {"commit": "b8c70cf", "date": "2026-07-02",
      "artifacts": "eval/data/gsmtg/keyfeat_gskey.jsonl"}),
    ("K2", E, "edma-modeclf-CONTAMINATED",
     {**KEY_COMMON, "system": "edma + mode clf trained on test set",
      "caveat": "CONTAMINATED - reference only, never cite"},
     {"weighted": 0.8010, "exact": 0.7430},
     {"commit": "legacy", "date": "2026-07-02",
      "artifacts": "src/jams/data/mode_model.json"}),
    ("K3", E, "honest-retrain", {**KEY_COMMON,
     "system": "cues-only mode logistic, GS-MTG-fit, thr 0.60 (CV)"},
     {"weighted": 0.8095, "weighted_ci_lo": 0.7799, "weighted_ci_hi": 0.8372,
      "exact": 0.7531},
     {"commit": "b8c70cf", "date": "2026-07-02",
      "artifacts": "replayed in eval/stats_significance.py"}),
    ("K4", E, "skey-standalone", {**KEY_COMMON,
     "system": "deezer/skey (self-supervised, MIT, in-repo ckpt)"},
     {"weighted": 0.8168, "weighted_ci_lo": 0.7887, "weighted_ci_hi": 0.8434,
      "exact": 0.7478},
     {"commit": "b8c70cf", "date": "2026-07-02",
      "artifacts": "eval/data/gsmtg/skey_gskey.jsonl"}),
    ("K5", E, "fusion-mode-only", {**KEY_COMMON,
     "system": "18-feat mode logistic (cues+conf+skey), thr 0.70 (CV)"},
     {"weighted": 0.8102, "exact": 0.7549},
     {"commit": "b8c70cf", "date": "2026-07-02", "artifacts": "tmp key_fusion.py E"}),
    ("K6", E, "production-fusion", {**KEY_COMMON,
     "system": "mode + rerank heads, thr 0.70/0.60 (CV) - SHIPPED default"},
     {"weighted": 0.8123, "weighted_ci_lo": 0.7831, "weighted_ci_hi": 0.8402,
      "exact": 0.7566},
     {"commit": "b8c70cf", "date": "2026-07-02",
      "artifacts": "src/jams/data/key_fusion.json; replay in paper/STATS.md"}),
    ("K7", E, "rule-tonic-agree", {**KEY_COMMON,
     "system": "rule: tonic-agree -> refined else S-KEY (H1)"},
     {"weighted": 0.8145, "exact": 0.7460},
     {"commit": "b8c70cf", "date": "2026-07-02", "artifacts": "tmp experiment"}),
    ("K8", E, "oracle-ceiling", {**KEY_COMMON,
     "system": "oracle max(refined, S-KEY) - upper bound, not a system"},
     {"weighted": 0.8683},
     {"commit": "b8c70cf", "date": "2026-07-02", "artifacts": "tmp experiment"}),
    ("K9", E, "madmom-cnn", {**KEY_COMMON,
     "system": "madmom CNNKeyRecognitionProcessor (Korzeniowski & Widmer 2018)",
      "license": "weights CC BY-NC-SA - eval-only, not shipped"},
     {"weighted": 0.8328, "weighted_ci_lo": 0.8063, "weighted_ci_hi": 0.8580,
      "exact": 0.7725},
     {"commit": "2bc2569", "date": "2026-07-03",
      "artifacts": "eval/data/gsmtg/madmom_gskey.jsonl"}),

    ("T1", T, "basic-pitch+adtof", {**TR_COMMON,
     "system": "basic-pitch (other onset 0.6/frame 0.25) + ADTOF drums"},
     {"bass_note_f": 0.7889, "bass_ci_lo": 0.7624, "bass_ci_hi": 0.8132,
      "other_note_f": 0.4897, "other_ci_lo": 0.4742, "other_ci_hi": 0.5055,
      "drums_onset_f": 0.6383},
     {"commit": "cf28158", "date": "2026-07-02",
      "artifacts": "eval/data/results_aws/slakh_test_oracle.json"}),
    ("T2", T, "yourmt3-plus", {**TR_COMMON,
     "system": "YourMT3+ (YPTF.MoE+Multi via mt3-infer), +12 bass - SHIPPED default"},
     {"bass_note_f": 0.8486, "other_note_f": 0.8488},
     {"commit": "4d466e2", "date": "2026-07-03",
      "artifacts": "eval/data/results_aws/yourmt3_notes.jsonl, yourmt3_scores.json"}),
    ("T3", T, "bass-octave-convention", {"dataset": "babyslakh", "n": 19,
     "system": "basic-pitch bass +12 semitones = written-pitch convention"},
     {"bass_note_f_before": 0.04, "bass_note_f_after": 0.80},
     {"commit": "6941991", "date": "2026-07-01", "artifacts": "eval/README.md"}),
    ("T4", T, "basicpitch-threshold-sweep", {"dataset": "babyslakh GT stems",
     "system": "basic-pitch 'other' onset/frame sweep -> (0.6, 0.25)"},
     {"other_note_f_default": 0.445, "other_note_f_tuned": 0.468},
     {"commit": "6941991", "date": "2026-07-01", "artifacts": "tmp sweep logs"}),
    ("T5", T, "quantize-ablation", {"dataset": "babyslakh", "n": 19,
     "system": "snap onsets to GT beat grid (ablation: quantize costs accuracy)"},
     {"bass_delta_pt": -2.5, "other_delta_pt": -0.3, "drums_delta_pt": -1.7},
     {"commit": "6941991", "date": "2026-07-01", "artifacts": "eval/README.md"}),
    ("T6", T, "egmd-drums", {"dataset": "E-GMD test", "n": 500,
     "system": "ADTOF-pytorch drums, isolated e-kit audio", "metric": "onset-F macro"},
     {"drums_onset_f": 0.6449},
     {"commit": "cf28158", "date": "2026-07-02",
      "artifacts": "eval/data/results_aws/egmd_oracle.json"}),
    ("T7", T, "bandwidth-artifact", {"dataset": "babyslakh (16 kHz) vs full",
     "system": "drums onset-F is bandwidth-limited at 16 kHz (no >8 kHz hats)"},
     {"drums_onset_f_16k": 0.455, "drums_onset_f_44k": 0.638,
      "adtof_self_test_macro_f": 1.0},
     {"commit": "6941991", "date": "2026-07-01", "artifacts": "eval/README.md"}),
    ("T8", T, "drum-velocity", {"dataset": "qualitative",
     "system": "per-hit 30 ms RMS velocity, per-class normalized (ADTOF emits 100)"},
     {},
     {"commit": "c907a57", "date": "2026-07-02", "artifacts": "commit message"}),
    ("T9", T, "vocals-smoke", {"dataset": "MedleyDB sample", "n": 1,
     "system": "basic-pitch on GT vocal stem vs MELODY1-derived notes",
      "caveat": "n=1 smoke; 61-track benchmark pending audio access"},
     {"vocals_note_f": 0.5816},
     {"commit": "local", "date": "2026-07-03", "artifacts": "MedleyDB_sample"}),
    ("T10", T, "shipped-system-e2e", {"dataset": "Slakh2100-redux test", "n": 151,
     "system": "e2e mix -> SCNet XL IHF -> YourMT3+ (pitched) + ADTOF (drums)",
     "note": "SI-SDR reproduces S4 on an independent box; drums path unchanged; "
             "e2e other exceeds basic-pitch oracle (0.4897)"},
     {"bass_note_f": 0.6613, "other_note_f": 0.7877, "overall_note_f": 0.7262,
      "drums_onset_f": 0.5741, "sdr_drums": 14.3098, "sdr_other": 11.7645,
      "sdr_bass": 5.9794, "failed": 0},
     {"commit": "4dd3f9e", "date": "2026-07-05",
      "artifacts": "eval/data/results_aws/slakh_test_e2e_scnet_yourmt3.json"}),

    ("S1", S, "htdemucs", {**SEP_COMMON, "system": "htdemucs (baseline)"},
     {"sisdr_drums": 11.61, "sisdr_other": 10.13, "sisdr_bass": 4.57,
      "bass_note_f": 0.5957, "other_note_f": 0.4585, "drums_onset_f": 0.5845},
     {"commit": "cf28158", "date": "2026-07-02",
      "artifacts": "eval/data/results_aws/slakh_test_e2e.json"}),
    ("S2", S, "htdemucs-sub50", {**SEP_COMMON, "n": 50,
     "system": "htdemucs, 50-track A/B subset"},
     {"sisdr_drums": 10.38, "sisdr_other": 10.43, "sisdr_bass": 4.62,
      "bass_note_f": 0.5822, "other_note_f": 0.4521, "drums_onset_f": 0.6049},
     {"commit": "cf28158", "date": "2026-07-02",
      "artifacts": "eval/data/results_aws/slakh_test_e2e_50.json"}),
    ("S3", S, "htdemucs-ft-sub50", {**SEP_COMMON, "n": 50,
     "system": "htdemucs_ft (~4x cost), 50-track A/B subset"},
     {"sisdr_drums": 11.00, "sisdr_other": 10.90, "sisdr_bass": 4.83,
      "bass_note_f": 0.6070, "other_note_f": 0.4545, "drums_onset_f": 0.6049},
     {"commit": "cf28158", "date": "2026-07-02",
      "artifacts": "eval/data/results_aws/slakh_test_e2e_ft50.json"}),
    ("S4", S, "scnet-xl-ihf", {**SEP_COMMON,
     "system": "SCNet XL IHF (ZFTurbo MSST zoo) - SHIPPED default"},
     {"sisdr_drums": 14.31, "sisdr_other": 11.77, "sisdr_bass": 5.98,
      "bass_note_f": 0.6448, "other_note_f": 0.4733, "drums_onset_f": 0.5741},
     {"commit": "8f14ff4", "date": "2026-07-03",
      "artifacts": "eval/data/results_aws/sep_scnet_{sdr,notes}.json"}),
    ("S5", S, "bs-roformer", {**SEP_COMMON, "n": 47,
     "system": "BS Roformer 4-stem (MSST zoo)",
      "caveat": "n=47 subset (checkpoint redownload mid-run)"},
     {"sisdr_drums": 13.11, "sisdr_other": 8.57, "sisdr_bass": 5.74,
      "bass_note_f": 0.6283, "other_note_f": 0.4682, "drums_onset_f": 0.5964},
     {"commit": "8f14ff4", "date": "2026-07-03",
      "artifacts": "eval/data/results_aws/sep_bsrofo_{sdr,notes}.json"}),
    ("S6", S, "htdemucs-6s-disqualified", {**SEP_COMMON,
     "system": "htdemucs_6s: guitar/piano stems dropped by 4-stem contract",
      "caveat": "disqualified - craters 'other'"},
     {"other_note_f_before": 0.42, "other_note_f_after": 0.22, "other_sisdr_delta": -1.2},
     {"commit": "6941991", "date": "2026-07-01", "artifacts": "eval/README.md"}),
]


def main() -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(TRACKING)
    client = MlflowClient(TRACKING)
    try:
        client.search_experiments()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"MLflow server unreachable at {TRACKING} — is the SSH tunnel up? "
                 f"(see README 'Experiment tracking') — {exc}")

    exp_ids: dict[str, str] = {}
    created = skipped = 0
    for ledger_id, exp_name, run_name, params, metrics, tags in LEDGER:
        if exp_name not in exp_ids:
            exp = client.get_experiment_by_name(exp_name)
            exp_ids[exp_name] = exp.experiment_id if exp else client.create_experiment(exp_name)
        eid = exp_ids[exp_name]
        if client.search_runs([eid], filter_string=f"tags.ledger_id = '{ledger_id}'"):
            skipped += 1
            continue
        run = client.create_run(eid, run_name=run_name,
                                tags={"ledger_id": ledger_id, **tags,
                                      "source": "ledger-backfill"})
        rid = run.info.run_id
        for k, v in params.items():
            client.log_param(rid, k, v)
        for k, v in metrics.items():
            client.log_metric(rid, k, float(v))
        client.set_terminated(rid, "FINISHED")
        created += 1
        print(f"  {ledger_id} -> {exp_name}/{run_name}")
    print(f"=> backfill: {created} created, {skipped} already present")


if __name__ == "__main__":
    main()
