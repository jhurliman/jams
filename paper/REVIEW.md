# Publication audit, 8 September 2026

This review covers PR #5 at `71e61b660a770f0c87539d23dce54f26da5ae032`.
It replaces the earlier claim that a compile-clean draft needed only metadata decisions.
The revised paper is a **retrospective Slakh2100 component study**. Its empirical
estimates remain reported historical results until the underlying archives are recovered.
No model training or inference was rerun during this review.

## Status update, 8 September 2026 (later the same day)

Recovery of the primary archives is complete. The four per-track score archives,
plus the HT-Demucs run S1, were found on the author's machine (`eval/data/results_aws/`,
files dated 2–5 July 2026), committed under `paper/evidence/` with SHA-256 checksums,
and `python paper/verify_results.py --data-dir paper/evidence` passes: 151 matched
IDs in every arm, no duplicate rows, zero recorded failures, means equal to the
four-decimal reported values, and the archived reference-input interval
[0.3471, 0.3713] reproduced exactly. Intervals the audit listed as unavailable are
now recomputed from stored scores: T10−S4 +0.3144 [0.3017, 0.3271] (YourMT3+ ahead on
151/151), S4−T1 −0.0164 [−0.0193, −0.0136], T10−T2b −0.0611 [−0.0677, −0.0547], and
drums S4−S1 −0.0104 [−0.0209, +0.0004] (includes zero). End-to-end bass support is 143
in every arm. The complete Slakh2100-redux and the model caches were located on the
author workstation (aleph0). Using its Slakh MIDI, the T2b predictions were rescored
against independently rebuilt grouped references: all 151 other and 143 bass rows
match the archive exactly (`paper/rescore_t2b.py`, `paper/evidence/rescore_t2b.json`).
That rescore also fixed one documentation error: the archived bass number uses the
+12 shift only, without the monophonic filter STATS.md claimed. `paper/PROVENANCE.md`
records checkpoint hashes (SCNet, YourMT3+), Slakh archive digests, scorer versions,
and the items that are not recoverable (original command lines, basic-pitch and
end-to-end predictions, box-side package versions). Items 1–3 below are done as far
as the evidence allows; item 4 is open; item 5 is done for the score archives; item 6
remains the author's.

Item 4 was then completed on aleph0 against the grouped references the S7 run built
with the same script (151 tracks, 445 groups): zero phantom-MIDI stems, zero
audio-without-MIDI stems, zero truncation, zero clipped samples (max summed peak
0.927), all grouped WAVs PCM_16 and equal to the recomputed float sums to 1 LSB, no
notes past the audio end. Onset-envelope/MIDI cross-correlation over the full
duration: 417/445 groups have their global maximum within 0–70 ms after note-on;
per-class best near-zero lags are drums 12–23 ms, bass 23–58 ms, other 23–46 ms
(instrument attack + ~1 frame estimator bias, calibrated at 11.6 ms on synthetic
impulses). The other 28 groups (22 drums, 3 bass, 3 other) have their global maximum
outside that window, at a one-beat multiple on periodic parts; for 20 of them the
near-zero peak is within 10% of the global one, for 8 it is at 79–90%. A separate
set of 6 dense "other" groups has correlation below 0.3 throughout (5 of the 6 have
near-zero global maxima), where the estimate is uninformative. For those groups the
method cannot exclude an offset, although none shows a consistent shift; the
manuscript states it that way. Items 1–4 are therefore done as far as the evidence allows. The manuscripts were updated to
match. Nothing in the paper depends on the project's S3 bucket.

Later the same day the structure evidence was recovered too: the raw predictions of
all gate arms (S3 `gates/` and aleph0) plus the scored JSONs are committed under
`paper/evidence/structure/`, and `eval/structure_class_cis.py` recomputes every
Appendix C interval from them with full fold coverage, the scored artifacts matching
per track (ST-v4 buildup −0.2405 [−0.308, −0.175], end −0.355; ST-v3 cooldown +0.233,
buildup −0.158; ST-v1 beat-F −0.054/−0.058, boundary −0.146/−0.206 on folds 2/1). The
raw YourMT3+ predictions are committed compressed so the T2b rescore runs from a fresh
checkout once the Slakh test MIDI is acquired.

## Main scientific judgment

The strongest existing comparison holds SCNet fixed and changes the accompaniment
transcriber: basic-pitch 0.4733 versus YourMT3+ 0.7877 mean onset-and-pitch F1.
The reference-input comparison (0.4897 versus 0.8488) supports the same ordering.
This is a useful engineering case study, with a large reported effect, but it is
not a new transcription architecture or a demonstration that source separation helps
YourMT3+. That requires the same checkpoint on the unseparated mixture.

The draft now states the input grouping, matching tolerances, macro aggregation,
checkpoint names, baseline asymmetries, test reuse, and unavailable provenance. Key,
bass, tempo, drums, and structure are secondary appendix material. The existing
open-vocabulary and two-pass experiments remain in the historical ledger; they answer
different questions and are not needed for this paper's main claim.

## Claim audit

| Previous assertion | Finding and evidence | Revision |
|---|---|---|
| The madmom score rises by 0.087 from subset selection alone | Impossible under the stated same-prediction, same-label, nonnegative-score assumptions: `604 * .746 / 567 = .7946808 < .8328`. The excluded tracks would need a negative total score. Even rounding .746 to three decimals cannot bridge the gap. | Withdraw the causal claim and the comparison with eight years of field progress; retain the bound in Appendix B. |
| The published .746 and local .8328 describe the same model | The [2018 paper](https://ismir2018.ircam.fr/doc/pdfs/7_Paper.pdf), Section 4/Table 2, reports a selected single AllConv model. The [madmom default processor](https://madmom.readthedocs.io/en/v0.16/modules/features/key.html) uses an ensemble. Exact checkpoints and annotations also require comparison. | No cross-paper ranking or calibration estimate. |
| The historical key scores are unambiguously the official MIREX score | `eval/stats_significance.py:mirex` awards .5 to both +5 and +7 semitones in the same mode. The 2018 paper also permits both directions; this is **not itself a discrepancy with that paper**. `mir_eval.key.weighted_score` 0.8.2 permits +7 only. | Label archived scores “legacy symmetric-fifth”; offer explicitly versioned mir_eval rescoring. Confirm the 2026 organizer implementation rather than assuming it. |
| K10 achieves parity/non-inferiority | The reported paired interval, -.0007 [-.0187, +.0182], crosses zero. It establishes neither equivalence nor non-inferiority; no justified margin was specified. | Say superiority was not established; report exploratory uncertainty. |
| A final one-shot test makes development benchmark-blind | K10's ledger motivation uses prior GiantSteps Key errors and oracle analysis to choose a new base model; an older mode model was also fit on GiantSteps Key. Track-ID disjointness of final training examples does not erase that history. | Disclose development reuse and correct the MIREX candidate README. |
| Slakh bass MIDI follows a universal written-pitch convention | The evidence is a small empirical preview audit. The orchestrator applies +12; the evaluator can add another shift. Neither the MIDI format nor bass notation establishes what a particular renderer's audio should sound like. | Bass is in the appendix, n=143 reference pairs, with a pitch-audit requirement. |
| Reference stems define an oracle ceiling, exceeded by the cascade | A weaker model's reference-input score is not an upper bound on another model. | Describe the four operating configurations and their arithmetic differences. |
| SCNet's drum SI-SDR improvement causes an onset-F1 loss | The means move in opposite directions; no paired interval or mechanism test is available. | Descriptive observation only; transient loss is one hypothesis, not a finding. |
| Separator families can be ranked from all ledger rows | HT-Demucs-ft has 50 tracks; BS-RoFormer 47, with 104 unmatched outputs caused by naming. Dropping two heads of a six-stem model is an integration bug. | No full-split/subset ranking; retain qualifications in Appendix A. |
| The contribution establishes EDM mix-to-MIDI performance | Slakh is synthetic instrumental audio; no adequate real-EDM/vocal evaluation is available. YourMT3+ was trained on Slakh training data. | Put Slakh in the title and limit the inference to that domain/configuration. |
| Failed structure gates prove label noise is the bottleneck | The interventions change weighting/readouts, not annotation quality experimentally; fold 2 was reused, seed variance is missing. Coverage is a segment-then-track mean, not corpus-duration accuracy or precision. | Appendix C states failed criteria without a causal explanation or general impossibility claim. |
| Every result and interval regenerates from public code | Code references ignored/private per-track files; the figure generator contained hand-entered summaries. Checkpoints and environments are incompletely pinned. | Commit a labeled summary snapshot and strict archive checker; state the recovery gap explicitly. |
| Everything learned is MIT licensed and unrestricted | Repository code licensing does not establish checkpoint, dataset, or audio rights; some external artifacts have distinct or missing terms. | Remove blanket claims. Review each artifact's actual terms before redistribution. |

## What was and was not verified

Verified by reading implementation and primary sources: group construction; onset/pitch
matching; drum-class aggregation; SI-SDR formula and signal preparation; key fifth
convention; public literature's model descriptions; venue dates/rules; arithmetic
consistency of the reported principal contrasts. The revised bibliography omits unused
citations and adds prior cascade work, YourMT3+ context, and recent lightweight work.

The principal numbers trace to ledger runs T1, T2b, S4, and T10 and the historical
`STATS.md`. The snapshot records those sources and the audited commit. It deliberately
does not contain invented per-track observations. Rendering that snapshot is a build,
not an experimental reproduction.

The raw `eval/data/results_aws/` files are absent from the checkout and its history,
and no GitHub release provides them. An anonymous request to the documented project
S3 bucket was denied (403). An authorized archive export is needed; fetching dataset
audio again would not recover historical predictions or run configurations.

## Required before submitting the empirical preprint

1. **[Done 2026-09-08] Recover the four primary result archives** listed in `results_snapshot.json`,
   plus their source MIDI predictions, reference manifests, and acquisition logs.
   Run `python paper/verify_results.py --data-dir eval/data --output eval/data/publication_verified.json`.
   All four arms must contain the same 151 eligible accompaniment IDs, with no
   duplicate rows, missing predictions, or unexplained failures. Compare newly
   computed paired intervals with the archival summaries; investigate differences.
2. **[Done 2026-09-08 for T2b, the only arm with archived predictions: all 294 rows reproduced exactly; see paper/rescore_t2b.py] Rescore from MIDI**, independently of stored `note_f` fields. Use the scorer
   in `eval/evaluate_transcription.py` with the exact grouped references. Confirm
   onset=50 ms, pitch=50 cents, offsets disabled, no beat quantization, and one-to-one
   matching. Publish per-track P/R/F, note counts, group eligibility, and failures.
   The new archive checker cannot establish this step by itself.
3. **[Done 2026-09-08 to the extent recoverable; PROVENANCE.md lists verified hashes and the open items] Pin provenance.** Record source-audio/annotation hashes, exact track lists and
   split identities, checkpoint SHA-256 values, model/configuration revisions,
   package lockfiles, run commands, preprocessing and note-decoding parameters.
   Verify the babyslakh development IDs and source-MIDI overlap, not just filenames.
   Do not equate subprocess isolation with an immutable environment.
4. **[Done 2026-09-08 on aleph0 against the S7-built references; paper/audit_grouped_refs.py, paper/evidence/grouped_reference_audit.json] Audit grouped references.** `acquire_slakh.py` can skip unreadable audio stems
   independently of MIDI merging and truncates group audio to its shortest stem.
   Its default WAV write subtype can clip sums outside the PCM range. These are
   concrete risks to inspect in the recovered data, not observed corruption claims.
   Log completeness, durations, peak amplitudes, and audio/MIDI alignment per group.
5. **[Done for score archives via repository commit; DOI deposit of larger artifacts pending] Release the evidence** with a durable version and checksums, excluding audio
   that cannot be redistributed. Replace the draft's pending-archive statement with
   the actual release URL only after it exists. Correct numbers if recomputation
   changes them; regenerate both PDFs.
6. **[Open, author] Human author review.** Read the complete revised manuscript, validate the
   recovered evidence, confirm the author identity and disclosure, and approve the
   title. This AI-assisted revision does not represent a completed human sign-off.

## Highest-value new experiments for an archival paper

Use a fixed development split and a fresh evaluation corpus before making a stronger
claim. Prioritize (1) direct-mixture YourMT3+ with the same checkpoint, defined
instrument-to-group mapping and paired IDs; (2) a current lightweight comparator;
(3) paired intervals for separator/transcriber contrasts, offsets and instrument-aware
metrics, and measured quality/latency/cost. A real-audio evaluation then tests domain
transfer. Bass shifted/unshifted sensitivity, multiple training seeds for locally
trained models, and an editing study answer separate secondary questions. Do not
rebrand another run on an already-used test split as an untouched confirmation.

## Publication consequences

See [VENUE.md](VENUE.md) for the proposed title, author text, arXiv categories and
verified deadlines. The LBD demo is the near-term dissemination path. The arXiv
research draft requires the evidence-recovery steps above. MIREX key eligibility
requires organizer clarification with the full development history disclosed; the
current package must not be represented as eligible merely because IDs are disjoint.

## Revision validation

The revised research PDF has ten pages (nine at the audit, one more after the evidence-recovery text); the LBD PDF has two pages including
references and AI disclosure. Both were compiled with TeX Live 2023 and visually
inspected. The arXiv source bundle was also compiled in a clean directory and its
extracted text matched the repository PDF. Final logs contain no LaTeX warnings,
undefined citations/references, or overfull/underfull boxes.

Ten integrity regressions pass (as of the 2026-09-08 recovery): missing-archive
refusal, mismatched IDs and support, invalid-score rejection, duplicate rows in both
documented schemas, paired-bootstrap behavior, the historical key scorer against
mir_eval 0.8.2 for all 576 major/minor key pairs (exactly 24 differ, all same-mode
+5-semitone estimates), validation of every published snapshot section, archive-level
recomputation of the bass-reference and separator sections (an internally consistent
snapshot edit is rejected), required contrast sections, and rejection of non-finite
or incomplete SI-SDR archives and of archives whose digests differ from the recorded
ones. These tests use an isolated numerical environment; they are not
production-worker verification. `python paper/verify_results.py` now selects
`paper/evidence` automatically and succeeds: all primary archives are committed
(see the status update above); the missing-archive refusal is exercised by the tests.

The project name is disambiguated from the existing
[JAMS annotation library](https://github.com/marl/jams) in both manuscripts.
