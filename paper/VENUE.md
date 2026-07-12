# Publication venue plan

## Timeline reality (checked Jul 3, 2026)

- **ISMIR 2026** (Abu Dhabi + online, **Nov 8–12, 2026**): main-track deadline has passed
  (abstracts Apr 20, papers Apr 27; notifications Jul 10). Not an option this cycle.
- **ISMIR 2026 Late-Breaking/Demo (LBD)**: call **not yet posted** (as of Jul 3);
  historically opens mid-summer with a deadline in **Aug–Sep**. 4 pages, typically
  **non-archival and non-anonymous** — ideal fit: early/system results allowed, demo
  encouraged (the web annotator + live analysis is a strong demo). **Primary conference
  target.** Action: watch https://ismir2026.ismir.net/ for the CfLBD.
- **arXiv preprint**: no deadline; establishes priority and is citable. **Do first**, once
  the structure results land (with or without them if they slip — the key + transcription
  results stand alone).
- **Blog post** (jhurliman.org or repo docs): adapted from the preprint; developer-facing
  framing (the contamination story + "auditable SOTA" theme travel well). Publish
  alongside arXiv.
- **ISMIR 2027 full paper**: the archival path if structure results + a MedleyDB vocals
  benchmark + seed-variance runs round out the story. Deadline ~Apr 2027.

## Sequencing

1. ~~Structure fine-tune completes → gate-check~~ **DONE (Jul 12)** — negative-results
   arc v1→v4, all gates FAIL, stock ships; ledgered with per-class CIs.
2. ~~madmom baseline row~~ **DONE** (K9 + calibration finding). MedleyDB vocals still
   blocked on access — documented limitation in the draft.
3. ~~Freeze results; figures~~ **DONE** — `paper/arxiv/make_figures.py` (4 figures).
4. **arXiv preprint: full draft written (Jul 12)** — `paper/arxiv/main.tex`, 14 pp
   compiled, number-audited against the ledger, bibliography web-verified,
   hostile-review pass applied. Remaining before upload: author list/acknowledgments
   (John), title sign-off, arXiv account/endorsement, then blog post derived from it.
5. **MIREX 2026** (new): Audio Key Detection / Beat Tracking / Music Structure Analysis
   tasks; submissions Jul 1–Oct 1, results Oct 15, top performers present at ISMIR LBD.
   Independent hidden-test-set validation of exactly our calibration finding — key
   system is the prime candidate; needs 2–4 pp extended abstract + packaging.
6. ISMIR 2026 LBD submission when the call opens (~Aug; condense preprint to 4 pp; demo
   video of the annotator + piano-roll).
7. Reassess ISMIR 2027 full paper after LBD feedback.

## Open/licensing posture (publication-relevant)

- Repo public; all eval scripts + per-track artifacts publishable (Slakh CC-BY-4.0 rows,
  GiantSteps annotations public; audio not redistributed — acquire scripts fetch).
- Shipped models: key_fusion.json (ours, trainable from public data), S-KEY (MIT),
  YourMT3 weights (Apache-2.0 via MIT mt3-infer; GPL upstream avoided), SCNet ckpt
  (MSST zoo, MIT code), ADTOF-pytorch (no declared license — subprocess-isolated, note in
  paper), madmom weights (CC BY-NC-SA — **evaluation-only**, never shipped).
- Author list / acknowledgments: TBD by John.
