# Manuscripts and evidence

The current research draft is [arxiv/main.pdf](arxiv/main.pdf), **Evaluating a Modular
Mix-to-MIDI Pipeline on Slakh2100**. The companion [LBD abstract](lbd/main.pdf) focuses
on the inspection interface. [REVIEW.md](REVIEW.md) explains the scientific corrections
and missing evidence; [VENUE.md](VENUE.md) records metadata decisions and current deadlines.
`DRAFT.md` and `EXPERIMENTS.md` are historical records, not current submission text.

## Build

From the repository root, with Python/matplotlib and a TeX installation containing
`latexmk`, `pdflatex`, and BibTeX:

```sh
python paper/verify_results.py
python paper/arxiv/make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/arxiv/main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/lbd/main.tex
```

The first command validates the summary snapshot and, because its status is
`stored_scores_recomputed`, automatically checks it against the committed archives in
`paper/evidence/` (digests, track sets and supports, means, intervals, paired contrasts,
bass reference, separator table); it fails on any mismatch. The second generates two
TeX tables and one PDF figure from `results_snapshot.json` (its `verification` block
records the recomputation). Neither command reruns an experiment. The superseded key
forest/calibration figure is removed so it cannot silently return in a submission.
The LBD manuscript fits within the 2026 limit of two scientific pages plus one
optional references/acknowledgments/AI page. Inspect page layout after every edit.

The review build used Python 3.12, matplotlib 3.10.8, and TeX Live 2023/Debian. These
are document-build tools, not the historical model-inference environment. Numerical
integrity checks additionally need numpy and mir_eval==0.8.2:

```sh
python -m unittest discover -s paper -p 'test_publication_checks.py' -v
```

## Verify the main results

The per-recording score archives are committed under `paper/evidence/results_aws/`
(SHA-256 sums in `paper/evidence/SHA256SUMS`; recovered from the author machine on
2026-09-08). They are:

- `results_aws/slakh_test_oracle.json` (T1, basic-pitch reference input)
- `results_aws/yourmt3_oracle_per_track.json` (T2b, YourMT3+ reference input)
- `results_aws/sep_scnet_notes.json` (S4, SCNet + basic-pitch)
- `results_aws/slakh_test_e2e_scnet_yourmt3.json` (T10, SCNet + YourMT3+)

Then run:

```sh
python paper/verify_results.py --data-dir paper/evidence --output paper/evidence/publication_verified.json
```

Missing archives, duplicate IDs, invalid scores, unmatched pairs, incorrect support,
and means inconsistent with four-decimal reported values fail verification. The
checker accepts the nested `per_track[].stems.other.note_f` evaluator schema and the
flat `per_track[].{track_id,stem,note_f}` YourMT3+ schema; it does not guess other formats.
It reports source-file SHA-256 values, sorted IDs, and paired 10,000-resample percentile
intervals (seed 0). Historical bootstrap row order is not known, so finite-bootstrap
endpoints can differ slightly even when data agree. Inspect those differences before
updating the manuscript; nothing is overwritten automatically.

This checks **stored scores**, not MIDI matching or model inference. For the one arm
with archived predictions (T2b), `python paper/rescore_t2b.py` rebuilds the grouped
Slakh references from the redux test MIDI (expected under `eval/data/slakh_home/`; see
the script header) and reproduces every archived per-track value; its output is
committed as `paper/evidence/rescore_t2b.json`. `paper/PROVENANCE.md` records
checkpoint hashes, versions, and the unrecoverable items. `paper/audit_grouped_refs.py`
audits the grouped references (completeness, truncation, clipping, onset/MIDI alignment)
where the redux and grouped files live; its report is `paper/evidence/grouped_reference_audit.json`. Recovery of
predictions, references, run configurations and checkpoint identities, independent
rescoring, and a durable evidence release remain necessary; see the ordered steps in
[REVIEW.md](REVIEW.md). Dataset-acquisition scripts do not recover historical predictions.

The broader historical `eval/stats_significance.py` requires additional key features,
training manifests and model dependencies. It now requires a new `--out` path, labels
the key convention, and cannot overwrite `STATS.md`. Its optional
`--key-metric mir-eval-0.8.2` mode requires that exact version; no scores have been
silently changed to the alternate convention.

## Prepare the arXiv source bundle

After the build and the scientific review, run from the repository root:

```sh
mkdir -p build
tar -czf build/jams-arxiv-source.tar.gz -C paper/arxiv main.tex main.bbl references.bib tables/transcription.tex tables/separation.tex figs/fig_transcription.pdf
```

This explicit file list excludes logs, prior figures, the demo, and stale manuscript
sources. It includes the generated bibliography for arXiv. Extract into a clean
directory and compile `main.tex` there before upload. A successful source build does
not certify the missing research evidence or human author approval.
