# Publication decisions and schedule

Updated 8 September 2026. The substantive audit is in [REVIEW.md](REVIEW.md).
A clean LaTeX build does not resolve the evidence gaps identified there.

## Proposed metadata, applied in this revision

| Decision | Selected wording / recommendation | Reason |
|---|---|---|
| Preprint title | **Evaluating a Modular Mix-to-MIDI Pipeline on Slakh2100** | States the actual task and dataset; avoids unsupported novelty, SOTA, and EDM-generalization claims. |
| Author | **John Hurliman**, independent researcher, `jhurliman@jhurliman.org` | Matches the existing project/PR metadata. No additional human contribution or institutional affiliation was documented. Confirm before submission. |
| Acknowledgments | Thank the authors and maintainers of the cited datasets, models, and evaluation software. | Gives credit without inventing collaborators, funding, or endorsement. |
| AI disclosure | Original drafting, implementation, and experiment assistance by Claude (Anthropic); source checking, code auditing, and substantive revision by OpenAI Codex. Human author responsible; no AI coauthor. | The involvement extends beyond proofreading and must be disclosed accurately. Complete wording is in both manuscripts. |
| arXiv primary category | **eess.AS — Audio and Speech Processing** | The narrowed contribution evaluates audio separation/transcription and MIR. |
| arXiv cross-list | **cs.SD — Sound**, if accepted by moderators | Appropriate visibility to the computing/music-audio community; classification is subject to moderation. |
| Manuscript license | Recommend arXiv's **non-exclusive distribution license** for the initial deposit | Grants arXiv distribution rights with limited reuse rights for others. Check the intended venue/funder terms; the author selects the actual license at upload. |
| LBD title | **jams: Inspecting a Modular Audio-to-MIDI Pipeline** | Demo-centered companion, with measured claims clearly distinguished from interface capabilities. |

The category recommendation follows the [arXiv taxonomy](https://arxiv.org/category_taxonomy).
License options and their permanence are explained in [arXiv license guidance](https://info.arxiv.org/help/license/index.html).
The [arXiv AI policy](https://info.arxiv.org/help/moderation/index.html) requires disclosure
of significant generative-AI involvement and human responsibility. The
[ISMIR AI policy](https://ismir2026.ismir.net/ai-usage-policy) similarly covers use in
research and analysis. Author approval of this PR can settle the title and wording;
it does not replace reading the evidence or the submission agreement. No account,
endorsement, authorship approval, upload, or acceptance has been assumed.

## Verified 2026 deadlines

| Route | Deadline / constraint | Action |
|---|---|---|
| **ISMIR Late-Breaking/Demo** | **25 September 2026, Anywhere on Earth**; rolling screening, capped at **75**, potentially closing early | Submit the revised demo after checking the working demonstration and author text. Do not wait until the final day. |
| LBD format | **2 pages scientific content + optional 1 page references, acknowledgments and AI statement**; non-anonymous | `paper/lbd/main.tex` uses the supplied 2026 LBD template. Any third page must contain only permitted matter. |
| LBD final materials | **16 October 2026** for camera-ready paper and final poster/video | Prepare the demonstration and materials; at least one author must register for in-person or online participation. |
| **MIREX submissions** | **1 October 2026**; general page does not specify a deadline time zone | Resolve key-task eligibility now. Do not assume an AoE cutoff. |
| MIREX results | **15 October 2026** | A task evaluation, not acceptance of this research manuscript. |
| arXiv | No conference deadline | Deposit after primary evidence recovery and human review; leave time for endorsement/moderation. arXiv is a preprint server, not peer review. |

Sources: [official LBD call](https://ismir2026.ismir.net/call-for-late-breaking-demo),
[MIREX 2026 schedule](https://music-ir.org/mirex/wiki/MIREX_HOME),
[arXiv moderation](https://info.arxiv.org/help/moderation/index.html).
The ISMIR 2026 main-paper route has closed. LBD is screened for suitability,
originality, and format; do not describe it as an archival main-track peer-reviewed paper.
A 2027 submission is a possible later route, not a verified deadline in this plan.

## MIREX key: material eligibility issue

The [2026 Audio Key Detection rules](https://music-ir.org/mirex/wiki/2026:Audio_Key_Detection)
prohibit GiantSteps Key use for training, validation, model selection, parameter
tuning, **or any other development**. The project ledger records prior test-set error
analysis that motivated K10 and an earlier mode model trained on GiantSteps Key.
The final K10 training IDs being disjoint, and its final checkpoint being scored once,
do not establish compliance with the broader development rule.

The inference wrapper and weights are a **candidate package**, not an eligible
submission. There is no 2–4 page task extended abstract in that package. If organizers
allow participation after full disclosure, prepare the task abstract, confirm the
scoring convention and runtime requirements, and validate on their prescribed platform.
Do not promise “independent hidden-set validation”: the task identifies GiantSteps
Key, and the actual evaluation conditions belong to the organizers.

Draft inquiry for the author to review and send (not sent by this revision):

> We are considering submitting K10, a small global-key CNN, to MIREX 2026 Audio Key
> Detection. Its final training corpus is mirdata beatport_key; our ledger reports no
> track-ID overlap with GiantSteps Key, and final architecture/epoch selection used
> training-corpus cross-validation. However, earlier project models were evaluated
> on GiantSteps Key, an earlier mode classifier was trained on that dataset, and
> analysis of those test errors motivated K10. The final K10 checkpoint was then
> evaluated once on GiantSteps Key. Does this development history make K10 ineligible
> under the rule prohibiting any development use of GiantSteps Key? If so, is any
> clearly labeled non-competitive participation possible? Please also confirm the
> 2026 fifth-error scoring convention and the October 1 cutoff time zone.

Do not start another “clean” retraining merely to relabel the same development
history. If the organizers decline, pursue the LBD demo and the transcription paper.

## Submission order and concrete remaining work

1. Review the revised title, sole-author line and full AI statement; confirm whether
   any actual human contributors or funding need to be added. Do not invent them.
2. Recover and verify the four primary archives as specified in [REVIEW.md](REVIEW.md).
   Build and read both PDFs. The arXiv source package build command is in [README.md](README.md).
3. For LBD, validate the current interface on one authorized audio example, record
   a short import/inspection/MIDI-export demonstration, and verify that its
   performance claims match the submitted abstract. Avoid unsupported speed claims.
4. Submit LBD before its rolling cap/deadline; pursue the MIREX inquiry in parallel.
5. Deposit the empirical preprint when its evidence and human review are complete.
   Prepare an archival expansion only after the direct-mixture and fresh-corpus controls.

Code, checkpoint, dataset and recording licenses are distinct. Check the actual
terms of each redistributed artifact. Neither an MIT repository nor process
isolation establishes commercial rights to every dependency or model weight.
