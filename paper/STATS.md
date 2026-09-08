> Historical aggregate snapshot. Interpretive corrections added 2026-09-08; numeric rows have not been independently recomputed. See [REVIEW.md](REVIEW.md) and [results_snapshot.json](results_snapshot.json).

# Statistical analysis — key detection (GiantSteps Key)

n = 567 tracks; Legacy symmetric-fifth weighted score; bootstrap 10,000 resamples, seed 0; 95% percentile CIs. Exploratory summaries conditional on previous benchmark use; no cross-paper ranking.

## Point estimates

| system | weighted [95% CI] | exact |
|---|---|---|
| edma-raw | 0.7589 [0.7277, 0.7885] | 0.6878 |
| honest-retrain | 0.8095 [0.7799, 0.8372] | 0.7531 |
| skey | 0.8168 [0.7887, 0.8434] | 0.7478 |
| fusion | 0.8123 [0.7831, 0.8402] | 0.7566 |
| madmom-cnn | 0.8328 [0.8063, 0.8580] | 0.7725 |
| k10-cnn | 0.8321 [0.8039, 0.8586] | 0.7795 |

## Paired deltas (bootstrap CI of per-track difference)

| comparison | Δ weighted [95% CI] | CI excludes zero (unadjusted) |
|---|---|---|
| fusion − edma-raw | +0.0534 [+0.0339, +0.0735] | yes |
| fusion − honest-retrain | +0.0028 [-0.0109, +0.0169] | no |
| fusion − skey | -0.0044 [-0.0254, +0.0173] | no |
| skey − edma-raw | +0.0578 [+0.0307, +0.0855] | yes |
| honest-retrain − edma-raw | +0.0506 [+0.0300, +0.0716] | yes |
| fusion − madmom-cnn | -0.0205 [-0.0437, +0.0028] | no |
| skey − madmom-cnn | -0.0160 [-0.0390, +0.0072] | no |
| k10-cnn − fusion | +0.0198 [-0.0023, +0.0427] | no |
| k10-cnn − skey | +0.0153 [-0.0081, +0.0390] | no |
| k10-cnn − madmom-cnn | -0.0007 [-0.0187, +0.0182] | no |

**Interpretation corrected:** An interval containing zero does not establish equivalence or non-inferiority. The earlier subset-only explanation is withdrawn: `604 * .746 / 567 = .7947` is the maximum possible nonnegative subset mean under unchanged predictions, labels and scoring, below .8328. The published selected single model also differs from madmom's default ensemble. Legacy scoring gives fifth credit in both directions, as in the 2018 paper, while mir_eval 0.8.2 gives +7-only credit. No published SOTA comparison or unrestricted-weight-license conclusion follows from this table.

# Statistical analysis — transcription (Slakh2100-redux test)

Paired per-track note-F (onset+pitch, 50 ms/50 c, offsets ignored), oracle (ground-truth) stems. YourMT3+ scored against the same Slakh GT with the same scoring functions as basic-pitch; paired bootstrap 10,000 resamples, seed 0.

| stem | basic-pitch [95% CI] | YourMT3+ [95% CI] | Δ paired [95% CI] | YourMT3+ wins |
|---|---|---|---|---|
| bass (n=143) | 0.7889 [0.7624, 0.8132] | 0.8486 [0.7939, 0.8983] | +0.0597 [+0.0034, +0.1100] | 87% |
| other (n=151) | 0.4897 [0.4742, 0.5055] | 0.8488 [0.8369, 0.8607] | +0.3591 [+0.3471, +0.3713] | 100% |

Reference-input supports differ: bass n=143, other n=151. Intervals are unadjusted exploratory summaries. Bass includes an empirical +12-semitone estimate shift and monophonic filtering, requiring an audio/MIDI audit; this is not a universal MIDI convention.
