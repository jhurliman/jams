# Statistical analysis — key detection (GiantSteps Key)

n = 567 tracks; MIREX weighted score; bootstrap 10,000 resamples, seed 0; 95% percentile CIs. Published honest SOTA reference: KeyMyna 0.7591 weighted.

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

| comparison | Δ weighted [95% CI] | significant |
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

**Fusion vs published SOTA value:** fusion CI [0.7831, 0.8402] excludes the best honest published number (0.7591).

**Subset-shift calibration (key finding):** madmom's CNN, published at 0.746 on full GiantSteps Key, scores 0.8328 on our n=567 usable-track subset — a +0.087 shift from subset selection alone. Comparisons of numbers measured on this subset against published full-set numbers are therefore inflated for every system; only the same-subset paired comparisons above are valid rankings. On those, madmom-cnn and k10-cnn (ours, MIT; ledger K10) are statistically indistinguishable (Δ −0.0007-scale), with k10-cnn holding the best exact accuracy; madmom's weights are CC BY-NC-SA (non-commercial), the k10/fusion/skey stack carries no such restriction.

# Statistical analysis — transcription (Slakh2100-redux test)

Paired per-track note-F (onset+pitch, 50 ms/50 c, offsets ignored), oracle (ground-truth) stems. YourMT3+ scored against the same Slakh GT with the same scoring functions as basic-pitch; paired bootstrap 10,000 resamples, seed 0.

| stem | basic-pitch [95% CI] | YourMT3+ [95% CI] | Δ paired [95% CI] | YourMT3+ wins |
|---|---|---|---|---|
| bass (n=143) | 0.7889 [0.7624, 0.8132] | 0.8486 [0.7939, 0.8983] | +0.0597 [+0.0034, +0.1100] | 87% |
| other (n=151) | 0.4897 [0.4742, 0.5055] | 0.8488 [0.8369, 0.8607] | +0.3591 [+0.3471, +0.3713] | 100% |

Both deltas are paired per-track (same 151 oracle stems for both systems); a CI excluding zero is a significant difference. Bass scores use the +12 written-pitch convention for both systems (see ledger T-entries).
