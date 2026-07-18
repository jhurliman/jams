# ES2 synthetic D&B stem generator — RUNBOOK

Offline, ship-clean generator that renders broad drum & bass multitracks (drums / bass / other /
vocals premaster buses + premaster mix + mastered mix) as SCNet fine-tune training data. Implements
the pre-registered ES2 spec (`paper/EXPERIMENTS.md`). **This is dataset-generation tooling** — the
rendered audio is NOT committed (see `.gitignore`); only the code + manifests + dataset card ship.

## Toolchain + license ledger (all ship-clean: MIT / GPL+output-grant / CC0 / CC-BY)

| Layer | Tool / asset | License | Evidence |
|---|---|---|---|
| Render engine | DawDreamer 0.8.x | MIT | prebuilt arm64 cp312 wheel |
| Bass + synths (subtractive/wavetable/FM/physical) | Surge XT 1.3.4 VST3 | GPL-3 **+ output grant** | surge-synthesizer.github.io/faq/ — driven by *procedural params from init*, no factory-preset content copied |
| FM synths | Dexed 1.0.1 VST3 | GPL-3 | own procedural FM patches (no cartridge content) |
| Wavetable synths | Vitalium VST3 (DISTRHO build of Vital, `NO_AUTH=1`) | GPL-3 | content-free (no factory presets/wavetables); own procedural params. Built from source: `build_vitalium.sh` |
| Real drum one-shots (all sub-styles) | E-GMD (Roland TD-17) | **CC-BY 4.0** | magenta.tensorflow.org/datasets/e-gmd — sliced to isolated GM-keyed one-shots via aligned MIDI, re-sequenced (no E-GMD grooves reproduced) |
| Real drum one-shots (electronic) | TidalCycles TR-808 (sounds-tr808-fischer) | **CC0-1.0** | github.com/tidalcycles/sounds-tr808-fischer (LICENSE checked in) |
| Drum DSP kit | our numpy synthesis | MIT / ours | 100% original |
| Loudness | pyloudnorm | MIT | — |

Excluded on license grounds: any copyrighted "Amen" recording; Cymatics packs (their EULA bars
redistributing isolated sounds — a separation stem *is* the sample); anything NonCommercial/unclear.

## Environment (M2 Max / macOS / arm64)

The generator needs DawDreamer + Surge/Dexed VST3s, which don't co-resolve with jams' py3.14 env,
so it runs in a dedicated **Python 3.12** venv (same pattern as the ES2 probe):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python dawdreamer numpy scipy soundfile pyloudnorm mido
# Surge XT + Dexed VST3s installed to ~/Library/Audio/Plug-Ins/VST3/ (no-sudo dmg extract; see probe)
```

Vitalium is built + installed via `build_vitalium.sh` (arm64 VST3, no sudo). Paths are
env-overridable: `SYNTH_SURGE_VST3`, `SYNTH_DEXED_VST3`, `SYNTH_VITALIUM_VST3`, `SYNTH_STEMS_WORKER`,
`SYNTH_WT_BANK` (CC0 wavetable bank), `SYNTH_PRESET_SRC` (license-vetted `.vital` preset seeds). If a
plugin *or* a bank/preset source is absent, the renderer degrades gracefully (Surge covers all roles;
the CC0 scan-synth and preset seeding simply drop out). The wavetable/preset sources are cloned into
a non-shipped staging dir (git-ignored); only rendered audio + code + manifest ship — see the
DATASET_CARD provenance manifest for repos, licenses, and pinned commit SHAs.

## Reproduce

```bash
cd eval
VENV=/path/to/.venv/bin/python

# 1. Build the CC0/CC-BY one-shot library from E-GMD (once).
$VENV -m synth.oneshots --manifest /path/to/egmd/manifest.jsonl --out-dir corpus/assets

# 2. Render the ~500-track corpus (deterministic per seed; resumable; parallel).
$VENV -m synth.corpus --out-dir corpus --n 500 --workers 8 \
      --oneshots corpus/assets/oneshots.pkl \
      --kit-808 /path/to/sounds-tr808-fischer-main --seed-base 20260715

# 3. Validate: stem-sum residual + shipped-SCNet SI-SDR realism distribution.
$VENV -m synth.validate --corpus corpus --n-per-sub 3 --out corpus/validation_report.json
```

Outputs under `corpus/`: `audio/<track_id>/{drums,bass,other,vocals,mix_premaster,mix_master}.flac`
+ `track.json`; `manifest.jsonl`; `split.json` (frozen ES2-synth-val, sha256'd); `corpus_summary.json`;
`validation_report.json`.

## Module map (`eval/synth/`)

- `config.py` — sub-style specs + per-track diversity sampling (`TrackSpec`).
- `theory.py` — keys / scales / chord progressions.
- `arrange.py` — arrangement timeline (two-drop grammar, per-role intensity, harmony, bus balance).
- `patches.py` — **procedural Surge patch randomization** (osc engine / filter / waveshaper / env).
- `surge.py` / `dexed.py` / `vitalium.py` — Surge XT + Dexed (FM) + Vitalium (wavetable) drivers.
- `wavetable.py` — **CC0 wavetable scan-synth** (numpy band-limited oscillator over public-domain
  `.vitaltable` banks; the #1 timbre lever). Bank env-overridable via `SYNTH_WT_BANK`.
- `presets.py` — **preset bank** (license-vetted `.vital` seeds: scalar overlay + raw-JSON loader for
  full fidelity; author spot-check; audio-only GPL position). Source env-overridable `SYNTH_PRESET_SRC`.
- `vital_state.py` — **full-fidelity `.vital` loader**: JUCE-wraps a (jittered) preset JSON into a
  Vitalium VST3 state chunk and `load_state`s it, so the preset's real embedded **wavetable** + all
  params render through the unmodified GPL plugin. Supersedes scalar-only seeding.
- `bass.py` — bass families (sub / reese / wobble / jumpup-bounce / foghorn / growl).
- `synths.py` — `other` bus (pad / rhodes / stab / lead / pluck / atmos).
- `oneshots.py` — E-GMD + TR-808 one-shot library extraction + loading.
- `drums.py` — D&B drum grammar: DSP kit + real one-shots, per-hit jitter, break layer.
- `fx.py` — per-stem bus FX (kick-keyed sidechain, saturation, convolution reverb+width, glue).
- `mixmaster.py` — sidechain envelope + master chain (comp+limiter to genre LUFS).
- `render.py` — per-track render orchestration (4 buses -> premaster -> master).
- `corpus.py` — batch driver + manifest + frozen split.
- `validate.py` — stem-sum + SCNet SI-SDR realism.

## musdb-XL target definition (binding)

GT stems are the **premaster** buses; they sum to the premaster mix **exactly** (float residual
≈ −215 dB; a 16-bit-FLAC quantization floor on disk). All per-stem processing (sidechain duck,
saturation, reverb, width) is baked into each stem *before* the linear sum. The master chain
(comp + brickwall limiter to the sub-style LUFS target, −8…−5) runs on the **mix only**; each
track records its premaster→master gain ratio so stems stay reconstructible. Training against
limited-master-summing stems is prohibited.
