"""Bass design families via Surge XT + numpy motion. Always >=2 layers (mono sub + hp'd mid).

Families: sub (sine <100 Hz), reese (detuned saws + slow filter sweep), wobble (synced filter
LFO), jumpup_bounce (pitch/filter-enveloped mid), foghorn (PWM/wavetable-ish drone), growl
(driven reese + faster formant motion). Surge renders the static oscillator/filter tone; the
family-specific *movement* (LFO/sweep/formant) is applied in numpy for reliable, controllable
modulation (Surge's mod matrix isn't addressable through raw automation IDs).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from . import arrange, patches
from . import presets as _presets
from . import vital_state as _vs
from . import wavetable as _wt
from .surge import render_layer

SR = 44100
_PRESET_SEED_PROB = 0.4
# Probability a wavetable-family mid-bass is rendered by the CC0 scan-synth (the neuro-bass lever).
# Tuned to hold the SCNet-realism bass SI-SDR median at ~8 dB (matched A/B; see DATASET_CARD).
_WT_BASS_PROB = 0.33


def _hp(x: np.ndarray, f: float) -> np.ndarray:
    return sosfilt(butter(4, f / (SR / 2), btype="high", output="sos"), x)


def _lp(x: np.ndarray, f: float) -> np.ndarray:
    return sosfilt(butter(4, min(f, SR / 2 - 100) / (SR / 2), btype="low", output="sos"), x)


def _lfo(n: int, rate_hz: float, phase: float = 0.0, shape: str = "sine") -> np.ndarray:
    t = np.arange(n) / SR
    if shape == "tri":
        return 2 * np.abs(2 * ((t * rate_hz + phase) % 1.0) - 1) - 1
    return np.sin(2 * np.pi * (rate_hz * t + phase))


def _sweep_filter(x: np.ndarray, rate_hz: float, lo_hz: float, hi_hz: float,
                  shape: str = "sine") -> np.ndarray:
    """Crossfade a dark and bright low-passed copy with an LFO (efficient synced motion)."""
    dark = _lp(x, lo_hz)
    bright = _lp(x, hi_hz)
    lfo = 0.5 * (_lfo(x.shape[-1], rate_hz, shape=shape) + 1.0)
    return dark * (1 - lfo) + bright * lfo


# --- Note generators ---------------------------------------------------------
def _sub_notes(spec, tl: arrange.Timeline, rng) -> list:
    out = []
    for bar in range(tl.total_bars):
        if tl.role_intensity("sub", bar) < 0.12:
            continue
        r = _root(spec, tl, bar, octave=24)              # deep sub, an octave below the kick
        for step, dur in [(0, 6), (8, 4)] if spec.substyle != "jungle" else [(0, 8)]:
            out.append((r, 110, arrange.swing_time(tl, bar, step, 0.0), dur * tl.step * 0.95))
    return out


def _mid_notes(spec, tl: arrange.Timeline, rng, family: str) -> list:
    out = []
    for bar in range(tl.total_bars):
        if tl.role_intensity("midbass", bar) < 0.12:
            continue
        r = _root(spec, tl, bar, octave=36)  # one octave up
        if family in ("reese", "wobble", "foghorn", "growl"):
            # sustained/legato — the movement comes from the filter modulation
            length = (2 if family == "foghorn" else 1) * tl.bar
            out.append((r, 100, tl.bar_start(bar), length * 0.98))
        else:  # jumpup_bounce: syncopated octave-hopping 16ths
            steps = [0, 3, 6, 8, 10, 11, 14]
            for st in steps:
                p = r + (12 if st in (6, 11) else 0)
                out.append((p, 104, arrange.swing_time(tl, bar, st, spec.swing),
                            tl.step * 1.3))
    return out


def _root(spec, tl: arrange.Timeline, bar: int, octave: int) -> int:
    from . import theory
    return theory.bass_root(spec.key, tl.degree_at(bar), octave)


def _modulate(x: np.ndarray, family: str, tl: arrange.Timeline, rng) -> np.ndarray:
    beat_hz = tl.bpm / 60.0
    if family == "wobble":
        rate = beat_hz / rng.choice([1, 2, 4])  # 1/4, 1/8, 1/16 wobble
        return _sweep_filter(x, rate, 180, 2600, shape="tri")
    if family == "reese":
        return _sweep_filter(x, beat_hz / 16.0, 500, 2200)  # slow evolving sweep
    if family == "growl":
        y = _sweep_filter(x, beat_hz / 2.0, 300, 2400, shape="tri")
        return np.tanh(2.2 * y)  # extra drive/formant grit
    if family == "foghorn":
        return _sweep_filter(x, beat_hz / 32.0, 250, 1400)  # very slow PWM-ish drift
    return x  # jumpup_bounce: tone is already plucky from the filter env


# Mid-bass families that can also come from a wavetable engine (Vitalium or the CC0 scan-synth).
# The CC0 bank's folding / FM / sync / phase-distortion tables are the Reese/growl/neuro fuel.
_VIT_BASS = {"reese", "wobble", "growl"}
_WT_BASS = {"reese", "wobble", "growl", "foghorn"}


def render_bass_bus(spec, tl: arrange.Timeline, rng,
                    vitalium=None) -> tuple[np.ndarray, list[dict]]:
    secs = tl.total_secs
    n = tl.total_samples
    bus = np.zeros((2, n))
    descriptors: list[dict] = []
    vit_ok = vitalium is not None and vitalium.available()
    cc0wt_ok = _wt.available()
    preset_ok = _presets.available()

    def fit(a: np.ndarray) -> np.ndarray:
        if a.shape[1] < n:
            a = np.pad(a, ((0, 0), (0, n - a.shape[1])))
        return a[:, :n]

    mids = np.zeros((2, n))
    for family in spec.bass_families:
        cfg, desc = patches.rand_bass_cfg(family, rng)
        if family == "sub":
            audio = fit(render_layer(cfg, _sub_notes(spec, tl, rng), secs, "sub"))
            audio = _lp(audio, 140)                       # sub owns the lows
            env = arrange.env_from_intensity(tl, "sub")
            audio = audio * env[None, :]
            bus += arrange.balance_to(audio, -9.5)        # sub is the dominant low-end voice
            descriptors.append({"family": family, **desc})
            continue
        notes = _mid_notes(spec, tl, rng, family)
        roll = rng.random()
        if cc0wt_ok and family in _WT_BASS and roll < _WT_BASS_PROB:
            # CC0 wavetable scan-synth — real public-domain growl/neuro tables.
            audio = fit(_wt.render(notes, secs, rng, family))
            desc = {"engine": "cc0-wavetable"}
        elif vit_ok and family in _VIT_BASS and roll < 0.6:
            seed = _presets.pick(rng) if (preset_ok and rng.random() < _PRESET_SEED_PROB) else None
            pj = _presets.load_json(seed) if (seed and _vs.available()) else None
            if pj is not None:
                audio = fit(_vs.render(notes, secs, rng, family, pj))
                desc = {"engine": "vitalium-fullstate", "preset_seeded": True,
                        "preset": seed.get("_name"), "preset_license": seed.get("_license")}
            else:
                audio = fit(vitalium.render(notes, secs, rng, family, seed=seed))
                desc = {"engine": "vitalium-wt", "preset_seeded": bool(seed),
                        "preset": (seed or {}).get("_name")}
        else:
            audio = fit(render_layer(cfg, notes, secs, family))
        audio = _modulate(audio, family, tl, rng)
        audio = _hp(audio, 110)                            # mid bass HP so the sub owns < ~100
        env = arrange.env_from_intensity(tl, "midbass")
        audio = audio * env[None, :]
        if np.abs(audio).max() > 1e-6:
            # Balance each family individually so a quiet random patch never drowns (and the
            # designed mid-bass stays clearly audible ~2.5 dB under the sub).
            mids += arrange.balance_to(audio, -12.0, cap=12.0)
        descriptors.append({"family": family, **desc})
    bus += mids
    return bus, descriptors
