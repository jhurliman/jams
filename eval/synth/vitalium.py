"""Vitalium (GPL-3 wavetable synth) voice — a genuine THIRD synthesis engine.

Vitalium is DISTRHO's content-free GPL-3 build of Vital (mtytel/vital, NO_AUTH=1): no factory
presets/wavetables, no login — we drive only its init state with our own parameters, so it is
ship-clean. Its wavetable + spectral-unison + wave-frame morph give spectra distinct from both
Surge (subtractive/wavetable) and Dexed (FM), widening the corpus's timbral span.

Phase-1 spike result (see the ``DATASET_CARD.md`` "Preset/wavetable ingestion" section): DawDreamer
CANNOT load a ``.vital`` preset or its embedded wavetable — ``load_preset('.vital')`` returns False
and the VST3 state chunk is opaque binary, so we can only drive the ~700 NAMED scalar params via
``set_parameter``. Two consequences wired in here:
  1. External CC0 wavetables are ingested by a separate numpy scan-synth engine (``wavetable.py``),
     NOT through Vitalium — Vitalium always plays its single built-in table.
  2. Preset "seeding" is a SCALAR overlay: license-vetted ``.vital`` ``settings`` scalars are mapped
     by name to Vitalium params (``presets.py``), set here, then band-jittered per family.

This module also fixes latent no-ops discovered in the spike: the init patch's **filter is not
routed** (Filter 1 Switch = Off) so cutoff/res/model were inert; **Osc 2 Switch = Off** so a set
osc-2 level was silent; **Distortion Type = None** so distortion amount did nothing; and wave-frame
morph is inert on the flat default table. We now enable filter routing + osc-2 + a distortion type,
so the scalar params actually shape timbre.
"""

from __future__ import annotations

import os

import dawdreamer as daw
import numpy as np

SR = 44100
BS = 512
VST = os.environ.get(
    "SYNTH_VITALIUM_VST3",
    os.path.expanduser("~/Library/Audio/Plug-Ins/VST3/Vitalium.vst3"),
)

# Parameter indices (Vitalium VST3 1.0.6, discovered via get_parameters_description + spike).
_P = {
    "vol": 462,                                          # master Volume (was 2 = Chorus Delay: bug)
    "o1_switch": 384, "o1_level": 382, "o1_frame": 397,
    "o1_uni_det": 395, "o1_uni_voices": 396,
    "o1_dist_type": 380, "o1_dist_amt": 377, "o1_transpose": 391, "o1_dest": 501,
    "o2_switch": 407, "o2_level": 405, "o2_frame": 420, "o2_transpose": 414,
    "o2_uni_voices": 419, "o2_uni_det": 418, "o2_dest": 503,
    "f1_switch": 114, "f1_input": 106, "f1_mix": 112, "f1_model": 113, "f1_style": 116,
    "f1_cut": 104, "f1_res": 115,
    "env1_a": 48, "env1_d": 50, "env1_s": 54, "env1_r": 52,
}

# Per-role Env-1 (amp) ADSR (normalized) + filter-cutoff [lo, hi] band (normalized 0..1).
_ROLE = {
    "pad": (0.45, 0.45, 0.85, 0.55, 0.40, 0.65),
    "reese_pad": (0.35, 0.45, 0.8, 0.5, 0.35, 0.6),
    "atmos": (0.6, 0.5, 0.8, 0.6, 0.30, 0.55),
    "stab": (0.02, 0.28, 0.15, 0.25, 0.5, 0.8),
    "lead": (0.05, 0.32, 0.7, 0.3, 0.55, 0.85),
    "pluck": (0.0, 0.22, 0.08, 0.22, 0.5, 0.8),
    "rhodes": (0.02, 0.4, 0.35, 0.35, 0.4, 0.7),
    "reese": (0.02, 0.4, 0.75, 0.3, 0.25, 0.5),
    "wobble": (0.03, 0.4, 0.8, 0.3, 0.3, 0.6),
    "growl": (0.02, 0.4, 0.75, 0.3, 0.25, 0.5),
}

# Roles that want grit (a distortion type + drive) — the reese/growl/neuro character.
_GRITTY = {"reese", "growl", "wobble", "reese_pad"}
# Distortion Type normalized values that need Osc 2 as a modulator source (FM/RM <- Osc).
_DIST_BEND = 0.3          # "Bend" — self-contained waveshape, safe without osc2


def available() -> bool:
    return os.path.exists(VST)


def _clampj(v, amt, rng, lo=0.0, hi=1.0):
    return float(np.clip(v + rng.uniform(-amt, amt), lo, hi))


def _patch(p, role: str, rng, seed: dict | None = None) -> dict:
    """Program a Vitalium patch. `seed` is an optional preset scalar overlay (normalized)."""
    a, d, s, r, cut_lo, cut_hi = _ROLE.get(role, _ROLE["pad"])

    def jit(v, amt=0.08):
        return _clampj(v, amt, rng)

    p.set_parameter(_P["vol"], 0.72)

    # --- Oscillator 1 (always on) ---
    p.set_parameter(_P["o1_switch"], 1.0)
    p.set_parameter(_P["o1_level"], seed.get("o1_level", 1.0) if seed else 1.0)
    # wave-frame morph is inert on the flat default table, but harmless; jitter lightly.
    frame = jit(seed.get("o1_frame", float(rng.uniform(0.0, 1.0))) if seed else
                float(rng.uniform(0.0, 1.0)), 0.15)
    p.set_parameter(_P["o1_frame"], frame)
    voices = int(seed["o1_uni_voices"]) if seed and "o1_uni_voices" in seed \
        else int(rng.integers(1, 8))
    voices = max(1, min(16, voices))
    p.set_parameter(_P["o1_uni_voices"], voices / 16.0)
    det = seed.get("o1_uni_det") if seed and "o1_uni_det" in seed else float(rng.uniform(0.1, 0.6))
    p.set_parameter(_P["o1_uni_det"], _clampj(det, 0.1, rng) if voices > 1 else 0.0)
    p.set_parameter(_P["o1_dest"], 0.0)                  # FILTER 1 (enum — set, never jitter)

    # --- Distortion (enable a type so the amount is audible) — grit for reese/growl ---
    dist_on = role in _GRITTY or rng.random() < 0.25
    if dist_on:
        p.set_parameter(_P["o1_dist_type"], _DIST_BEND)  # enum: fixed, not jittered
        p.set_parameter(_P["o1_dist_amt"], jit(float(rng.uniform(0.3, 0.7))))
    else:
        p.set_parameter(_P["o1_dist_type"], 0.0)

    # --- Oscillator 2 (must switch ON to be heard) ---
    o2 = (seed.get("o2_on") if seed else None)
    o2 = bool(rng.random() < 0.5) if o2 is None else bool(o2)
    if o2:
        p.set_parameter(_P["o2_switch"], 1.0)
        p.set_parameter(_P["o2_level"], seed.get("o2_level", float(rng.uniform(0.4, 0.9)))
                        if seed else float(rng.uniform(0.4, 0.9)))
        p.set_parameter(_P["o2_frame"], float(rng.uniform(0.0, 1.0)))
        p.set_parameter(_P["o2_transpose"], 0.5 + rng.choice([-1, 0, 1]) * (12 / 96.0))
        p.set_parameter(_P["o2_uni_voices"], int(rng.integers(1, 5)) / 16.0)
        p.set_parameter(_P["o2_dest"], 0.0)              # FILTER 1
    else:
        p.set_parameter(_P["o2_switch"], 0.0)

    # --- Filter 1: ROUTE + enable so cutoff/res/model actually shape the sound ---
    p.set_parameter(_P["f1_switch"], 1.0)
    p.set_parameter(_P["f1_input"], 1.0)
    p.set_parameter(_P["f1_mix"], 1.0)
    p.set_parameter(_P["f1_model"], float(rng.choice([0.0, 0.5])))   # Analog/Diode (enum: choose)
    if seed and "f1_cut" in seed:
        cutoff = _clampj(seed["f1_cut"], 0.1, rng, cut_lo - 0.05, cut_hi + 0.1)
    else:
        cutoff = float(rng.uniform(cut_lo, cut_hi))
    p.set_parameter(_P["f1_cut"], cutoff)
    res = seed.get("f1_res") if seed and "f1_res" in seed else float(rng.uniform(0.1, 0.5))
    p.set_parameter(_P["f1_res"], _clampj(res, 0.08, rng, 0.0, 0.7))

    # --- Amp envelope (Vital ADSR already 0..1; band-jitter, clamp role character) ---
    ea = seed.get("env1_a", a) if seed else a
    ed = seed.get("env1_d", d) if seed else d
    es = seed.get("env1_s", s) if seed else s
    er = seed.get("env1_r", r) if seed else r
    # clamp so jitter can't invert role: sustain>0 for sustained roles, attack≈0 for plucks
    ea = jit(ea) if role not in ("pluck", "stab") else _clampj(min(ea, 0.05), 0.02, rng, 0.0, 0.08)
    es = _clampj(es, 0.08, rng, 0.25, 1.0) if role in (
        "pad", "atmos", "reese", "wobble", "growl", "reese_pad") else jit(es)
    p.set_parameter(_P["env1_a"], ea)
    p.set_parameter(_P["env1_d"], jit(ed))
    p.set_parameter(_P["env1_s"], es)
    p.set_parameter(_P["env1_r"], jit(er))

    return {"engine": "vitalium-wt", "seeded": bool(seed),
            "preset": (seed or {}).get("_name"),
            "wave_frame": round(frame, 2), "unison": voices,
            "cutoff_norm": round(cutoff, 2), "osc2": bool(o2), "dist": bool(dist_on)}


def render(notes, secs: float, rng, role: str = "pad", seed: dict | None = None) -> np.ndarray:
    """Render a wavetable layer (2, N) for the note list with a randomized Vitalium patch.

    `seed` (optional) is a preset-derived normalized scalar overlay (see ``presets.py``).
    """
    engine = daw.RenderEngine(SR, BS)
    p = engine.make_plugin_processor("vitalium", VST)
    _patch(p, role, rng, seed)
    for (pitch, vel, t0, dur) in notes:
        p.add_midi_note(int(pitch), int(np.clip(vel, 1, 127)), float(t0), float(max(dur, 0.02)))
    engine.load_graph([(p, [])])
    engine.render(secs)
    return np.asarray(engine.get_audio())
