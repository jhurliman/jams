"""Vitalium (GPL-3 wavetable synth) voice — a genuine THIRD synthesis engine.

Vitalium is DISTRHO's content-free GPL-3 build of Vital (mtytel/vital, NO_AUTH=1): no factory
presets/wavetables, no login — we drive only its init state with our own parameters, so it is
ship-clean. Its wavetable + spectral-unison + wave-frame morph give spectra distinct from both
Surge (subtractive/wavetable) and Dexed (FM), widening the corpus's timbral span (the key
synth->real transfer lever). See ``build_vitalium.sh`` for the reproducible build + license.
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

# Parameter indices (Vitalium VST3, discovered via get_parameters_description).
_P = {
    "vol": 2, "f1_cut": 104, "f1_res": 115,
    "env1_a": 48, "env1_d": 50, "env1_s": 54, "env1_r": 52,
    "o1_level": 382, "o1_frame": 397, "o1_uni_det": 395, "o1_uni_voices": 396,
    "o1_dist_amt": 377, "o2_level": 405, "o2_frame": 420, "o2_transpose": 414,
    "o2_uni_voices": 419, "o2_uni_det": 418,
}

# Per-role Env-1 (amp) ADSR as normalized Vital values + filter-cutoff range.
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


def available() -> bool:
    return os.path.exists(VST)


def _patch(p, role: str, rng) -> dict:
    a, d, s, r, cut_lo, cut_hi = _ROLE.get(role, _ROLE["pad"])

    def jit(v, amt=0.08):
        return float(np.clip(v + rng.uniform(-amt, amt), 0.0, 1.0))

    p.set_parameter(_P["vol"], 0.7)
    frame = float(rng.uniform(0.0, 1.0))                     # wavetable position (timbre morph)
    p.set_parameter(_P["o1_level"], 1.0)
    p.set_parameter(_P["o1_frame"], frame)
    voices = int(rng.integers(1, 8))
    p.set_parameter(_P["o1_uni_voices"], voices / 16.0)
    p.set_parameter(_P["o1_uni_det"], float(rng.uniform(0.1, 0.6)) if voices > 1 else 0.0)
    dist = float(rng.uniform(0.0, 0.5))
    p.set_parameter(_P["o1_dist_amt"], dist)
    o2 = rng.random() < 0.5
    if o2:                                                   # add a detuned/octave 2nd osc
        p.set_parameter(_P["o2_level"], float(rng.uniform(0.4, 0.9)))
        p.set_parameter(_P["o2_frame"], float(rng.uniform(0.0, 1.0)))
        p.set_parameter(_P["o2_transpose"], 0.5 + rng.choice([-1, 0, 1]) * (12 / 96.0))
        p.set_parameter(_P["o2_uni_voices"], int(rng.integers(1, 5)) / 16.0)
    cutoff = float(rng.uniform(cut_lo, cut_hi))
    p.set_parameter(_P["f1_cut"], cutoff)
    p.set_parameter(_P["f1_res"], float(rng.uniform(0.1, 0.5)))
    p.set_parameter(_P["env1_a"], jit(a))
    p.set_parameter(_P["env1_d"], jit(d))
    p.set_parameter(_P["env1_s"], jit(s))
    p.set_parameter(_P["env1_r"], jit(r))
    return {"engine": "vitalium-wt", "wave_frame": round(frame, 2), "unison": voices,
            "cutoff_norm": round(cutoff, 2), "osc2": bool(o2)}


def render(notes, secs: float, rng, role: str = "pad") -> np.ndarray:
    """Render a wavetable layer (2, N) for the note list with a randomized Vitalium patch."""
    engine = daw.RenderEngine(SR, BS)
    p = engine.make_plugin_processor("vitalium", VST)
    _patch(p, role, rng)
    for (pitch, vel, t0, dur) in notes:
        p.add_midi_note(int(pitch), int(np.clip(vel, 1, 127)), float(t0), float(max(dur, 0.02)))
    engine.load_graph([(p, [])])
    engine.render(secs)
    return np.asarray(engine.get_audio())
