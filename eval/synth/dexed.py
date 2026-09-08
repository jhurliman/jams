"""Dexed (DX7-clone FM) voice for the ``other`` bus — a genuinely different synthesis engine.

Adds FM/metallic/bell/e-piano timbres that subtractive/wavetable Surge can't produce, widening
the spectral variety of the corpus (a key synth->real transfer lever). Dexed is GPL-3; patches
here are authored procedurally (own FM patches, no cartridge content). One flexible algorithm
(DX7 algo 5: 3 carrier+modulator pairs) with randomized ratios / levels / feedback / envelopes
yields e-piano, bell and metallic tones per track.
"""

from __future__ import annotations

import os

import dawdreamer as daw
import numpy as np

SR = 44100
BS = 512
VST = os.environ.get(
    "SYNTH_DEXED_VST3",
    os.path.expanduser("~/Library/Audio/Plug-Ins/VST3/Dexed.vst3"),
)

_CARRIERS = ("OP1", "OP3", "OP5")
_MODS = ("OP2", "OP4", "OP6")
_RATIOS = [0.5, 1, 1, 2, 3, 3.5, 4, 7, 11, 14]


def available() -> bool:
    return os.path.exists(VST)


def _names(p) -> dict[str, int]:
    return {pr["name"]: pr["index"] for pr in p.get_parameters_description()}


def _coarse_norm(ratio: float) -> float:
    # F COARSE: 0..1 maps to integer 0..31 (0 -> ratio 0.5, else the integer).
    val = 0 if ratio < 1 else int(round(min(31, ratio)))
    return val / 31.0


def _patch(p, names: dict[str, int], rng) -> dict:
    def sp(name, v):
        if name in names:
            p.set_parameter(names[name], float(np.clip(v, 0, 1)))

    sp("Output", 1.0)
    sp("ALGORITHM", 4 / 31)                       # algo 5: three 2-op stacks
    fb = float(rng.uniform(0.0, 0.55))
    sp("FEEDBACK", fb)
    car_ratio = rng.choice([0.5, 1, 1, 2])
    mod_ratios = []
    for car in _CARRIERS:
        sp(f"{car} OUTPUT LEVEL", 1.0)
        sp(f"{car} MODE", 0.0)                    # ratio mode
        sp(f"{car} F COARSE", _coarse_norm(car_ratio))
        sp(f"{car} EG LEVEL 1", 1.0)
        sp(f"{car} EG LEVEL 2", float(rng.uniform(0.7, 0.95)))
        sp(f"{car} EG LEVEL 3", float(rng.uniform(0.5, 0.85)))
        sp(f"{car} EG RATE 1", float(rng.uniform(0.75, 0.98)))
        sp(f"{car} EG RATE 2", float(rng.uniform(0.4, 0.7)))
        sp(f"{car} EG RATE 4", float(rng.uniform(0.4, 0.7)))
    for mod in _MODS:
        r = float(_RATIOS[int(rng.integers(len(_RATIOS)))])
        mod_ratios.append(r)
        sp(f"{mod} OUTPUT LEVEL", float(rng.uniform(0.35, 0.85)))   # FM depth
        sp(f"{mod} MODE", 0.0)
        sp(f"{mod} F COARSE", _coarse_norm(r))
        sp(f"{mod} EG LEVEL 1", 1.0)
        sp(f"{mod} EG LEVEL 2", float(rng.uniform(0.5, 0.9)))
        sp(f"{mod} EG RATE 1", float(rng.uniform(0.7, 0.98)))
        sp(f"{mod} EG RATE 4", float(rng.uniform(0.4, 0.7)))
    return {"engine": "dexed-fm", "feedback": round(fb, 2),
            "carrier_ratio": float(car_ratio), "mod_ratios": mod_ratios}


def render(notes, secs: float, rng) -> np.ndarray:
    """Render an FM layer (2, N) for the given note list with a randomized DX patch."""
    engine = daw.RenderEngine(SR, BS)
    p = engine.make_plugin_processor("dexed", VST)
    _patch(p, _names(p), rng)
    for (pitch, vel, t0, dur) in notes:
        p.add_midi_note(int(pitch), int(np.clip(vel, 1, 127)), float(t0), float(max(dur, 0.02)))
    engine.load_graph([(p, [])])
    engine.render(secs)
    return np.asarray(engine.get_audio())
