"""Surge XT VST3 driver (DawDreamer host), extended from the ES2 probe's ``surge_ctl``.

Surge is driven by procedurally-set parameters from its *init* patch — no factory-preset content
is copied into output. Surge XT is GPL-3 and additionally grants rights to audio rendered even
from factory presets (https://surge-synthesizer.github.io/faq/), so the render is doubly clean.
Parameter indices are Surge's public automation IDs (Surge XT 1.3.4).
"""

from __future__ import annotations

import os
import re

import dawdreamer as daw
import numpy as np

SR = 44100
BS = 512

VST = os.environ.get(
    "SYNTH_SURGE_VST3",
    os.path.expanduser("~/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"),
)

# Scene-A parameter indices (Surge XT 1.3.4).
P = {
    "vca_gain": 246, "filter_balance": 250, "waveshaper_type": 252, "waveshaper_drive": 253,
    "o1_type": 256, "o1_octave": 257, "o1_pitch": 258, "o1_shape": 259,
    "o1_uni_detune": 264, "o1_uni_voices": 265,
    "o2_type": 268, "o2_octave": 269, "o2_pitch": 270, "o2_shape": 271,
    "o1_vol": 292, "o1_mute": 293, "o2_vol": 296, "o2_mute": 297,
    "noise_vol": 312, "noise_mute": 313,
    "f1_type": 317, "f1_cutoff": 319, "f1_reso": 320, "f1_feg": 321,
    "amp_a": 329, "amp_d": 331, "amp_s": 333, "amp_r": 334,
    "feg_a": 322, "feg_d": 324, "feg_s": 326, "feg_r": 327,
    "fx_a1_type": 19, "fx_a1_p1": 20, "fx_a1_p2": 21, "fx_a1_p3": 22,
    "send1_lvl": 240,
}


def _num(text: str) -> float | None:
    m = re.search(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(m.group()) if m else None


class Surge:
    def __init__(self, engine, name: str) -> None:
        self.p = engine.make_plugin_processor(name, VST)

    def set_choice(self, idx: int, target: str, grid: int = 200):
        best = None
        for i in range(grid + 1):
            v = i / grid
            self.p.set_parameter(idx, v)
            t = self.p.get_parameter_text(idx)
            if t == target:
                return v
            if best is None and target.lower() in t.lower():
                best = v
        if best is not None:
            self.p.set_parameter(idx, best)
            return best
        raise ValueError(f"param {idx}: '{target}' not found")

    def set_num(self, idx: int, target: float, lo: float = 0.0, hi: float = 1.0, iters: int = 40):
        for _ in range(iters):
            mid = (lo + hi) / 2
            self.p.set_parameter(idx, mid)
            cur = _num(self.p.get_parameter_text(idx))
            if cur is None:
                break
            if cur < target:
                lo = mid
            else:
                hi = mid
        self.p.set_parameter(idx, (lo + hi) / 2)
        return self.p.get_parameter_text(idx)

    def set(self, idx: int, v: float) -> None:
        self.p.set_parameter(idx, float(np.clip(v, 0.0, 1.0)))

    def text(self, idx: int) -> str:
        return self.p.get_parameter_text(idx)


def render_layer(cfg, notes, secs: float, name: str = "s") -> np.ndarray:
    """Render one Surge layer to a (2, N) float array given a patch cfg + note list.

    ``notes`` is a list of (pitch, velocity, start_s, dur_s). ``cfg(surge)`` programs the patch.
    """
    engine = daw.RenderEngine(SR, BS)
    s = Surge(engine, name)
    cfg(s)
    for (pitch, vel, t0, dur) in notes:
        s.p.add_midi_note(int(pitch), int(np.clip(vel, 1, 127)), float(t0), float(max(dur, 0.02)))
    engine.load_graph([(s.p, [])])
    engine.render(secs)
    return np.asarray(engine.get_audio())
