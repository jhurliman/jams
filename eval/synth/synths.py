"""The ``other`` bus: pads / rhodes / stabs / leads / plucks / atmos, layered per track.

Surge XT covers the palette; Dexed (FM) is used for the ``rhodes`` role when available (see
``dexed.py``) for real FM timbre diversity. A light numpy reverb lifts pads/atmos so the mid
arrangement is adequately populated (the ES2 probe's sparse ``other`` bus scored only 1.8 dB).
"""

from __future__ import annotations

import numpy as np

from . import arrange, patches, theory
from . import presets as _presets
from . import wavetable as _wt
from .surge import render_layer

SR = 44100

# Fraction of Vitalium voices seeded from a license-vetted preset anchor (sparse quality anchors).
_PRESET_SEED_PROB = 0.4


# --- Note generation ---------------------------------------------------------
def _chord(spec, tl, bar: int, lo: int, hi: int, seventh: bool) -> list[int]:
    deg = tl.degree_at(bar)
    base = 12 * (lo // 12)                       # octave multiple; degree_pitch adds the root
    fn = theory.seventh if seventh else theory.triad
    notes = fn(spec.key, deg, base)
    return [p for p in notes if lo <= p <= hi] or [max(lo, min(hi, notes[0]))]


def _sustained_notes(spec, tl, rng, role: str, bars_per_chord: int, lo: int, hi: int) -> list:
    out = []
    seventh = role in ("pad", "rhodes", "reese_pad")
    for bar in range(0, tl.total_bars, bars_per_chord):
        if tl.role_intensity(role, bar) < 0.1:
            continue
        for p in _chord(spec, tl, bar, lo, hi, seventh):
            out.append((p, 78, tl.bar_start(bar), bars_per_chord * tl.bar * 0.98))
    return out


def _stab_notes(spec, tl, rng, role: str) -> list:
    out = []
    steps = {"stab": [2, 6, 10, 14], "rhodes": [0, 4, 8, 12]}.get(role, [0, 8])
    for bar in range(tl.total_bars):
        if tl.role_intensity(role, bar) < 0.15:
            continue
        chord = _chord(spec, tl, bar, 55, 74, role == "rhodes")
        for st in steps:
            if rng.random() < 0.85:
                for p in chord:
                    out.append((p, rng.integers(70, 95),
                                arrange.swing_time(tl, bar, st, spec.swing), tl.step * 2.2))
    return out


def _lead_notes(spec, tl, rng) -> list:
    out = []
    motif = [0, 2, 4, 2, 0, -1, 0, 4, 5, 4, 2, 0]
    for bar in range(tl.total_bars):
        if tl.role_intensity("lead", bar) < 0.3:
            continue
        deg = tl.degree_at(bar)
        for i, off in enumerate(motif):
            p = spec.key.degree_pitch(deg + off, 60)
            t = tl.bar_start(bar) + i * 1.3 * tl.step
            out.append((p, 88, t, tl.step * 1.2))
    return out


def _pluck_notes(spec, tl, rng) -> list:
    out = []
    for bar in range(tl.total_bars):
        if tl.role_intensity("pluck", bar) < 0.2:
            continue
        chord = _chord(spec, tl, bar, 60, 84, False)
        for st in range(0, 16, 2):
            p = chord[(st // 2) % len(chord)]
            out.append((p, rng.integers(60, 85),
                        arrange.swing_time(tl, bar, st, spec.swing), tl.step * 1.4))
    return out


def _notes_for(spec, tl, rng, role: str) -> list:
    if role in ("pad", "reese_pad"):
        return _sustained_notes(spec, tl, rng, role, 2, 55, 76)
    if role == "atmos":
        return _sustained_notes(spec, tl, rng, role, 4, 60, 84)
    if role in ("stab", "rhodes"):
        return _stab_notes(spec, tl, rng, role)
    if role == "lead":
        return _lead_notes(spec, tl, rng)
    if role == "pluck":
        return _pluck_notes(spec, tl, rng)
    return _sustained_notes(spec, tl, rng, role, 2, 55, 76)


# Per-role synth-engine weights. Engines: subtractive/wavetable Surge, FM Dexed, Vitalium
# (built-in wavetable), and `cc0wt` — the numpy CC0-wavetable scan-synth (real public-domain
# tables, the #1 timbre lever). `cc0wt` is weighted highest for the atmos/formant-rich roles.
_ENG_W = {
    "pad": {"surge": 3, "vitalium": 2, "dexed": 1, "cc0wt": 2},
    "reese_pad": {"surge": 2, "vitalium": 2, "dexed": 1, "cc0wt": 2},
    "stab": {"surge": 3, "vitalium": 2, "dexed": 1, "cc0wt": 2},
    "lead": {"surge": 2, "vitalium": 2, "dexed": 2, "cc0wt": 2},
    "pluck": {"surge": 3, "vitalium": 1, "dexed": 1, "cc0wt": 2},
    "atmos": {"surge": 1, "vitalium": 2, "dexed": 1, "cc0wt": 3},
    "rhodes": {"surge": 1, "vitalium": 1, "dexed": 4, "cc0wt": 1},
}


def _pick_engine(role: str, rng, dexed_ok: bool, vit_ok: bool, cc0wt_ok: bool) -> str:
    w = dict(_ENG_W.get(role, {"surge": 3, "vitalium": 1, "dexed": 1, "cc0wt": 1}))
    if not dexed_ok:
        w.pop("dexed", None)
    if not vit_ok:
        w.pop("vitalium", None)
    if not cc0wt_ok:
        w.pop("cc0wt", None)
    names = list(w)
    wt = np.array([w[n] for n in names], dtype=float)
    return names[int(rng.choice(len(names), p=wt / wt.sum()))]


def render_other_bus(spec, tl, rng, dexed=None, vitalium=None) -> tuple[np.ndarray, list[dict]]:
    secs = tl.total_secs
    n = tl.total_samples
    bus = np.zeros((2, n))
    descriptors: list[dict] = []

    def fit(a):
        if a.shape[1] < n:
            a = np.pad(a, ((0, 0), (0, n - a.shape[1])))
        return a[:, :n]

    dexed_ok = dexed is not None and dexed.available()
    vit_ok = vitalium is not None and vitalium.available()
    cc0wt_ok = _wt.available()
    preset_ok = _presets.available()
    for role in spec.synth_roles:
        notes = _notes_for(spec, tl, rng, role)
        if not notes:
            continue
        engine = _pick_engine(role, rng, dexed_ok, vit_ok, cc0wt_ok)
        if engine == "dexed":
            audio = fit(dexed.render(notes, secs, rng))
            descriptors.append({"role": role, "engine": "dexed-fm"})
        elif engine == "cc0wt":
            audio = fit(_wt.render(notes, secs, rng, role))
            descriptors.append({"role": role, "engine": "cc0-wavetable"})
        elif engine == "vitalium":
            seed = _presets.pick(rng) if (preset_ok and rng.random() < _PRESET_SEED_PROB) else None
            audio = fit(vitalium.render(notes, secs, rng, role, seed=seed))
            descriptors.append({"role": role, "engine": "vitalium-wt",
                                "preset_seeded": bool(seed),
                                "preset": (seed or {}).get("_name")})
        else:
            cfg, desc = patches.rand_synth_cfg(role, rng)
            audio = fit(render_layer(cfg, notes, secs, role))
            descriptors.append({"role": role, "engine": "surge", **desc})
        if role == "reese_pad":
            from .bass import _sweep_filter
            audio = _sweep_filter(audio, tl.bpm / 60.0 / 8.0, 600, 2400)
        env = arrange.env_from_intensity(tl, role)
        pan = float(rng.uniform(-0.35, 0.35))
        g = 0.5 * (pan + 1.0)
        panner = np.array([[np.sqrt(1 - g)], [np.sqrt(g)]])
        audio = audio * env[None, :] * panner
        if np.abs(audio).max() > 1e-6:
            # Balance each voice individually so a quiet random patch stays present in the mid.
            bus += arrange.balance_to(audio, -16.0, cap=12.0)
    return bus, descriptors
