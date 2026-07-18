"""Phase-3 preset seeding: license-vetted ``.vital`` presets as sparse SCALAR "quality anchors".

The Phase-1 spike proved DawDreamer cannot load a ``.vital`` (opaque VST3 binary state,
``load_preset`` returns False), so we do NOT ship or inject preset FILES. Instead we parse each
preset's JSON ``settings`` scalars and map them BY NAME to Vitalium's normalized VST3 params, to
seed an otherwise-procedural Vitalium patch which is then band-jittered per family. Presets act as
occasional anchors layered onto the procedural backbone, not a replacement.

LICENSING (bound to the user's GIMP/Audacity doctrine): the rendered AUDIO is program output, so a
GPL preset's copyleft does not propagate to it. We therefore may use GPL/CC0/Unlicense presets as
render *seeds*. GUARDRAIL: only rendered audio ships; the ``.vital`` files stay in the non-shipped
staging dir and are NEVER copied into the corpus. Vital's own factory presets are EXCLUDED.

Only SAFE, well-scaled scalar families are imported (levels, unison, filter cutoff/res, ADSR — all
either already 0..1 in Vital or trivially normalizable). Enum/topology/version fields and the
oddly-scaled master ``volume`` are ignored. Discrete/topology params are never jittered downstream.
"""

from __future__ import annotations

import functools
import glob
import json
import os

import numpy as np

# Non-shipped staging dir (see DATASET_CARD manifest). Env-overridable.
_SRC = os.environ.get("SYNTH_PRESET_SRC", "/Users/jhurliman/.claude/staging/vital-src")

# License-vetted preset sources: (glob, license, author-allowlist | None). author-allowlist
# excludes uploads whose JSON `author` doesn't match the repo uploader (task spot-check rule).
_SOURCES = [
    ("vitalium-presets/Presets/**/*.vital", "CC0-1.0", None),
    ("vital-presets/**/*.vital", "Unlicense", {"mxmfrpr"}),
    ("abstractionmage-Vital-Presets/**/*.vital", "CC0-1.0", None),
    ("nahush2321-Vitalium-presets/**/*.vital", "GPL-3.0", None),
]


def _norm_cutoff(midi_note: float) -> float:
    # Vital filter cutoff is a MIDI note (~8..136); Vitalium param 104 is normalized 0..1.
    return float(np.clip((midi_note - 8.0) / 128.0, 0.0, 1.0))


def _parse(path: str) -> dict | None:
    """Map one preset's settings -> normalized Vitalium scalar seed dict (safe params only)."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception:
        return None
    s = d.get("settings")
    if not isinstance(s, dict):
        return None
    out: dict = {"_name": d.get("preset_name") or os.path.splitext(os.path.basename(path))[0],
                 "_author": d.get("author", ""), "_path": path}

    def g(k):
        v = s.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    lvl = g("osc_1_level")
    if lvl is not None:
        out["o1_level"] = float(np.clip(lvl, 0.0, 1.0))
    uv = g("osc_1_unison_voices")
    if uv is not None:
        out["o1_uni_voices"] = int(np.clip(round(uv), 1, 16))
    ud = g("osc_1_unison_detune")
    if ud is not None:
        out["o1_uni_det"] = float(np.clip(ud / 15.0, 0.0, 1.0))   # Vital detune ~0..15
    wf = g("osc_1_wave_frame")
    if wf is not None:
        out["o1_frame"] = float(np.clip(wf / 256.0, 0.0, 1.0))
    o2l = g("osc_2_level")
    o2on = g("osc_2_on")
    if o2on is not None:
        out["o2_on"] = bool(o2on) and (o2l is None or o2l > 0.01)
        if o2l is not None:
            out["o2_level"] = float(np.clip(o2l, 0.0, 1.0))
    fc = g("filter_1_cutoff")
    if fc is not None and (g("filter_1_on") or 0) >= 1.0:
        out["f1_cut"] = _norm_cutoff(fc)
    fr = g("filter_1_resonance")
    if fr is not None:
        out["f1_res"] = float(np.clip(fr, 0.0, 1.0))
    for src, dst in (("env_1_attack", "env1_a"), ("env_1_decay", "env1_d"),
                     ("env_1_sustain", "env1_s"), ("env_1_release", "env1_r")):
        v = g(src)
        if v is not None:
            out[dst] = float(np.clip(v, 0.0, 1.0))       # Vital ADSR already normalized 0..1
    # need at least a few usable scalars to be a worthwhile anchor
    return out if len([k for k in out if not k.startswith("_")]) >= 4 else None


@functools.lru_cache(maxsize=1)
def _bank() -> list[dict]:
    if not os.path.isdir(_SRC):
        return []
    seeds: list[dict] = []
    for pat, lic, allow in _SOURCES:
        for f in glob.glob(os.path.join(_SRC, pat), recursive=True):
            seed = _parse(f)
            if seed is None:
                continue
            if allow is not None and seed.get("_author", "").strip() not in allow:
                continue                                  # author spot-check: skip non-uploader
            seed["_license"] = lic
            seeds.append(seed)
    return seeds


def available() -> bool:
    return len(_bank()) > 0


def count() -> int:
    return len(_bank())


def pick(rng) -> dict | None:
    """Return a random preset seed dict, or None if no bank is staged.

    The dict carries the normalized scalar overlay (backward-compatible) plus ``_path`` /
    ``_name`` / ``_author`` / ``_license`` metadata. Use ``load_json`` on it for full-fidelity
    (``vital_state``) loading of the preset's real wavetable + all params.
    """
    b = _bank()
    if not b:
        return None
    return b[int(rng.integers(len(b)))]


def load_json(seed: dict) -> dict | None:
    """Read the raw ``.vital`` JSON for a seed dict from ``pick`` (for full-fidelity loading)."""
    path = seed.get("_path")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None
