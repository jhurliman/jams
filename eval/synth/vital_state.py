"""Full-fidelity `.vital` preset loader for Vitalium via DawDreamer ``load_state``.

Supersedes the scalar-only preset overlay. **Corrected root cause:** the earlier spike concluded
"scalar params only, wavetables lost" because it fed raw `.vital` bytes to ``load_preset`` and
STANDARD-base64-decoded the state chunk. In fact Vitalium's VST3 state chunk literally IS the
`.vital` JSON (wavetables included), just JUCE-wrapped — so ``load_state`` on a correctly-wrapped
state loads the preset at **full fidelity**, real embedded wavetable and all. Verified end-to-end on
arm64 Vitalium (splicing a square wave into a preset's wavetable renders the textbook odd-harmonics
spectrum through the unmodified plugin).

State chunk format (all derived at runtime — nothing hardcoded):
  - Outer: bytes ``VC2!`` + little-endian uint32 XML length, then
    ``<VST3PluginState><IComponent>…</IComponent></VST3PluginState>`` XML.
  - The ``<IComponent>`` text is a JUCE ``MemoryBlock`` base64: custom alphabet
    ``".ABC…XYZabc…xyz0-9+"`` (``.``=0, then A–Z, a–z, 0–9, ``+``), **LSB-first** bit packing,
    formatted ``<length>.<data>``.
  - Decoded, that blob = the `.vital` JSON + a fixed 32-byte ``\\x00…JUCEPrivateData`` trailer.

The XML skeleton, trailer bytes, and JSON bounds are learned once per machine from a fresh
``save_state`` (``_template``), so JUCE-version quirks auto-track.

LICENSING is unchanged: an **unmodified** redistributable GPL Vitalium binary is fed its own native
state; the rendered AUDIO is program output (GPL/CC0/Unlicense preset seeds do not propagate their
copyleft to it, per the GIMP/Audacity doctrine). We ship rendered audio, **never** the `.vital`s.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
import os
import re
import struct
import tempfile

import dawdreamer as daw
import numpy as np

SR = 44100
BS = 512
VST = os.environ.get(
    "SYNTH_VITALIUM_VST3",
    os.path.expanduser("~/Library/Audio/Plug-Ins/VST3/Vitalium.vst3"),
)

# JUCE MemoryBlock base64 alphabet (LSB-first bit packing), format "<len>.<data>".
_ALPHA = ("." + "".join(chr(c) for c in range(ord("A"), ord("Z") + 1))
          + "".join(chr(c) for c in range(ord("a"), ord("z") + 1))
          + "".join(str(d) for d in range(10)) + "+")
_IDX = {c: i for i, c in enumerate(_ALPHA)}


def available() -> bool:
    return os.path.exists(VST)


def _juce_decode(s: str) -> bytes:
    dot = s.index(".")
    n = int(s[:dot])
    bits = nbits = 0
    out = bytearray()
    for ch in s[dot + 1:]:
        bits |= _IDX[ch] << nbits
        nbits += 6
        while nbits >= 8:
            out.append(bits & 0xFF)
            bits >>= 8
            nbits -= 8
    return bytes(out[:n])


def _juce_encode(b: bytes) -> str:
    bits = nbits = 0
    chars: list[str] = []
    for byte in b:
        bits |= byte << nbits
        nbits += 8
        while nbits >= 6:
            chars.append(_ALPHA[bits & 0x3F])
            bits >>= 6
            nbits -= 6
    if nbits > 0:
        chars.append(_ALPHA[bits & 0x3F])
    return f"{len(b)}.{''.join(chars)}"


@functools.lru_cache(maxsize=1)
def _template() -> tuple[bytes, bytes, bytes]:
    """Derive (trailer, pre_xml, post_xml) once from a fresh Vitalium ``save_state``."""
    engine = daw.RenderEngine(SR, BS)
    p = engine.make_plugin_processor("v", VST)
    tf = tempfile.mktemp(suffix=".vitalstate")
    p.save_state(tf)
    with open(tf, "rb") as fh:
        raw = fh.read()
    with contextlib.suppress(OSError):
        os.remove(tf)
    if raw[:4] != b"VC2!":
        raise RuntimeError("unexpected Vitalium state header (not VC2!)")
    xlen = struct.unpack("<I", raw[4:8])[0]
    xml = raw[8:8 + xlen]
    m = re.search(rb"<IComponent>(.*?)</IComponent>", xml, re.S)
    if not m:
        raise RuntimeError("no <IComponent> in Vitalium state XML")
    blob = _juce_decode(m.group(1).decode().strip())
    bj = blob.rfind(b"}")
    trailer = blob[bj + 1:]                      # fixed \x00…JUCEPrivateData tail
    return trailer, xml[:m.start(1)], xml[m.end(1):]


def craft_state(vital_json: dict) -> bytes:
    """Wrap a `.vital` JSON dict into a Vitalium-loadable VST3 state chunk."""
    trailer, pre, post = _template()
    blob = json.dumps(vital_json, separators=(",", ":")).encode() + trailer
    xml = pre + _juce_encode(blob).encode() + post
    return b"VC2!" + struct.pack("<I", len(xml)) + xml


# --- Banded jitter applied to the JSON `settings` BEFORE encoding -------------------------------
# Per-family (role) amp-ADSR clamp targets mirror vitalium._ROLE; jitter is in NATIVE Vital units.
# NEVER-jitter: any enum / routing / topology / sync / stereo / mode / order key (see _is_enum).
_ROLE_ENV = {
    "pad": (0.45, 0.85), "reese_pad": (0.35, 0.8), "atmos": (0.6, 0.8),
    "stab": (0.02, 0.15), "lead": (0.05, 0.7), "pluck": (0.0, 0.08),
    "rhodes": (0.02, 0.35), "reese": (0.02, 0.75), "wobble": (0.03, 0.8), "growl": (0.02, 0.75),
}
_SUSTAINED = {"pad", "atmos", "reese", "wobble", "growl", "reese_pad"}
_ENUM_HINT = ("_type", "_mode", "_style", "_model", "_sync", "_stereo", "_order",
              "_routing", "_destination", "_on", "_switch", "_snap", "_quantize", "_track")


def _is_enum(key: str) -> bool:
    return any(h in key for h in _ENUM_HINT)


def _j(rng, v, amt, lo=0.0, hi=1.0):
    return float(np.clip(v + rng.uniform(-amt, amt), lo, hi))


def jitter_settings(vital_json: dict, role: str, rng) -> dict:
    """Return a deep-copied preset with per-family banded jitter on continuous scalar params only.

    Jitters osc frame/level/detune, filter cutoff/res, amp ADSR, and FX dry/wet — the timbre-morph
    families — in Vital's native units. Discrete/enum/routing/topology keys are never touched, so
    the preset's wavetables, modulation routing and structural character are preserved.
    """
    out = copy.deepcopy(vital_json)
    s = out.get("settings")
    if not isinstance(s, dict):
        return out
    a_lo, s_hi = _ROLE_ENV.get(role, (0.05, 0.6))

    def setj(k, amt, lo, hi):
        if k in s and isinstance(s[k], (int, float)) and not _is_enum(k):
            s[k] = _j(rng, float(s[k]), amt, lo, hi)

    for o in ("1", "2", "3"):
        # wave-frame / wavetable position: the timbre morph (native 0..256), ±0.15*256
        setj(f"osc_{o}_wave_frame", 0.15 * 256, 0.0, 256.0)
        setj(f"osc_{o}_level", 0.1, 0.0, 1.0)
        setj(f"osc_{o}_unison_detune", 0.1 * 15, 0.0, 15.0)      # native detune ~0..15
        setj(f"osc_{o}_spectral_morph_amount", 0.12, 0.0, 1.0)
        setj(f"osc_{o}_distortion_amount", 0.1, 0.0, 1.0)
    for f in ("1", "2", "fx"):
        setj(f"filter_{f}_cutoff", 0.1 * 128, 8.0, 136.0)        # native MIDI-note cutoff
        setj(f"filter_{f}_resonance", 0.08, 0.0, 1.0)
    # amp envelope = env_1; clamp so jitter can't invert role character
    if "env_1_attack" in s and not _is_enum("env_1_attack"):
        if role in ("pluck", "stab"):
            s["env_1_attack"] = _j(rng, min(float(s["env_1_attack"]), a_lo), 0.02, 0.0, 0.08)
        else:
            s["env_1_attack"] = _j(rng, float(s["env_1_attack"]), 0.08, 0.0, 1.0)
    setj("env_1_decay", 0.08, 0.0, 1.0)
    if "env_1_sustain" in s:
        if role in _SUSTAINED:
            s["env_1_sustain"] = _j(rng, max(float(s["env_1_sustain"]), 0.3), 0.08, 0.25, 1.0)
        else:
            s["env_1_sustain"] = _j(rng, float(s["env_1_sustain"]), 0.08, 0.0, 1.0)
    setj("env_1_release", 0.08, 0.0, 1.0)
    # FX dry/wet (continuous), never the effect_chain_order / *_on toggles
    for fx in ("reverb", "delay", "chorus", "distortion", "phaser", "flanger"):
        setj(f"{fx}_dry_wet", 0.1, 0.0, 1.0)
        setj(f"{fx}_feedback", 0.08, 0.0, 1.0)
        setj(f"{fx}_drive", 0.08, 0.0, 1.0)
    return out


def render(notes, secs: float, rng, role: str, preset_json: dict) -> np.ndarray:
    """Render a full-fidelity preset voice (2, N): jitter settings → load_state → render notes."""
    jittered = jitter_settings(preset_json, role, rng)
    state = craft_state(jittered)
    tf = tempfile.mktemp(suffix=".vitalstate")
    with open(tf, "wb") as fh:
        fh.write(state)
    engine = daw.RenderEngine(SR, BS)
    p = engine.make_plugin_processor("vitalium", VST)
    p.load_state(tf)
    with contextlib.suppress(OSError):
        os.remove(tf)
    for (pitch, vel, t0, dur) in notes:
        p.add_midi_note(int(pitch), int(np.clip(vel, 1, 127)), float(t0), float(max(dur, 0.02)))
    engine.load_graph([(p, [])])
    engine.render(secs)
    return np.asarray(engine.get_audio())
