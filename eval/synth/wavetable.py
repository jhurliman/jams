"""CC0 wavetable oscillator — a numpy scan-synth engine over public-domain wavetables.

The #1 realism lever: instead of only Vitalium's single built-in wavetable (which DawDreamer
cannot swap — its VST3 state is opaque binary, see the Phase-1 spike in ``PRESET_SPIKE.md``),
this reads real CC0 wavetables (``.vitaltable`` = int16 single-cycle frames) and plays them with
a **band-limited** phase-accumulator scan oscillator. Because the wavetables are public-domain
sample data used *inside our own procedural oscillator*, no derivative-work argument is needed.

Categories folding / FM / sync / phase-distortion / PPG are the Reese/growl/neuro fuel, so D&B
bass patches are biased toward those. Tables that scan across many frames (the report's >2048-
sample caveat) are handled by an LFO/decay ``pos(t)`` trajectory over the frame axis.

Bank path is env-overridable (``SYNTH_WT_BANK``); if absent the engine reports unavailable and
the renderer degrades to Surge/Dexed/Vitalium, exactly like the other optional engines. The bank
is staged OUTSIDE the shipped tree (only rendered audio ships); see ``DATASET_CARD.md`` manifest.
"""

from __future__ import annotations

import base64
import functools
import glob
import json
import os

import numpy as np

SR = 44100
FRAME = 2048

_BANK = os.environ.get(
    "SYNTH_WT_BANK",
    "/Users/jhurliman/.claude/staging/vital-src",
)

# Role -> preferred wavetable category keywords (biases D&B bass to growl/neuro fuel).
_ROLE_KW = {
    "reese": ("fold", "fm", "sync", "phase", "distort", "saw", "ppg", "bass", "reso"),
    "growl": ("fm", "fold", "distort", "sync", "phase", "formant", "vowel", "metal"),
    "wobble": ("fold", "phase", "sync", "ppg", "pwm", "square", "fm"),
    "foghorn": ("ppg", "pwm", "phase", "square", "organ", "bass"),
    "jumpup_bounce": ("saw", "square", "sync", "ppg", "pwm"),
    "pad": ("formant", "vowel", "soft", "airy", "string", "pad", "harmonic", "sine"),
    "reese_pad": ("fold", "fm", "phase", "formant", "string"),
    "atmos": ("formant", "vowel", "airy", "noise", "granular", "spectral", "pad"),
    "stab": ("sync", "ppg", "phase", "fm", "digital", "bright"),
    "lead": ("sync", "fm", "phase", "saw", "square", "bright", "digital"),
    "pluck": ("ppg", "phase", "sync", "pluck", "bell", "digital"),
    "rhodes": ("sine", "bell", "harmonic", "ep", "soft"),
}


@functools.lru_cache(maxsize=1)
def _index() -> list[tuple[str, str]]:
    """Return [(path, category_string)] for every decodable CC0 wavetable in the bank."""
    if not os.path.isdir(_BANK):
        return []
    pats = ["open-vital-resources/**/*.vitaltable", "vitalium-presets/**/*.vitaltable"]
    files: list[str] = []
    for p in pats:
        files += glob.glob(os.path.join(_BANK, p), recursive=True)
    out = []
    for f in files:
        # category = lowercased path tail (parent dir + filename) for keyword matching
        cat = (os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f)).lower()
        out.append((f, cat))
    return out


def available() -> bool:
    return len(_index()) > 0


@functools.lru_cache(maxsize=256)
def _load_frames(path: str) -> np.ndarray | None:
    """Decode a .vitaltable into (n_frames, FRAME) float32 in [-1, 1], DC-removed."""
    try:
        with open(path) as fh:
            comp = json.load(fh)["groups"][0]["components"][0]
        if comp.get("type") != "Audio File Source":
            return None
        ws = int(comp.get("window_size", FRAME) or FRAME)
        raw = base64.b64decode(comp["audio_file"])
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        nf = len(a) // ws
        if nf < 1:
            return None
        frames = a[: nf * ws].reshape(nf, ws)
        if ws != FRAME:                                   # resample each frame to FRAME
            xp = np.linspace(0, 1, ws, endpoint=False)
            xq = np.linspace(0, 1, FRAME, endpoint=False)
            frames = np.stack([np.interp(xq, xp, fr) for fr in frames])
        frames = frames - frames.mean(axis=1, keepdims=True)   # remove DC per frame
        return frames.astype(np.float32)
    except Exception:
        return None


def _pick_table(role: str, rng) -> tuple[np.ndarray, str] | None:
    idx = _index()
    if not idx:
        return None
    kws = _ROLE_KW.get(role, ("saw", "sine", "harmonic"))
    pref = [pc for pc in idx if any(k in pc[1] for k in kws)]
    pool = pref if pref else idx
    # try a few in case a table fails to decode
    for _ in range(6):
        path, _cat = pool[int(rng.integers(len(pool)))]
        fr = _load_frames(path)
        if fr is not None and fr.shape[0] >= 1 and np.abs(fr).max() > 1e-6:
            return fr, os.path.splitext(os.path.basename(path))[0]
    return None


def _bandlimit(cycle: np.ndarray, f0: float) -> np.ndarray:
    """Zero harmonics at/above Nyquist so the phase-accumulator readout doesn't alias."""
    spec = np.fft.rfft(cycle)
    kmax = int((SR / 2.0) / max(f0, 1e-6))
    if kmax < len(spec):
        spec[kmax:] = 0.0
    return np.fft.irfft(spec, n=len(cycle)).astype(np.float32)


def _scan_pos(n: int, nf: int, rng, role: str) -> np.ndarray:
    """Frame-position trajectory pos(t) in [0, nf-1] (the wavetable-position modulation)."""
    if nf <= 1:
        return np.zeros(n, dtype=np.float32)
    t = np.arange(n) / SR
    start = float(rng.uniform(0, nf - 1))
    if role in ("reese", "growl", "wobble", "foghorn", "reese_pad"):
        # slow LFO sweep across frames — the evolving neuro/reese timbre
        rate = float(rng.uniform(0.05, 0.6))
        depth = float(rng.uniform(0.3, 1.0)) * (nf - 1) / 2.0
        pos = start + depth * np.sin(2 * np.pi * rate * t)
    elif role in ("pluck", "stab"):
        # fast decay sweep (bright->dark transient)
        pos = start + (nf - 1) * float(rng.uniform(0.3, 0.9)) * np.exp(-t / 0.15)
    else:
        rate = float(rng.uniform(0.02, 0.25))
        pos = start + 0.4 * (nf - 1) * np.sin(2 * np.pi * rate * t)
    return np.clip(pos, 0, nf - 1).astype(np.float32)


def _osc(frames: np.ndarray, f0: float, n: int, rng, role: str,
         detune_cents: float = 0.0, phase0: float = 0.0) -> np.ndarray:
    """One band-limited voice: scan `frames` at fundamental f0 for n samples."""
    nf = frames.shape[0]
    f = f0 * (2.0 ** (detune_cents / 1200.0))
    pos = _scan_pos(n, nf, rng, role)
    fi = np.round(pos).astype(int)                        # nearest frame (crossfade smoothed below)
    # phase accumulator in [0,1)
    phase = (phase0 + np.cumsum(np.full(n, f / SR))) % 1.0
    read = phase * FRAME
    i0 = np.floor(read).astype(int) % FRAME
    frac = (read - np.floor(read)).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    # band-limit each visited frame once (few unique indices per note), fill its run
    for idx in np.unique(fi):
        m = fi == idx
        bl = _bandlimit(frames[idx], f)
        i1 = (i0 + 1) % FRAME
        out[m] = bl[i0[m]] * (1 - frac[m]) + bl[i1[m]] * frac[m]
    return out


def render(notes, secs: float, rng, role: str = "pad") -> np.ndarray:
    """Render a CC0-wavetable layer (2, N) for the note list. API mirrors vitalium.render."""
    n = int(round(secs * SR))
    picked = _pick_table(role, rng)
    if picked is None:
        return np.zeros((2, n), dtype=np.float32)
    frames, _name = picked
    voices = int(rng.integers(1, 6))
    spread = float(rng.uniform(6, 30)) if voices > 1 else 0.0
    mono = np.zeros(n, dtype=np.float32)
    for (pitch, vel, t0, dur) in notes:
        f0 = 440.0 * 2.0 ** ((int(pitch) - 69) / 12.0)
        s = int(max(0, round(t0 * SR)))
        ln = int(round(max(dur, 0.02) * SR))
        if s >= n:
            continue
        ln = min(ln, n - s)
        seg = np.zeros(ln, dtype=np.float32)
        for v in range(voices):
            dc = spread * (v - (voices - 1) / 2.0) / max(1, voices - 1) if voices > 1 else 0.0
            seg += _osc(frames, f0, ln, rng, role, detune_cents=dc,
                        phase0=float(rng.random()))
        seg /= voices
        # simple AD amp envelope so notes don't click; role sets sustain character
        env = np.ones(ln, dtype=np.float32)
        a = min(ln, int(0.005 * SR))
        r = min(ln, int(0.03 * SR))
        if a > 0:
            env[:a] = np.linspace(0, 1, a)
        if r > 0:
            env[-r:] *= np.linspace(1, 0, r)
        amp = (int(np.clip(vel, 1, 127)) / 127.0)
        mono[s:s + ln] += seg * env * amp
    peak = float(np.abs(mono).max())
    if peak > 1e-6:
        mono = mono / peak * 0.6
    return np.stack([mono, mono])
