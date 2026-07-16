"""Per-stem bus FX to close the synthetic->real realism gap (adapted from the ES2 v2 probe).

Every function runs on an individual stem BEFORE the 4-bus sum, so the stems still sum to the
premaster mix exactly (ducking / space / drive / width are baked into each stem, exactly like a
real multitrack). Only the master (comp + limiter) is post-sum. Mechanisms verified to lift the
``other`` bus from 1.8 -> ~10.5 dB SI-SDR in the probe.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

SR = 44100


def sidechain_env(trigger_times, n: int, floor: float = 0.35, tau: float = 0.085,
                  predip: float = 0.004) -> np.ndarray:
    """Kick-triggered ducking envelope: snap to ``floor`` at each trigger, recover ~exp."""
    t = np.arange(n) / SR
    env = np.ones(n)
    for t0 in trigger_times:
        i0 = int(max(0.0, t0 - predip) * SR)
        if i0 >= n:
            continue
        rec = floor + (1 - floor) * (1 - np.exp(-(t[i0:] - t[i0]) / tau))
        env[i0:] = np.minimum(env[i0:], rec)
    return env


def duck(stem: np.ndarray, env: np.ndarray) -> np.ndarray:
    return stem * env[None, :]


def saturate(stem: np.ndarray, drive: float = 1.6, mix: float = 1.0) -> np.ndarray:
    wet = np.tanh(drive * stem) / np.tanh(drive)
    return (1 - mix) * stem + mix * wet


def _ir(dur: float = 1.4, decay: float = 5.0, lp: float = 6500, seed: int = 1) -> np.ndarray:
    n = int(dur * SR)
    r = np.random.default_rng(seed)
    env = np.exp(-np.linspace(0, decay, n))
    ir = r.standard_normal(n) * env
    ir = sosfilt(butter(4, lp / (SR / 2), btype="low", output="sos"), ir)
    ir[: int(0.005 * SR)] = 0.0
    return ir / (np.abs(ir).max() + 1e-9)


_IR_L, _IR_R = _ir(seed=1), _ir(seed=2)          # decorrelated L/R for width


def reverb(stem: np.ndarray, wet: float = 0.18) -> np.ndarray:
    left = fftconvolve(stem[0], _IR_L)[: stem.shape[1]]
    right = fftconvolve(stem[1], _IR_R)[: stem.shape[1]]
    w = np.vstack([left, right])
    w *= 1.0 / (np.abs(w).max() + 1e-9) * (np.abs(stem).max() + 1e-9)
    return (1 - wet) * stem + wet * w


def widen(stem: np.ndarray, w: float = 1.35) -> np.ndarray:
    mid = 0.5 * (stem[0] + stem[1])
    side = 0.5 * (stem[0] - stem[1]) * w
    return np.vstack([mid + side, mid - side])


def hp(stem: np.ndarray, f: float) -> np.ndarray:
    sos = butter(2, f / (SR / 2), btype="high", output="sos")
    return np.vstack([sosfilt(sos, stem[0]), sosfilt(sos, stem[1])])


def glue(stem: np.ndarray, thresh_db: float = -20.0, ratio: float = 2.5,
         win: float = 0.015) -> np.ndarray:
    """Gentle bus glue compression (drums), keyed off the smoothed level."""
    mono = stem.mean(0)
    a = np.exp(-1.0 / (win * SR))
    envf = lfilter([1 - a], [1, -a], np.abs(mono))
    over = np.maximum(0.0, 20 * np.log10(envf + 1e-9) - thresh_db)
    gain_db = -over * (1 - 1 / ratio)
    return stem * (10 ** (gain_db / 20))[None, :]
