"""Sidechain ducking + master bus (comp + limiter to genre LUFS).

Sidechain is a *per-stem linear gain* (applied to bass/other before summing) so the premaster
stems still sum exactly to the premaster mix. The master chain (comp + brickwall limiter to the
sub-style LUFS target) is the only across-stem nonlinearity and is applied to the MIX ONLY, with
the premaster->master gain ratio recorded so stems stay reconstructible (musdb-XL convention).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

SR = 44100


def sidechain_envelope(spec, tl) -> np.ndarray:
    """(N,) ducking gain in [1-amt, 1]. Triggers on beats (pump) or the backbeat (duck)."""
    n = tl.total_samples
    env = np.ones(n)
    amt = spec.sidechain_amt
    tau = max(spec.sidechain_ms, 5.0) / 1000.0
    steps = [0, 4, 8, 12] if spec.sidechain_style == "pump" else [0, 8]
    win = int(min(6 * tau, 0.4) * SR)
    shape_t = np.arange(win) / SR
    shape = 1.0 - amt * np.exp(-shape_t / tau)
    for bar in range(tl.total_bars):
        if tl.drum_intensity(bar) < 0.4:
            continue
        for st in steps:
            i0 = int((tl.bar_start(bar) + st * tl.step) * SR)
            i1 = min(n, i0 + win)
            if i0 < 0 or i0 >= n:
                continue
            env[i0:i1] = np.minimum(env[i0:i1], shape[: i1 - i0])
    return env


def _compress(x: np.ndarray, thresh_db=-18.0, ratio=3.0, win=0.010) -> np.ndarray:
    mono = x.mean(0)
    a = np.exp(-1.0 / (win * SR))
    envf = lfilter([1 - a], [1, -a], np.abs(mono))
    env_db = 20 * np.log10(envf + 1e-9)
    over = np.maximum(0.0, env_db - thresh_db)
    gain_db = -over * (1 - 1 / ratio)
    return x * (10 ** (gain_db / 20))[None, :]


def _brickwall(x: np.ndarray, ceil=0.985) -> np.ndarray:
    return ceil * np.tanh(x / ceil)


def master_chain(premaster: np.ndarray,
                 target_lufs: float) -> tuple[np.ndarray, float, float, float]:
    """Return (master, lufs_pre, lufs_master, premaster->master gain ratio)."""
    import pyloudnorm as pyln

    meter = pyln.Meter(SR)
    l_pre = float(meter.integrated_loudness(premaster.T))
    x = _compress(premaster * 10 ** ((target_lufs - l_pre) / 20))
    x = _brickwall(x)
    for _ in range(2):
        loud = float(meter.integrated_loudness(x.T))
        x = _brickwall(x * 10 ** ((target_lufs - loud) / 20))
    l_mas = float(meter.integrated_loudness(x.T))
    # net linear gain premaster->master, measured on RMS (records the loud master's makeup).
    pre_rms = float(np.sqrt((premaster ** 2).mean()) + 1e-12)
    mas_rms = float(np.sqrt((x ** 2).mean()) + 1e-12)
    gain_ratio = mas_rms / pre_rms
    return x, l_pre, l_mas, gain_ratio
