"""Per-track render: 4 premaster buses -> premaster mix -> mastered mix (musdb-XL target).

GT stems are the premaster buses (drums/bass/other/vocals) and sum *exactly* to the premaster
mix. Per-stem FX (sidechain duck, saturation, convolution reverb + width, glue) are baked into
each stem BEFORE the sum, so linearity holds. The master chain runs on the mix only; the
premaster->master gain ratio is recorded so stems remain reconstructible. Deterministic per seed.
"""

from __future__ import annotations

import numpy as np

from . import arrange, bass, drums, fx, mixmaster, synths
from . import dexed as _dexed
from . import vitalium as _vitalium

SR = 44100
# Target drop-level bus loudness (RMS dBFS): drums & bass forward, "other" present below them.
_TARGET_DBFS = {"drums": -10.0, "bass": -10.0, "other": -14.5}


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream])


def _sidechain_triggers(spec, tl, kick_times: list[float]) -> list[float]:
    if spec.sidechain_style == "pump":
        return [tl.bar_start(bar) + b * tl.beat
                for bar in range(tl.total_bars) if tl.drum_intensity(bar) > 0.4
                for b in range(4)]
    return kick_times


def render_track(spec, sources: dict, dexed=_dexed, vitalium=_vitalium) -> dict:
    tl = arrange.Timeline(spec)
    n = tl.total_samples
    rd = _rng(spec.seed, 2)

    drm_raw, kick_times, drum_desc = drums.render_drum_bus(spec, tl, sources, _rng(spec.seed, 1))
    drm = fx.glue(drm_raw)
    drm = fx.saturate(drm, drive=1.4, mix=0.3)
    drm = arrange.balance_to(drm, _TARGET_DBFS["drums"])

    bss_raw, bass_desc = bass.render_bass_bus(spec, tl, rd, vitalium)
    bss = fx.saturate(bss_raw, drive=1.3, mix=0.25)
    bss = arrange.balance_to(bss, _TARGET_DBFS["bass"])

    oth_raw, other_desc = synths.render_other_bus(spec, tl, _rng(spec.seed, 3), dexed, vitalium)
    oth = fx.reverb(oth_raw, wet=float(rd.uniform(0.12, 0.26)))
    oth = fx.widen(oth, w=float(rd.uniform(1.2, 1.45)))
    oth = arrange.balance_to(oth, _TARGET_DBFS["other"])

    # Sidechain: per-stem linear gain (stems still sum to the premaster exactly).
    triggers = _sidechain_triggers(spec, tl, kick_times)
    tau = max(spec.sidechain_ms, 5.0) / 1000.0
    bss = bss * fx.sidechain_env(triggers, n, floor=1 - spec.sidechain_amt, tau=tau)[None, :]
    oth = oth * fx.sidechain_env(triggers, n, floor=1 - 0.5 * spec.sidechain_amt, tau=tau)[None, :]

    voc = np.zeros((2, n))                        # instrumental pilot: vocals bus is TRUE silence
    stems = {"drums": drm, "bass": bss, "other": oth, "vocals": voc}
    premaster = drm + bss + oth + voc

    pk = float(np.abs(premaster).max())
    if pk > 0.9:
        g = 0.9 / pk
        for k in stems:
            stems[k] = stems[k] * g
        premaster = premaster * g

    master, l_pre, l_mas, gain_ratio = mixmaster.master_chain(premaster, spec.lufs_target)

    recon = stems["drums"] + stems["bass"] + stems["other"] + stems["vocals"]
    res = premaster - recon
    res_db = float(20 * np.log10(
        (np.sqrt((res ** 2).mean()) + 1e-12) / (np.sqrt((premaster ** 2).mean()) + 1e-12)))

    info = {
        "loudness_lufs": {"premaster": round(l_pre, 2), "master": round(l_mas, 2)},
        "master_gain_ratio": round(gain_ratio, 4),
        "premaster_to_master_gain_db": round(float(20 * np.log10(gain_ratio + 1e-12)), 2),
        "stemsum_residual_db_float": round(res_db, 1),
        "peak_premaster": round(float(np.abs(premaster).max()), 3),
        "peak_master": round(float(np.abs(master).max()), 3),
        "duration_sec": round(tl.total_secs, 2),
        "vocals": "silent",
        "timbres": {"drums": drum_desc, "bass": bass_desc, "other": other_desc},
    }
    return {"stems": stems, "premaster": premaster, "master": master, "info": info}
