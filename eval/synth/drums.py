"""D&B drum bus: our numpy DSP kit layered with real CC-BY one-shots + pattern grammar.

Combines original procedural drum synthesis (deterministic, seeded) with real single-drum
one-shots sliced from E-GMD (see ``oneshots.py``) to get modern drum character. Patterns are
D&B two-step / breakbeat with ghost notes, swing, phrase-end fills, section-aware intensity,
and an optional high-passed chopped-break layer (esp. jungle). All per-stem processing here is
linear across the mix, so drums+bass+other+vocals still sum exactly to the premaster.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

SR = 44100

# --- Base patterns (16th grid, one bar). kick/snare/kick2 step indices. -------
PATTERNS: dict[str, dict[str, list[int]]] = {
    "twostep_jumpup": {"kick": [0], "kick2": [6], "snare": [8]},
    "twostep_classic": {"kick": [0], "kick2": [10], "snare": [8]},
    "amen_like": {"kick": [0, 10], "kick2": [3], "snare": [4, 12]},
    "rolling": {"kick": [0, 6], "kick2": [10], "snare": [8]},
    "techstep_min": {"kick": [0], "kick2": [], "snare": [8]},
    "liquid_soft": {"kick": [0], "kick2": [10], "snare": [8]},
}

SUBSTYLE_PATTERNS: dict[str, list[str]] = {
    "jumpup": ["twostep_jumpup", "twostep_classic", "rolling"],
    "neurofunk": ["twostep_classic", "rolling", "amen_like"],
    "liquid": ["liquid_soft", "twostep_classic"],
    "jungle": ["amen_like", "rolling", "twostep_classic"],
    "techstep": ["techstep_min", "twostep_classic"],
    "dancefloor": ["twostep_jumpup", "rolling", "twostep_classic"],
}

GHOST_STEPS = [2, 3, 5, 7, 9, 11, 13, 14, 15]


def _sos_hp(x: np.ndarray, f: float, order: int = 4) -> np.ndarray:
    return sosfilt(butter(order, f / (SR / 2), btype="high", output="sos"), x)


def _sos_lp(x: np.ndarray, f: float, order: int = 4) -> np.ndarray:
    return sosfilt(butter(order, min(f, SR / 2 - 100) / (SR / 2), btype="low", output="sos"), x)


def _sos_bp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return sosfilt(butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos"), x)


def _env(n: int, a: float, d: float) -> np.ndarray:
    a_n = max(1, int(a * SR))
    e = np.ones(n)
    e[:a_n] = np.linspace(0, 1, a_n)
    d_n = min(int(d * SR), n - a_n)
    if d_n > 0:
        e[a_n:a_n + d_n] = np.exp(-np.linspace(0, 6, d_n))
    e[a_n + max(d_n, 0):] = 0.0
    return e


# --- DSP one-shots (original synthesis) --------------------------------------
def _dsp_kick(rng, f0=180, f1=70, dur=0.28) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    pitch = f1 + (f0 - f1) * np.exp(-t * 55)
    ph = 2 * np.pi * np.cumsum(pitch) / SR
    body = np.sin(ph) * _env(n, 0.001, 0.20)          # punchy, not deep-sub sustained
    click = _sos_hp(rng.standard_normal(n), 1500) * _env(n, 0.0005, 0.008) * 0.35
    return 0.9 * np.tanh(1.8 * (body + click))


def _dsp_snare(rng, tone=190, dur=0.20) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = _sos_bp(rng.standard_normal(n), 900, 8000) * _env(n, 0.0005, dur)
    body = (np.sin(2 * np.pi * tone * t) + 0.5 * np.sin(2 * np.pi * tone * 1.6 * t))
    body *= _env(n, 0.0005, 0.09)
    return 0.8 * np.tanh(1.4 * (0.7 * noise + 0.5 * body))


def _dsp_hat(rng, open_=False, bright=1.0) -> np.ndarray:
    d = 0.16 if open_ else 0.05
    n = int(d * SR)
    return 0.35 * _sos_hp(rng.standard_normal(n), 7000 * bright) * _env(n, 0.0003, d)


def _dsp_crash(rng, dur=0.7) -> np.ndarray:
    n = int(dur * SR)
    y = _sos_hp(rng.standard_normal(n), 4000) * _env(n, 0.001, dur)
    return 0.4 * y


def _stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    g = 0.5 * (pan + 1.0)
    return np.vstack([mono * np.sqrt(1 - g), mono * np.sqrt(g)])


# Per-substyle preference for the real drum source ("808" = CC0 electronic, "egmd" = CC-BY kit).
_SRC_WEIGHTS = {
    "jumpup": {"808": 3, "egmd": 2}, "dancefloor": {"808": 3, "egmd": 2},
    "techstep": {"808": 2, "egmd": 2}, "neurofunk": {"808": 2, "egmd": 3},
    "liquid": {"808": 1, "egmd": 3}, "jungle": {"808": 1, "egmd": 3},
}


class DrumEngine:
    """Per-track drum kit: chooses a real source (808/E-GMD) + DSP, with per-hit variation."""

    def __init__(self, spec, sources: dict, rng: np.random.Generator) -> None:
        self.spec = spec
        self.rng = rng
        self.bright = spec.kit_bright
        sources = sources or {}
        # Choose the real one-shot source for this track (weighted by sub-style).
        weights = _SRC_WEIGHTS.get(spec.substyle, {"egmd": 1})
        avail = {k: weights.get(k, 1) for k in sources}
        self.src_name = "none"
        self.lib = {}
        if avail:
            names = list(avail)
            w = np.array([avail[n] for n in names], dtype=float)
            self.src_name = names[int(rng.choice(len(names), p=w / w.sum()))]
            self.lib = sources[self.src_name]
        self.use_real = spec.use_real_layer and bool(self.lib.get("kick"))

        def pick(cat: str, k: int) -> list[np.ndarray]:
            pool = self.lib.get(cat, [])
            if not pool:
                return []
            idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
            return [pool[i] for i in idx]

        # Fixed per-track sample sets (round-robin for micro-variation).
        self.rk = pick("kick", 2)
        self.rs = pick("snare", 3)
        self.rh = pick("closed_hat", 3)
        self.roh = pick("open_hat", 2)
        self.rr = pick("ride", 2)
        self.rt = pick("tom", 4)
        self.rrim = pick("rim", 3)
        # Source blend: how much of the kit comes from real one-shots.
        self.kick_real = self.use_real and rng.random() < 0.7 and bool(self.rk)
        self.snare_real = self.use_real and rng.random() < 0.75 and bool(self.rs)
        self.hat_real = self.use_real and rng.random() < 0.8 and bool(self.rh)
        self.patterns = SUBSTYLE_PATTERNS[spec.substyle]
        # kick tuning per track
        self._kf0 = float(rng.uniform(155, 185))
        self._kf1 = float(rng.uniform(64, 74))
        self._stone = float(rng.uniform(170, 220))

    def _vary(self, sample: np.ndarray, semis: float = 0.6, gain_db: float = 2.5) -> np.ndarray:
        """Per-hit pitch + gain jitter (kills the machine-gun 'this is synthetic' tell)."""
        r = 2 ** (self.rng.uniform(-semis, semis) / 12)
        idx = np.arange(0, len(sample), r)
        y = np.interp(idx, np.arange(len(sample)), sample.astype(np.float64))
        return y * 10 ** (self.rng.uniform(-gain_db, gain_db) / 20)

    def _rr(self, pool: list[np.ndarray], salt: int) -> np.ndarray:
        return self._vary(pool[salt % len(pool)])

    def kick(self, vel: float, salt: int = 0) -> np.ndarray:
        dsp = _dsp_kick(self.rng, self._kf0, self._kf1)
        if self.kick_real:
            real = self._rr(self.rk, salt)
            n = max(len(dsp), len(real))
            out = np.zeros(n)
            out[:len(real)] += 0.85 * real
            out[:len(dsp)] += 0.5 * dsp        # dsp sub-punch under the real body
            return vel * out
        return vel * dsp

    def snare(self, vel: float, salt: int = 0) -> np.ndarray:
        if self.snare_real:
            real = self._rr(self.rs, salt)
            dsp = _dsp_snare(self.rng, self._stone)
            n = max(len(real), len(dsp))
            out = np.zeros(n)
            out[:len(real)] += 0.9 * real
            out[:len(dsp)] += 0.25 * dsp
            return vel * out
        return vel * _dsp_snare(self.rng, self._stone)

    def ghost(self, vel: float, salt: int = 0) -> np.ndarray:
        if self.rrim and self.rng.random() < 0.6:
            return 0.5 * vel * self._rr(self.rrim, salt)
        return 0.35 * vel * _dsp_snare(self.rng, self._stone + 20, dur=0.09)

    def hat(self, open_: bool, vel: float, salt: int = 0) -> np.ndarray:
        if open_ and self.roh and self.hat_real:
            return vel * self._rr(self.roh, salt)
        if (not open_) and self.hat_real and self.rh:
            return 0.9 * vel * self._rr(self.rh, salt)
        return vel * _dsp_hat(self.rng, open_, self.bright)

    def ride(self, vel: float, salt: int = 0) -> np.ndarray:
        if self.rr:
            return 0.8 * vel * self._rr(self.rr, salt)
        return vel * _dsp_hat(self.rng, False, self.bright * 0.8)

    def tom(self, vel: float, salt: int = 0) -> np.ndarray:
        if self.rt:
            return vel * self._rr(self.rt, salt)
        return vel * _dsp_snare(self.rng, 120, dur=0.25)

    def crash(self, vel: float) -> np.ndarray:
        if self.lib.get("crash"):
            pool = self.lib["crash"]
            return vel * self._vary(pool[int(self.rng.integers(len(pool)))], semis=0.2)
        return vel * _dsp_crash(self.rng)


def _add(buf: np.ndarray, mono: np.ndarray, t: float, pan: float, gain: float) -> None:
    st = _stereo(mono, pan) * gain
    i = int(t * SR)
    if i >= buf.shape[1]:
        return
    j = min(buf.shape[1], i + st.shape[1])
    buf[:, i:j] += st[:, : j - i]


def render_drum_bus(spec, timeline, sources: dict,
                    rng: np.random.Generator) -> tuple[np.ndarray, list[float], dict]:
    """Render the drum bus (2, N); also return kick trigger times + a kit descriptor."""
    eng = DrumEngine(spec, sources, rng)
    n = timeline.total_samples
    buf = np.zeros((2, n))
    step_s = timeline.step
    pattern_name = eng.patterns[int(rng.integers(0, len(eng.patterns)))]
    pattern = PATTERNS[pattern_name]
    hat_grid = 1 if spec.substyle in ("jumpup", "dancefloor", "jungle") else 2  # 16th vs 8th
    use_ride = spec.substyle in ("liquid", "jungle") and rng.random() < 0.5
    kick_times: list[float] = []

    def htime(bar: int, step: int) -> float:
        sw = spec.swing * step_s if (step % 2 == 1) else 0.0
        return timeline.bar_start(bar) + step * step_s + sw

    for bar in range(timeline.total_bars):
        inten = timeline.drum_intensity(bar)
        if inten <= 0.001:
            continue
        section = timeline.section_of(bar)
        phrase_end = timeline.is_phrase_end(bar)
        # Kicks
        for st in pattern["kick"]:
            t = htime(bar, st)
            _add(buf, eng.kick(rng.uniform(0.9, 1.0), st), t, 0.0, inten)
            kick_times.append(t)
        if inten > 0.6:
            for st in pattern.get("kick2", []):
                t = htime(bar, st)
                _add(buf, eng.kick(rng.uniform(0.7, 0.9), st + bar), t, 0.0, inten)
                kick_times.append(t)
        # Snares (backbeat) — only once drums are substantially in
        if inten > 0.45:
            for si, st in enumerate(pattern["snare"]):
                _add(buf, eng.snare(rng.uniform(0.9, 1.0), st + si),
                     htime(bar, st), 0.0, inten)
        # Ghost snares
        gd = spec.ghost_density * inten
        for st in GHOST_STEPS:
            if st in pattern["snare"] or st in pattern["kick"]:
                continue
            if rng.random() < gd:
                _add(buf, eng.ghost(rng.uniform(0.4, 0.8), st + bar),
                     htime(bar, st), 0.12 * (1 if st % 2 else -1), inten)
        # Hats / ride
        for st in range(0, 16, hat_grid):
            if rng.random() > (0.6 + 0.4 * inten):
                continue
            open_ = (st == 14 and rng.random() < 0.5)
            vel = rng.uniform(0.5, 0.9) * (0.7 if st % 4 else 1.0)
            if use_ride:
                _add(buf, eng.ride(vel, st + bar), htime(bar, st), 0.2, inten * 0.8)
            else:
                _add(buf, eng.hat(open_, vel, st + bar), htime(bar, st), 0.18, inten * 0.9)
        # Fills at phrase ends (esp. before drops)
        if phrase_end and inten > 0.4:
            fill_steps = [8, 10, 12, 13, 14, 15]
            for k, st in enumerate(fill_steps):
                if rng.random() < 0.5 + 0.4 * inten:
                    _add(buf, eng.tom(rng.uniform(0.6, 0.95), st + k),
                         htime(bar, st), rng.uniform(-0.3, 0.3), inten)
        # Crash on the downbeat of drops
        if timeline.is_section_start(bar) and section.startswith("drop"):
            _add(buf, eng.crash(0.8), timeline.bar_start(bar), 0.0, inten)

    # Break layer: chopped fast real hits high-passed under the kit (jungle/neuro flavour).
    if spec.use_break and eng.rs:
        buf += _break_layer(spec, timeline, eng, rng)

    # Section filtering: builds open a high-pass into the drop.
    _apply_build_filter(buf, timeline)
    # Clear the deepest sub from the drum bus so the bass owns < ~50 Hz (fixes the ES2
    # collapse signature where the drums stem absorbed the sub band).
    buf = _sos_hp(buf, 58)
    descriptor = {"real_source": eng.src_name, "pattern": pattern_name,
                  "kick_real": eng.kick_real, "snare_real": eng.snare_real,
                  "break_layer": bool(spec.use_break and eng.rs)}
    return buf, kick_times, descriptor


def _break_layer(spec, timeline, eng: DrumEngine, rng: np.random.Generator) -> np.ndarray:
    """A busy chopped layer of real snare/hat one-shots, high-passed and tucked under the kit."""
    n = timeline.total_samples
    layer = np.zeros((2, n))
    step_s = timeline.step
    for bar in range(timeline.total_bars):
        inten = timeline.drum_intensity(bar)
        if inten < 0.5:
            continue
        for st in range(16):
            if rng.random() < 0.35 * spec.ghost_density + 0.1:
                mono = eng.ghost(rng.uniform(0.3, 0.6), st + bar)
                t = timeline.bar_start(bar) + st * step_s
                _add(layer, mono, t, rng.uniform(-0.35, 0.35), inten * 0.5)
    layer = _sos_hp(layer, 220)
    return layer * 0.6


def _apply_build_filter(buf: np.ndarray, timeline) -> None:
    """High-pass build sections with a cutoff that falls back to full-range at the drop."""
    for name, b0, b1 in timeline.section_spans():
        if not name.startswith("build"):
            continue
        i0 = int(timeline.bar_start(b0) * SR)
        i1 = int(timeline.bar_start(b1) * SR) if b1 < timeline.total_bars else buf.shape[1]
        seg = buf[:, i0:i1]
        if seg.shape[1] < 32:
            continue
        # rising gate: attenuate lows early in the build, open up toward the drop
        filt = _sos_hp(seg, 400)
        ramp = np.linspace(1.0, 0.0, seg.shape[1])[None, :]
        buf[:, i0:i1] = filt * ramp + seg * (1 - ramp)
