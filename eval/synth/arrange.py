"""Arrangement timeline: section grammar, per-role intensity, and harmonic context.

Every instrument (drums, bass, synths) queries a single ``Timeline`` so their activity is
coherent: pads/atmos hold the intro & breakdown, drums+bass+stabs fill the drops, builds ramp
into each drop. This realises the pre-registered two-drop D&B arrangement grammar
(intro -> build -> drop1 -> break -> build2 -> drop2 -> outro), phrase-quantized.
"""

from __future__ import annotations

import numpy as np

SR = 44100
TAIL_S = 1.0

# Section name -> coarse type.
def _stype(name: str) -> str:
    for t in ("intro", "build", "drop", "break", "outro"):
        if name.startswith(t):
            return t
    return "other"


# Base per-role intensity by section type (0..1). Builds/drops override for rhythm section.
_ACTIVITY: dict[str, dict[str, float]] = {
    "drums":     {"intro": 0.0, "build": 0.6, "drop": 1.0, "break": 0.15, "outro": 0.4},
    "sub":       {"intro": 0.0, "build": 0.4, "drop": 1.0, "break": 0.3, "outro": 0.35},
    "midbass":   {"intro": 0.0, "build": 0.25, "drop": 1.0, "break": 0.1, "outro": 0.2},
    "stab":      {"intro": 0.0, "build": 0.45, "drop": 0.9, "break": 0.25, "outro": 0.3},
    "lead":      {"intro": 0.0, "build": 0.2, "drop": 0.85, "break": 0.35, "outro": 0.2},
    "pluck":     {"intro": 0.3, "build": 0.5, "drop": 0.7, "break": 0.4, "outro": 0.3},
    "pad":       {"intro": 0.85, "build": 0.6, "drop": 0.35, "break": 0.9, "outro": 0.7},
    "rhodes":    {"intro": 0.7, "build": 0.45, "drop": 0.25, "break": 0.85, "outro": 0.55},
    "atmos":     {"intro": 0.75, "build": 0.65, "drop": 0.45, "break": 0.85, "outro": 0.6},
    "reese_pad": {"intro": 0.4, "build": 0.55, "drop": 0.75, "break": 0.35, "outro": 0.3},
}


class Timeline:
    def __init__(self, spec) -> None:
        self.spec = spec
        self.bpm = spec.bpm
        self.beat = 60.0 / spec.bpm
        self.bar = 4 * self.beat
        self.step = self.beat / 4
        self.sections = spec.sections
        self.total_bars = spec.total_bars
        self.total_secs = self.total_bars * self.bar + TAIL_S
        self.total_samples = int(self.total_secs * SR)
        self.progression = spec.progression

        # Per-bar section name + section-start flags + spans.
        self._bar_section: list[str] = []
        self._spans: list[tuple[str, int, int]] = []
        b = 0
        for name, bars in self.sections:
            self._spans.append((name, b, b + bars))
            for _ in range(bars):
                self._bar_section.append(name)
            b += bars

    # --- timing ---
    def bar_start(self, bar: int) -> float:
        return bar * self.bar

    def section_of(self, bar: int) -> str:
        return self._bar_section[min(bar, self.total_bars - 1)]

    def section_spans(self):
        return self._spans

    def is_section_start(self, bar: int) -> bool:
        return any(b0 == bar for _, b0, _ in self._spans)

    def is_phrase_end(self, bar: int) -> bool:
        # last bar of the section, or every 4th bar within a long section.
        for _, b0, b1 in self._spans:
            if b0 <= bar < b1:
                return bar == b1 - 1 or ((bar - b0 + 1) % 4 == 0)
        return False

    def _build_ramp(self, bar: int) -> float:
        for _, b0, b1 in self._spans:
            if b0 <= bar < b1 and self.section_of(bar).startswith("build"):
                span = max(1, b1 - b0)
                return 0.3 + 0.7 * ((bar - b0) / span)
        return 1.0

    # --- intensities ---
    def drum_intensity(self, bar: int) -> float:
        t = _stype(self.section_of(bar))
        base = _ACTIVITY["drums"].get(t, 0.0)
        if t == "build":
            return base * self._build_ramp(bar)
        return base

    def role_intensity(self, role: str, bar: int) -> float:
        t = _stype(self.section_of(bar))
        base = _ACTIVITY.get(role, _ACTIVITY["pad"]).get(t, 0.3)
        if t == "build" and role in ("sub", "midbass", "stab", "reese_pad"):
            return base * self._build_ramp(bar)
        return base

    # --- harmony ---
    def degree_at(self, bar: int) -> int:
        return self.progression[(bar // 2) % len(self.progression)]

    def section_type(self, bar: int) -> str:
        return _stype(self.section_of(bar))


def active_rms(bus: np.ndarray) -> float:
    """RMS over frames where the bus is playing (ignores silence, doesn't over-weight transients).

    A fair 'perceived level while sounding' for both transient (drums) and sustained (bass/pad)
    buses, so balancing all buses to the same value yields a musically even mix.
    """
    mono = bus.mean(0) if bus.ndim > 1 else bus
    frame = int(0.05 * SR)
    nf = len(mono) // frame
    if nf < 2:
        return float(np.sqrt((mono ** 2).mean()) + 1e-12)
    fr = np.sqrt((mono[: nf * frame].reshape(nf, frame) ** 2).mean(axis=1))
    thr = max(0.1 * float(fr.max()), 1e-5)
    active = fr[fr > thr]
    if active.size == 0:
        return float(np.sqrt((mono ** 2).mean()) + 1e-12)
    return float(np.sqrt((active ** 2).mean()) + 1e-12)


def balance_to(bus: np.ndarray, target_dbfs: float, cap: float = 10.0) -> np.ndarray:
    return bus * min((10 ** (target_dbfs / 20)) / active_rms(bus), cap)


def swing_time(timeline: Timeline, bar: int, step: int, swing: float) -> float:
    sw = swing * timeline.step if (step % 2 == 1) else 0.0
    return timeline.bar_start(bar) + step * timeline.step + sw


def env_from_intensity(timeline: Timeline, role: str) -> np.ndarray:
    """Sample-rate gain envelope for a role across the whole track (smoothed per-bar intensity).

    Used to gate/duck sustained buses so a stem's on-disk energy matches its arrangement role,
    while staying a *linear* per-stem gain (stems still sum to the premaster exactly).
    """
    n = timeline.total_samples
    env = np.zeros(n)
    for bar in range(timeline.total_bars):
        i0 = int(timeline.bar_start(bar) * SR)
        i1 = int(timeline.bar_start(bar + 1) * SR)
        env[i0:min(i1, n)] = timeline.role_intensity(role, bar)
    # smooth transitions (~40 ms)
    k = int(0.04 * SR)
    if k > 1:
        w = np.ones(k) / k
        env = np.convolve(env, w, mode="same")
    return env
