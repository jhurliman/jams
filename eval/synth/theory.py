"""Music-theory helpers for the D&B generator: keys, scales, chord progressions.

Everything is deterministic given a numpy Generator so a track's harmony is reproducible from
its seed. Pitches are MIDI note numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Scale interval sets (semitones from the tonic).
SCALES: dict[str, list[int]] = {
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
}

# Common minor-key progressions as scale degrees (0 = tonic). Each entry is one chord per slot;
# the arranger repeats/cycles them across the track. Roman-numeral feel noted in comments.
PROGRESSIONS: list[list[int]] = [
    [0, 5, 6, 0],       # i - VI - VII - i
    [0, 6, 5, 6],       # i - VII - VI - VII (rolling)
    [0, 3, 4, 0],       # i - iv - v - i
    [0, 5, 2, 6],       # i - VI - III - VII (epic)
    [0, 0, 5, 6],       # i - i - VI - VII
    [0, 4, 5, 0],       # i - v - VI - i
    [0, 2, 5, 6],       # i - III - VI - VII (liquid)
    [0, 6, 3, 4],       # i - VII - iv - v (dark)
]


@dataclass(frozen=True)
class Key:
    root: int          # pitch class 0-11
    scale: str         # key into SCALES
    name: str          # e.g. "A natural_minor"

    def degrees(self) -> list[int]:
        return SCALES[self.scale]

    def scale_pitches(self, low: int, high: int) -> list[int]:
        """All in-scale MIDI pitches in [low, high]."""
        degs = self.degrees()
        out = []
        for octave in range(0, 11):
            base = 12 * octave + self.root
            for d in degs:
                p = base + d
                if low <= p <= high:
                    out.append(p)
        return sorted(out)

    def degree_pitch(self, degree: int, octave: int) -> int:
        """MIDI pitch of a scale degree. ``octave`` is a multiple of 12; tonic = octave + root."""
        degs = self.degrees()
        octs, idx = divmod(degree, len(degs))
        return octave + self.root + 12 * octs + degs[idx]


def pick_key(rng: np.random.Generator, scale_weights: dict[str, float] | None = None) -> Key:
    root = int(rng.integers(0, 12))
    if scale_weights:
        names = list(scale_weights)
        w = np.array([scale_weights[n] for n in names], dtype=float)
        scale = names[int(rng.choice(len(names), p=w / w.sum()))]
    else:
        scale = str(rng.choice(list(SCALES)))
    return Key(root=root, scale=scale, name=f"{NOTE_NAMES[root]} {scale}")


def pick_progression(rng: np.random.Generator) -> list[int]:
    return list(PROGRESSIONS[int(rng.integers(0, len(PROGRESSIONS)))])


def triad(key: Key, degree: int, octave: int) -> list[int]:
    """Diatonic triad (root, third, fifth) built on a scale degree."""
    return [key.degree_pitch(degree + k, octave) for k in (0, 2, 4)]


def seventh(key: Key, degree: int, octave: int) -> list[int]:
    """Diatonic seventh chord (root, third, fifth, seventh)."""
    return [key.degree_pitch(degree + k, octave) for k in (0, 2, 4, 6)]


def bass_root(key: Key, degree: int, octave: int = 24) -> int:
    """Chord-root pitch class placed in a fixed octave window [octave, octave+12)."""
    pc = key.degree_pitch(degree, octave) % 12
    return octave + pc
