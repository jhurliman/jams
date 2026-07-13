"""Octave-resolution logic — pure, fast, no audio needed."""

from __future__ import annotations

import pytest

from jams.analysis.tempo import resolve_tempo_octave


@pytest.mark.parametrize(
    "bpm,genre,expected",
    [
        (87.0, "Drum & Bass", 174.0),   # half-time label folded up to full tempo
        (175.0, "Drum & Bass", 175.0),  # already full — unchanged
        (70.0, "Dubstep", 140.0),       # half folded up
        (128.0, "Deep House", 128.0),   # in range — untouched
        (174.0, None, 174.0),           # no hint — never silently folded
    ],
)
def test_genre_resolution(bpm, genre, expected):
    assert resolve_tempo_octave(bpm, genre=genre) == pytest.approx(expected, abs=0.5)


def test_explicit_range_overrides_genre():
    # explicit range wins over the genre table
    assert resolve_tempo_octave(174.0, bpm_range=(80, 100), genre="Drum & Bass") == pytest.approx(87.0, abs=0.5)


def test_no_hint_is_identity():
    assert resolve_tempo_octave(151.3) == 151.3
