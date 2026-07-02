"""Per-track analysis orchestration."""

from __future__ import annotations

import logging

from jams.analysis.audio import duration_seconds, validate_audio_path
from jams.analysis.key import detect_key
from jams.analysis.tempo import detect_tempo

logger = logging.getLogger(__name__)

__all__ = ["detect_key", "detect_tempo", "analyze_track"]


def analyze_track(
    path: str,
    *,
    key: bool = True,
    tempo: bool = True,
    structure: bool = False,
    stems: bool = False,
    genre: str | None = None,
    bpm_range: tuple[float, float] | None = None,
) -> dict:
    """Run the requested analyses on one file and return a plain dict.

    Synchronous and CPU-bound — API routes call this inside a threadpool.
    """
    validate_audio_path(path)
    out: dict = {"duration_sec": duration_seconds(path)}

    if key:
        out["key"] = detect_key(path)
    if tempo:
        out["tempo"] = detect_tempo(path, genre=genre, bpm_range=bpm_range)
    if structure:
        from jams.analysis.structure import analyze_structure

        # Lock structure's beat tracker to the tempo we already resolved (full-tempo
        # for D&B etc.) — the half-time fix that matters most for DJ genres. Fall
        # back to the genre/range midpoint when tempo wasn't requested.
        if tempo:
            target = out["tempo"]["bpm"]
        elif bpm_range:
            target = (bpm_range[0] + bpm_range[1]) / 2
        else:
            target = None
        out["structure"] = analyze_structure(path, target_bpm=target)
    if stems:
        from jams.analysis.stems import analyze_stems
        from jams.config import get_settings

        # Reuse the beat grid (structure's, if computed) so transcribed onsets can snap
        # to musical positions instead of raw model timings.
        beats = out.get("structure", {}).get("beats") if structure else None
        out["stems"] = analyze_stems(
            path, beats=beats, quantize=get_settings().stems_quantize
        )
    return out
