"""Per-track analysis orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from jams.analysis.audio import duration_seconds, validate_audio_path
from jams.analysis.key import detect_key
from jams.analysis.tempo import detect_tempo

logger = logging.getLogger(__name__)

__all__ = ["detect_key", "detect_tempo", "analyze_track"]

# Called as on_stage(stage, event) with event in {"start", "done"}. Stages may overlap
# in time when parallel execution is enabled.
StageCallback = Callable[[str, str], None]


def analyze_track(
    path: str,
    *,
    key: bool = True,
    tempo: bool = True,
    structure: bool = False,
    stems: bool = False,
    genre: str | None = None,
    bpm_range: tuple[float, float] | None = None,
    on_stage: StageCallback | None = None,
) -> dict:
    """Run the requested analyses on one file and return a plain dict.

    Synchronous and CPU-bound — API routes call this inside a threadpool.

    Independent stages run concurrently (key ∥ tempo→structure) unless
    ``parallel_stages`` is disabled in settings: the key CNN and structure model live
    in subprocess workers (each with its own lock) and tempo's TensorFlow inference
    releases the GIL, so threads buy real wall time. Structure stays downstream of
    tempo because it locks its beat tracker to the resolved BPM (the half-time fix),
    and stems stays last because it consumes the structure beat grid.
    """
    from jams.config import get_settings

    validate_audio_path(path)

    def _stage(name: str, event: str) -> None:
        if on_stage is not None:
            on_stage(name, event)

    def _timed(name: str, fn: Callable[[], dict]) -> dict:
        _stage(name, "start")
        t0 = time.monotonic()
        try:
            result = fn()
        finally:
            logger.info("stage %s took %.2fs", name, time.monotonic() - t0)
        _stage(name, "done")
        return result

    out: dict = {"duration_sec": duration_seconds(path)}

    def run_key() -> dict:
        return _timed("key", lambda: detect_key(path))

    def run_tempo() -> dict:
        return _timed("tempo", lambda: detect_tempo(path, genre=genre, bpm_range=bpm_range))

    def run_structure(tempo_result: dict | None) -> dict:
        from jams.analysis.structure import analyze_structure

        # Lock structure's beat tracker to the tempo we already resolved (full-tempo
        # for D&B etc.) — the half-time fix that matters most for DJ genres. Fall
        # back to the genre/range midpoint when tempo wasn't requested.
        if tempo_result is not None:
            target = tempo_result["bpm"]
        elif bpm_range:
            target = (bpm_range[0] + bpm_range[1]) / 2
        else:
            target = None
        return _timed("structure", lambda: analyze_structure(path, target_bpm=target))

    parallel = get_settings().parallel_stages and key and (tempo or structure)
    if parallel:
        def rhythm_chain() -> tuple[dict | None, dict | None]:
            t = run_tempo() if tempo else None
            s = run_structure(t) if structure else None
            return t, s

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="analyze") as pool:
            key_future = pool.submit(run_key)
            rhythm_future = pool.submit(rhythm_chain)
            # Surface the first failure; wait for both so workers aren't orphaned.
            tempo_result, structure_result = rhythm_future.result()
            out["key"] = key_future.result()
        if tempo_result is not None:
            out["tempo"] = tempo_result
        if structure_result is not None:
            out["structure"] = structure_result
    else:
        if key:
            out["key"] = run_key()
        tempo_result = run_tempo() if tempo else None
        if tempo_result is not None:
            out["tempo"] = tempo_result
        if structure:
            out["structure"] = run_structure(tempo_result)

    if stems:
        from jams.analysis.stems import analyze_stems

        # Reuse the beat grid (structure's, if computed) so transcribed onsets can snap
        # to musical positions instead of raw model timings.
        beats = out.get("structure", {}).get("beats") if structure else None
        out["stems"] = _timed(
            "stems",
            lambda: analyze_stems(path, beats=beats, quantize=get_settings().stems_quantize),
        )
    return out
