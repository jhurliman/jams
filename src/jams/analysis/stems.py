"""Stem separation + per-stem MIDI transcription, orchestrated across two uv workers.

Splits a track into 4 stems (drums/bass/other/vocals) with Demucs, then transcribes each to
MIDI (pitched stems via basic-pitch; drums to General MIDI percussion via the ADTOF Frame_RNN
model, torch port). The heavy models have no Python 3.14 wheels and can't live in jams' env,
so they run in self-contained uv worker scripts launched via ``uv run --script`` and kept
resident, served over JSON-lines pipes — the same pattern as the structure backend.

Two workers, so the pipeline pieces stay independently replaceable and the drum model's
licensing stays subprocess-isolated (see ``drum_worker.py``):

  * ``data/stems_worker.py`` — Demucs separation + basic-pitch pitched transcription.
  * ``data/drum_worker.py``  — ADTOF (torch) drum transcription. Cross-platform, incl. arm64.

This module coordinates them and owns the cheap parts (GM canonicalisation, beat-grid
quantization, MIDI assembly — see ``jams.analysis.gm``), which run in jams' own env.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import threading
from pathlib import Path

from jams.analysis import gm
from jams.analysis.audio import validate_audio_path
from jams.config import get_settings

logger = logging.getLogger(__name__)

_DATA = Path(__file__).resolve().parent.parent / "data"
_STEMS_WORKER = _DATA / "stems_worker.py"
_DRUM_WORKER = _DATA / "drum_worker.py"

_STEM_SORT = {"drums": 0, "bass": 1, "other": 2, "vocals": 3}


def analyze_stems(
    path: str | None,
    *,
    model: str | None = None,
    stems: dict[str, str] | None = None,
    beats: list[float] | None = None,
    quantize: bool = True,
    out_dir: str | None = None,
    transcribe_drums: bool = True,
) -> dict:
    """Return separated stems + per-stem MIDI transcription for a track.

    Normally pass ``path`` (a mix) and Demucs separates it. For eval against ground-truth
    stems, pass ``stems={stem_type: wav_path}`` instead (separation is skipped). ``beats``
    (jams' resolved grid) enables onset quantization when ``quantize``.

    ``transcribe_drums=False`` skips the drum worker entirely (drums are still separated, just
    not transcribed). This is an explicit caller opt-out — e.g. on Apple Silicon, where the OaF
    drum env can't build — NOT a silent fallback; when True and the drum worker fails, the
    error propagates.
    """
    if stems is None:
        if path is None:
            raise ValueError("analyze_stems requires either 'path' or 'stems'")
        validate_audio_path(path)
    settings = get_settings()
    model = model or settings.stems_model
    work_dir = Path(out_dir or _default_out_dir(path, stems, settings))
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Separation + pitched transcription (stems_worker).
    sreq: dict = {"out_dir": str(work_dir), "model": model}
    if stems is not None:
        sreq["stems"] = stems
    else:
        sreq["audio"] = path
    sres = _stems_worker().analyze(sreq)
    stem_paths = {s["stem_type"]: s["audio_path"] for s in sres["stems"]}
    transcriptions: list[dict] = list(sres["transcriptions"])

    # 2. Drum transcription (drum_worker), if a drums stem exists and drums are requested.
    drums_wav = stem_paths.get("drums")
    drums_done = bool(drums_wav and transcribe_drums)
    if drums_done:
        raw = _drum_worker().analyze({"drums_wav": drums_wav})["notes"]
        transcriptions.append(
            {
                "stem_type": "drums",
                "gm_program": 0,
                "is_drums": True,
                "notes": gm.canon_drum_notes(raw),
                "method": "adtof",
            }
        )
    transcriptions.sort(key=lambda t: _STEM_SORT.get(t["stem_type"], 9))

    # 3. Beat-grid quantization (jams env).
    if quantize and beats:
        for t in transcriptions:
            t["notes"] = gm.quantize_notes(t["notes"], beats)

    # 4. MIDI assembly (jams env): per-stem + combined multitrack.
    midi_paths: dict[str, str] = {}
    for t in transcriptions:
        dest = work_dir / f"{t['stem_type']}.mid"
        gm.write_stem_midi(t["notes"], t["gm_program"], t["is_drums"], str(dest))
        midi_paths[t["stem_type"]] = str(dest)
    combined = work_dir / "combined.mid"
    gm.write_combined_midi(transcriptions, str(combined))
    midi_paths["combined"] = str(combined)

    sep_method = f"demucs-{model}" if stems is None else "oracle-stems"
    method = "+".join([sep_method, "basic-pitch", *(["adtof"] if drums_done else [])])
    return {
        "stems": sres["stems"],
        "transcriptions": transcriptions,
        "midi_paths": midi_paths,
        "method": method,
        "duration_sec": sres.get("duration_sec"),
    }


def _default_out_dir(path: str | None, stems: dict[str, str] | None, settings) -> str:
    base = Path(settings.stems_out_dir) if settings.stems_out_dir else (
        Path(tempfile.gettempdir()) / "jams_stems"
    )
    key = path or (next(iter(stems.values())) if stems else "stems")
    return str(base / Path(key).stem)


# --- Resident uv worker subprocesses ---------------------------------------


class _Worker:
    """Lazily-spawned, long-lived uv worker reused across requests (JSONL round-trip).

    Loads its (heavy) models once; serializes access with a lock and transparently respawns
    if the subprocess has died. Mirrors ``structure._LocalWorker``.
    """

    def __init__(self, script: Path, label: str) -> None:
        self._script = script
        self._label = label
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _spawn(self) -> None:
        uv = get_settings().stems_uv
        cmd = [uv, "run", "--script", str(self._script), "--serve"]
        logger.info("Starting %s worker: %s", self._label, " ".join(cmd))
        self._proc = subprocess.Popen(  # noqa: S603 - args are not user-controlled
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )

    def _ensure_alive(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()

    def _round_trip(self, request: str) -> str:
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(request + "\n")
        self._proc.stdin.flush()
        return self._proc.stdout.readline()

    def analyze(self, req: dict) -> dict:
        request = json.dumps(req)
        with self._lock:
            self._ensure_alive()
            try:
                line = self._round_trip(request)
            except (BrokenPipeError, ValueError, OSError):
                line = ""
            if not line:  # worker died mid-request; respawn once and retry
                logger.warning("%s worker unresponsive; respawning", self._label)
                self._spawn()
                line = self._round_trip(request)
        if not line:
            raise RuntimeError(
                f"{self._label} worker produced no output. Is `uv` installed and on PATH "
                "(JAMS_STEMS_UV)?"
            )
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"{self._label} failed: {resp.get('error', 'unknown error')}")
        return resp["result"]


_stems_singleton: _Worker | None = None
_drum_singleton: _Worker | None = None
_singleton_lock = threading.Lock()


def _stems_worker() -> _Worker:
    global _stems_singleton
    if _stems_singleton is None:
        with _singleton_lock:
            if _stems_singleton is None:
                _stems_singleton = _Worker(_STEMS_WORKER, "stems")
    return _stems_singleton


def _drum_worker() -> _Worker:
    global _drum_singleton
    if _drum_singleton is None:
        with _singleton_lock:
            if _drum_singleton is None:
                _drum_singleton = _Worker(_DRUM_WORKER, "drums")
    return _drum_singleton
