"""Song-structure analysis (BPM, beats, downbeats, segments) via All-In-One.

Two backends, selected by ``JAMS_STRUCTURE_BACKEND``:

- ``local`` (default): runs All-In-One on-device (Apple Silicon / PyTorch-MPS).
  The model needs torch/natten-mps/demucs, which have no Python 3.14 wheels and so
  can't live in jams' own env; instead a self-contained uv worker script
  (``data/structure_worker.py``) is launched once via ``uv run --script`` and kept
  resident, serving requests over a JSON-lines pipe. First request pays a one-time
  env build + model load (~tens of seconds); subsequent ones are ~10 s/track.
- ``replicate``: calls the hosted All-In-One model. Network + cost, needs a token.

``target_bpm`` constrains beat tracking to ``target_bpm +/- 1`` — critical for
half-time genres (D&B/dubstep) where the tracker otherwise lands an octave low.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path

from jams.analysis.audio import validate_audio_path
from jams.config import get_settings

logger = logging.getLogger(__name__)

REPLICATE_MODEL = "jhurliman/allinone-targetbpm"
_WORKER_PATH = Path(__file__).resolve().parent.parent / "data" / "structure_worker.py"


def analyze_structure(
    path: str, *, target_bpm: float | None = None, model: str | None = None
) -> dict:
    """Return ``bpm, beats, downbeats, segments, method`` for a track.

    Dispatches to the configured backend. ``model`` overrides the configured
    All-In-One model (e.g. ``harmonix-fold3`` for per-fold cross-validation).
    """
    validate_audio_path(path)
    settings = get_settings()
    if settings.structure_backend == "replicate":
        return _analyze_replicate(path, target_bpm=target_bpm)
    return _local_worker().analyze(path, target_bpm, model or settings.structure_model)


# --- Local backend (resident uv worker subprocess) -------------------------


class _LocalWorker:
    """Lazily-spawned, long-lived All-In-One worker reused across requests.

    The worker loads the (heavy) model once; we serialize access with a lock and
    transparently respawn it if it has died.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _spawn(self) -> None:
        uv = get_settings().structure_uv
        cmd = [uv, "run", "--script", str(_WORKER_PATH), "--serve"]
        logger.info("Starting All-In-One structure worker: %s", " ".join(cmd))
        # stderr is inherited so uv's env-build progress and any tracebacks are
        # visible; the worker keeps stdout clean (one JSON line per request).
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

    def analyze(self, audio: str, target_bpm: float | None, model: str) -> dict:
        request = json.dumps({"audio": audio, "target_bpm": target_bpm, "model": model})
        with self._lock:
            self._ensure_alive()
            try:
                line = self._round_trip(request)
            except (BrokenPipeError, ValueError, OSError):
                line = ""
            if not line:  # worker died mid-request; respawn once and retry
                logger.warning("Structure worker unresponsive; respawning")
                self._spawn()
                line = self._round_trip(request)
        if not line:
            raise RuntimeError(
                "Structure worker produced no output. Is `uv` installed and on PATH "
                "(JAMS_STRUCTURE_UV), and is this an Apple Silicon Mac?"
            )
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"Structure analysis failed: {resp.get('error', 'unknown error')}")
        return resp["result"]


_worker_singleton: _LocalWorker | None = None
_worker_singleton_lock = threading.Lock()


def _local_worker() -> _LocalWorker:
    global _worker_singleton
    if _worker_singleton is None:
        with _worker_singleton_lock:
            if _worker_singleton is None:
                _worker_singleton = _LocalWorker()
    return _worker_singleton


# --- Replicate backend (hosted, opt-in) ------------------------------------


def _analyze_replicate(path: str, *, target_bpm: float | None = None) -> dict:
    token = get_settings().resolved_replicate_token()
    if not token:
        raise RuntimeError(
            "Structure backend 'replicate' requires a token "
            "(set JAMS_REPLICATE_API_TOKEN or REPLICATE_API_TOKEN)."
        )
    try:
        import replicate
    except ImportError as exc:
        raise RuntimeError("Install the 'structure' extra: pip install 'jams[structure]'") from exc

    client = replicate.Client(api_token=token)
    with open(path, "rb") as fh:
        params: dict = {"audio": fh}
        if target_bpm is not None:
            params["target_bpm"] = target_bpm
        output = client.run(REPLICATE_MODEL, input=params)

    if isinstance(output, str):
        result = json.loads(output)
    elif isinstance(output, list) and output:
        import requests

        url = next((u for u in output if isinstance(u, str) and u.endswith(".json")), output[0])
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        result = resp.json()
    else:
        raise RuntimeError(f"Unexpected Replicate output: {type(output)}")

    return {
        "bpm": result.get("bpm"),
        "beats": result.get("beats", []),
        "downbeats": result.get("downbeats", []),
        "segments": result.get("segments", []),
        "method": "allin1-replicate",
    }
