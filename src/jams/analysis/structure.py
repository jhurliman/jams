"""Song-structure analysis (BPM, beats, downbeats, segments) via All-In-One on Replicate.

Optional: needs the ``structure`` extra (``replicate``) and a Replicate API token.
Unlike key/tempo this is a network call and incurs cost, so it is opt-in per request.
"""

from __future__ import annotations

import json
import logging

from jams.analysis.audio import validate_audio_path
from jams.config import get_settings

logger = logging.getLogger(__name__)

MODEL = "jhurliman/allinone-targetbpm"


def analyze_structure(path: str, *, target_bpm: float | None = None) -> dict:
    """Return bpm, beats, downbeats, segments for a track.

    Raises ``RuntimeError`` if Replicate isn't configured/installed.
    """
    validate_audio_path(path)
    token = get_settings().resolved_replicate_token()
    if not token:
        raise RuntimeError(
            "Structure analysis requires a Replicate token "
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
        output = client.run(MODEL, input=params)

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
