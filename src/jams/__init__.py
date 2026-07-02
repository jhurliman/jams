"""jams — on-demand music-information-retrieval for DJ / electronic music.

Per-track analysis (essentia-tensorflow is a hard requirement — no silent fallbacks):
  - key    : Essentia ``KeyExtractor`` with the EDM-tuned ``edma`` profile + learned
             major/minor refinement
  - tempo  : pretrained TempoCNN + genre-aware octave resolution
  - structure (optional) : All-In-One on-device (beats / downbeats / segments)
  - stems (optional)     : Demucs 4-stem split + per-stem MIDI transcription
"""

from jams.analysis.key import detect_key
from jams.analysis.tempo import detect_tempo, resolve_tempo_octave

__version__ = "0.1.0"
__all__ = ["detect_key", "detect_tempo", "resolve_tempo_octave", "__version__"]
