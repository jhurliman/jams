"""jams — on-demand music-information-retrieval for DJ / electronic music.

Per-track analysis (no silent fallbacks — failures raise):
  - key    : our 24-class key CNN (MIT, bundled weights, in-process)
  - tempo  : our 256-class tempo CNN (MIT, bundled weights, in-process)
             + genre-aware octave resolution
  - structure (optional) : All-In-One on-device (beats / downbeats / segments)
  - stems (optional)     : Demucs 4-stem split + per-stem MIDI transcription
"""

from jams.analysis.key import detect_key
from jams.analysis.tempo import detect_tempo, resolve_tempo_octave

__version__ = "0.1.0"
__all__ = ["detect_key", "detect_tempo", "resolve_tempo_octave", "__version__"]
