"""jams — on-demand music-information-retrieval for DJ / electronic music.

Per-track analysis:
  - key    : Essentia ``KeyExtractor`` with the EDM-tuned ``edma`` profile
             (librosa Krumhansl-Schmuckler fallback)
  - tempo  : pretrained TempoCNN + genre-aware octave resolution
             (RhythmExtractor2013 / librosa fallback)
  - structure (optional) : All-In-One via Replicate (beats / downbeats / segments)
"""

from jams.analysis.key import detect_key
from jams.analysis.tempo import detect_tempo, resolve_tempo_octave

__version__ = "0.1.0"
__all__ = ["detect_key", "detect_tempo", "resolve_tempo_octave", "__version__"]
