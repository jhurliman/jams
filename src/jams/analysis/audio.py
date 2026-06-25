"""Audio loading and validation helpers."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".aiff", ".aif", ".aac", ".ogg", ".flac", ".m4a"}


def validate_audio_path(path: str | Path) -> Path:
    """Return the path as a ``Path`` or raise ``ValueError`` if it is not usable."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"File not found: {p}")
    if p.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{p.suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}")
    return p


def load_mono(path: str | Path, sample_rate: int):
    """Load audio as a mono float32 numpy array at ``sample_rate``.

    Uses Essentia's ``MonoLoader`` when available (fast, handles resampling), and
    falls back to librosa so the package stays usable without Essentia installed.
    """
    path = str(path)
    try:
        import essentia
        essentia.log.infoActive = False
        essentia.log.warningActive = False
        import essentia.standard as es

        return es.MonoLoader(filename=path, sampleRate=sample_rate)()
    except Exception as exc:  # pragma: no cover - exercised only without essentia
        logger.debug("Essentia load failed (%s); using librosa", exc)
        import librosa

        y, _ = librosa.load(path, sr=sample_rate, mono=True)
        return y


def duration_seconds(path: str | Path) -> float | None:
    """Best-effort track duration without decoding the whole file."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return round(info.frames / info.samplerate, 2)
    except Exception:
        try:
            import librosa

            return round(float(librosa.get_duration(path=str(path))), 2)
        except Exception:
            return None
