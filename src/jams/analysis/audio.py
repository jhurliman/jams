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

    Uses Essentia's ``MonoLoader`` (fast, native mp3 decode, handles resampling).
    No librosa fallback: the two decoders produce subtly different samples, which
    would silently perturb every downstream feature. A failure here is either a
    broken install (essentia is a hard dependency) or an unreadable file — both
    must surface as errors, not as quality variance.
    """
    path = str(path)
    try:
        import essentia
        essentia.log.infoActive = False
        essentia.log.warningActive = False
        import essentia.standard as es
    except ImportError as exc:
        raise RuntimeError(
            "essentia-tensorflow is required for audio loading (no fallback by design). "
            "It ships wheels for macOS arm64 and Linux x86_64 on CPython 3.14."
        ) from exc
    return es.MonoLoader(filename=path, sampleRate=sample_rate)()


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
