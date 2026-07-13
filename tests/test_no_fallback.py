"""Guard the no-silent-fallback contract for key/tempo/audio loading.

Analysis quality must never vary with installation accidents: when essentia (or a
bundled model file) is unavailable, the analysis raises a clear RuntimeError instead of
quietly switching to a lower-accuracy method. These tests pin that behavior so a future
"helpful" fallback can't sneak back in.
"""

from __future__ import annotations

import pytest

from jams.analysis import key as K
from jams.analysis import tempo as T


def test_detect_key_propagates_cnn_worker_failure(monkeypatch, cmajor_wav):
    # Default backend is the K10 CNN worker; its failures must surface, not degrade.
    from jams.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("JAMS_KEY_BACKEND", "cnn")

    def boom(path):
        raise RuntimeError("key-cnn worker exploded")

    monkeypatch.setattr(K, "_detect_cnn", boom)
    with pytest.raises(RuntimeError, match="key-cnn worker exploded"):
        K.detect_key(cmajor_wav)
    get_settings.cache_clear()


def test_detect_key_propagates_essentia_failure(monkeypatch, cmajor_wav):
    from jams.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("JAMS_KEY_BACKEND", "fusion")

    def boom(path, refine_mode):
        raise RuntimeError("essentia exploded")

    monkeypatch.setattr(K, "_detect_essentia", boom)
    with pytest.raises(RuntimeError, match="essentia exploded"):
        K.detect_key(cmajor_wav)
    get_settings.cache_clear()


def test_detect_tempo_propagates_worker_failure(monkeypatch, cmajor_wav):
    def boom(path):
        raise RuntimeError("tempo-cnn worker exploded")

    monkeypatch.setattr(T, "_raw_bpm", boom)
    with pytest.raises(RuntimeError, match="tempo-cnn worker exploded"):
        T.detect_tempo(cmajor_wav)


def test_detect_tempo_propagates_worker_error_response(monkeypatch, cmajor_wav):
    # A worker-level {"ok": false} becomes a RuntimeError from _Worker.analyze —
    # detect_tempo must let it surface, not degrade to another tracker.
    class FakeWorker:
        def analyze(self, req):
            raise RuntimeError("tempo-cnn failed: weights missing")

    monkeypatch.setattr(T, "_tempo_cnn_worker", lambda: FakeWorker())
    with pytest.raises(RuntimeError, match="weights missing"):
        T.detect_tempo(cmajor_wav)


def test_mode_model_missing_raises(monkeypatch, tmp_path):
    K._mode_model.cache_clear()
    monkeypatch.setattr(K, "_MODE_MODEL_PATH", tmp_path / "nope.json")
    try:
        with pytest.raises(RuntimeError, match="mode model"):
            K._mode_model()
    finally:
        K._mode_model.cache_clear()


def test_no_fallback_symbols_remain():
    # The librosa key detector and secondary tempo trackers must stay deleted.
    assert not hasattr(K, "_detect_librosa")
    assert not hasattr(T, "_rhythm2013")
