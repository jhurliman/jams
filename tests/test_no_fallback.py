"""Guard the no-silent-fallback contract for key/tempo/audio loading.

Analysis quality must never vary with installation accidents: when a model or a
bundled weights file is unavailable, the analysis raises a clear RuntimeError instead
of quietly switching to a lower-accuracy method. These tests pin that behavior so a
future "helpful" fallback can't sneak back in.
"""

from __future__ import annotations

import pytest

from jams.analysis import key as K
from jams.analysis import tempo as T


def test_detect_key_propagates_cnn_failure(monkeypatch, cmajor_wav):
    # The K10 CNN is the only key backend; its failures must surface, not degrade.
    def boom(path):
        raise RuntimeError("key-cnn exploded")

    monkeypatch.setattr(K, "_detect_cnn", boom)
    with pytest.raises(RuntimeError, match="key-cnn exploded"):
        K.detect_key(cmajor_wav)


def test_detect_tempo_propagates_cnn_failure(monkeypatch, cmajor_wav):
    def boom(path):
        raise RuntimeError("tempo-cnn exploded")

    monkeypatch.setattr(T, "_raw_bpm", boom)
    with pytest.raises(RuntimeError, match="tempo-cnn exploded"):
        T.detect_tempo(cmajor_wav)


def test_detect_tempo_propagates_missing_weights(monkeypatch, cmajor_wav):
    # A missing bundled weights file raises from the in-process loader —
    # detect_tempo must let it surface, not degrade to another tracker.
    from jams.analysis import tempo_cnn

    def boom(path):
        raise RuntimeError("tempo-cnn failed: weights missing")

    monkeypatch.setattr(tempo_cnn, "analyze", boom)
    with pytest.raises(RuntimeError, match="weights missing"):
        T.detect_tempo(cmajor_wav)


def test_no_fallback_symbols_remain():
    # The librosa key detector and secondary tempo trackers must stay deleted.
    assert not hasattr(K, "_detect_librosa")
    assert not hasattr(T, "_rhythm2013")
