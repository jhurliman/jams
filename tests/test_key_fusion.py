"""Unit tests for the S-KEY fusion in key detection.

The skey uv worker is mocked (nothing heavy runs). Pure-function coverage for the
posterior feature vector, both fusion decision heads, and S-KEY key-string parsing.
"""

from __future__ import annotations

import math

import pytest

from jams.analysis import key as K


def _uniform_posterior(bumps: dict[tuple[str, str], float]) -> list[float]:
    """A near-uniform 24-class posterior with named (tonic, mode) bumps."""
    p = [1.0] * 24
    for k, v in bumps.items():
        p[K._SKEY_IDX[k]] += v
    s = sum(p)
    return [x / s for x in p]


# --- pure functions ----------------------------------------------------------


def test_parse_skey_key_normalizes():
    assert K._parse_skey_key("D Major") == ("D", "major")
    assert K._parse_skey_key("Bb minor") == ("A#", "minor")
    with pytest.raises(ValueError):
        K._parse_skey_key("nonsense")


def test_skey_feats_order_and_values():
    # Bump C minor strongly; anchor at tonic C major.
    post = _uniform_posterior({("C", "minor"): 1.0})
    f = K._skey_feats(post, tonic_idx=0, edma_mode="major")
    assert len(f) == 9
    p_cmin, p_cmaj, diff = f[0], f[1], f[2]
    assert p_cmin == pytest.approx(2.0 / 25.0)
    assert p_cmaj == pytest.approx(1.0 / 25.0)
    assert diff == pytest.approx(p_cmin - p_cmaj)
    # relative minor of C = A minor; relative major anchor = D# major ((0+3)%12)
    assert f[3] == pytest.approx(1.0 / 25.0)  # A minor (uniform)
    assert f[4] == pytest.approx(1.0 / 25.0)  # D# major (uniform)
    assert f[7] == pytest.approx(max(post))
    assert f[8] == pytest.approx(-sum(x * math.log(x + 1e-12) for x in post))


def test_logistic_matches_hand_computation():
    model = {"intercept": 0.0, "mean": [0.0], "scale": [1.0], "coef": [2.0]}
    assert K._logistic(model, [0.0]) == pytest.approx(0.5)
    assert K._logistic(model, [1.0]) == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))


# --- fusion decision paths (worker + heads mocked at the model level) --------


class _FakeSkeyWorker:
    def __init__(self, key: str, posterior: list[float]):
        self.key, self.posterior = key, posterior
        self.calls: list[dict] = []

    def analyze(self, req):
        self.calls.append(req)
        return {"skey_key": self.key, "posterior": self.posterior}


def _fusion_model(mode_thr=0.7, rerank_thr=0.6, mode_bias=0.0, rerank_bias=-10.0):
    """A fusion model whose heads are pure bias terms (coef 0) — decisions forced."""
    def head(n, bias, thr):
        return {"intercept": bias, "mean": [0.0] * n, "scale": [1.0] * n,
                "coef": [0.0] * n, "threshold": thr}
    return {"mode": head(18, mode_bias, mode_thr), "rerank": head(22, rerank_bias, rerank_thr)}


@pytest.fixture()
def fused_env(monkeypatch):
    """Patch cues, worker, and fusion model; return a dict to tweak per-test."""
    env = {
        "worker": _FakeSkeyWorker("A minor", _uniform_posterior({("A", "minor"): 1.0})),
        "model": _fusion_model(),
    }
    monkeypatch.setattr(K, "_mode_features", lambda path, t: [0.0] * 8)
    monkeypatch.setattr(K, "_skey_worker", lambda: env["worker"])
    monkeypatch.setattr(K, "_key_fusion_model", lambda: env["model"])
    return env


def test_fuse_keeps_edma_when_heads_are_neutral(fused_env):
    # mode head at 0.5 (< thr both ways) keeps edma's mode; rerank bias -10 => keep.
    tonic, mode = K._fuse("/x.wav", "C", "major", 0.9)
    assert (tonic, mode) == ("C", "major")


def test_fuse_mode_head_flips_mode(fused_env):
    fused_env["model"] = _fusion_model(mode_bias=+10.0)  # P(minor) ~ 1 => minor
    tonic, mode = K._fuse("/x.wav", "C", "major", 0.9)
    assert (tonic, mode) == ("C", "minor")


def test_fuse_rerank_switches_to_skey(fused_env):
    fused_env["model"] = _fusion_model(rerank_bias=+10.0)  # P(switch) ~ 1
    tonic, mode = K._fuse("/x.wav", "C", "major", 0.9)
    assert (tonic, mode) == ("A", "minor")  # skey's key wins


def test_fuse_normalizes_skey_flats(fused_env):
    fused_env["worker"] = _FakeSkeyWorker("Bb minor", _uniform_posterior({}))
    fused_env["model"] = _fusion_model(rerank_bias=+10.0)
    tonic, mode = K._fuse("/x.wav", "C", "major", 0.9)
    assert (tonic, mode) == ("A#", "minor")


def test_fusion_model_missing_raises(monkeypatch, tmp_path):
    K._key_fusion_model.cache_clear()
    monkeypatch.setattr(K, "_KEY_FUSION_PATH", tmp_path / "nope.json")
    try:
        with pytest.raises(RuntimeError, match="key-fusion model"):
            K._key_fusion_model()
    finally:
        K._key_fusion_model.cache_clear()
