"""Pure-function coverage for the archived-fusion replay helpers in key.py.

These helpers no longer run in production — eval/stats_significance.py uses them to
replay the retired edma + S-KEY fusion heads from banked features, reproducing the
paper's baseline rows. The tests pin the feature ordering and math they depend on.
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
