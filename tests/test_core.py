"""Model-free tests for the measurement core.

Run: python -m pytest tests/ -q
"""
import numpy as np
import pytest

from provenance_audit.core import (auroc, lda_scores_loo, loo_axis,
                                   midpoint_threshold, paired_direction, unit)


def test_unit_rejects_zero():
    with pytest.raises(ValueError):
        unit(np.zeros(8))


def test_paired_direction_points_from_neg_to_pos():
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((5, 16)) + 3.0
    neg = rng.standard_normal((5, 16))
    d = paired_direction(pos, neg)
    assert d @ (pos.mean(0) - neg.mean(0)) > 0


def test_midpoint_threshold_beats_zero_with_common_offset():
    """A large common offset makes a 0-cut degenerate; LDA cut fixes it."""
    rng = np.random.default_rng(1)
    offset = np.full(16, 20.0)
    pos = offset + rng.standard_normal((20, 16)) * 1.0
    neg = offset + np.array([6.0] + [0.0] * 15) + rng.standard_normal((20, 16)) * 1.0
    d = paired_direction(pos, neg)
    zero_cut = np.concatenate([pos @ d, neg @ d]) > 0
    lda_cut = (np.concatenate([pos @ d, neg @ d])
               - midpoint_threshold(pos, neg, d)) > 0
    y = np.concatenate([np.ones(20), np.zeros(20)])
    zero_acc = float((zero_cut == y).mean())
    lda_acc = float((lda_cut == y).mean())
    assert zero_acc == 0.5                 # the 0-cut is a coin flip
    assert lda_acc >= 0.9                  # the LDA cut separates


def test_loo_axis_recovers_signal():
    rng = np.random.default_rng(2)
    shift = np.zeros(64)
    shift[3] = 8.0
    pos = rng.standard_normal((20, 3, 64)) + shift
    neg = rng.standard_normal((20, 3, 64))
    ax = loo_axis(pos, neg, n_perm=100)
    assert ax["acc"] >= 0.9
    assert ax["p"] < 0.05


def test_loo_axis_null_is_not_significant():
    rng = np.random.default_rng(3)
    pos = rng.standard_normal((20, 3, 64))
    neg = rng.standard_normal((20, 3, 64))
    ax = loo_axis(pos, neg, n_perm=100)
    assert 0.2 <= ax["acc"] <= 0.8
    assert ax["p"] > 0.05


def test_permutation_does_not_collapse_to_identity():
    """Rows and labels must not be permuted together (that is the identity)."""
    rng = np.random.default_rng(4)
    pos = rng.standard_normal((20, 1, 32))
    neg = rng.standard_normal((20, 1, 32))
    ax = loo_axis(pos, neg, n_perm=50)
    assert ax["p"] > 0.05  # no signal -> no significance


def test_auroc_known_values():
    assert auroc([1, 0], [1.0, 0.0]) == 1.0
    assert auroc([1, 0], [0.0, 1.0]) == 0.0
    assert auroc([1, 0], [0.5, 0.5]) == 0.5


def test_lda_scores_loo_centres_on_boundary():
    rng = np.random.default_rng(5)
    H = rng.standard_normal((40, 16))
    H[20:] += 4.0
    y = np.array([True] * 20 + [False] * 20)
    s = lda_scores_loo(H, y)
    assert ((s > 0) == y).mean() >= 0.9
