"""Core measurement math: directions, leave-one-out axes, permutation tests.

All functions are model-free and take numpy arrays, so they can be tested
without loading a model. A "backend" only has to provide hidden states.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    """Unit-normalise; a zero vector is an instrument fault, not a result."""
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError("unit: zero vector (instrument fault, no fallback).")
    return v / n


def paired_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Mean difference direction between two groups of hidden states."""
    return unit(np.asarray(pos, np.float64).mean(axis=0)
                - np.asarray(neg, np.float64).mean(axis=0))


def midpoint_threshold(pos: np.ndarray, neg: np.ndarray,
                       d: np.ndarray) -> float:
    """LDA-style threshold: midpoint of the two projected means.

    A threshold at 0 is wrong for residual streams: they carry a large common
    offset, so a 0-cut makes the accuracy bimodal (all-positive or all-negative)
    and any permutation test meaningless.
    """
    return 0.5 * (float(np.asarray(pos, np.float64).mean(axis=0) @ d)
                  + float(np.asarray(neg, np.float64).mean(axis=0) @ d))


def _loo_stats(Hpos: np.ndarray, Hneg: np.ndarray, layer: int
               ) -> Tuple[float, np.ndarray, float]:
    """LOO accuracy on one layer: direction from the other items, LDA cut."""
    pos = Hpos[:, layer, :]
    neg = Hneg[:, layer, :]
    H = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos), bool), np.zeros(len(neg), bool)])
    hit = 0
    for i in range(len(H)):
        keep = np.arange(len(H)) != i
        p, q = keep & y, keep & ~y
        d = paired_direction(H[p], H[q])
        thr = midpoint_threshold(H[p], H[q], d)
        hit += int((float(H[i] @ d) - thr > 0) == bool(y[i]))
    d = paired_direction(pos, neg)
    return hit / len(H), d, midpoint_threshold(pos, neg, d)


def loo_axis(pos: np.ndarray, neg: np.ndarray,
             n_perm: int = 500, seed: int = 42
             ) -> Dict[str, object]:
    """Best layer + LOO accuracy + direction + threshold + permutation p.

    pos/neg: (n_items, n_layers, hidden). The permutation shuffles the labels
    independently of the rows; permuting rows *and* labels together is the
    identity and yields p = 1.0 always (a real bug we hit).
    """
    pos = np.asarray(pos, np.float64)
    neg = np.asarray(neg, np.float64)
    if pos.shape[1:] != neg.shape[1:]:
        raise ValueError("loo_axis: group shapes differ.")
    n_layers = pos.shape[1]
    stats = [_loo_stats(pos, neg, l) for l in range(n_layers)]
    accs = np.array([s[0] for s in stats])
    best = int(np.argmax(accs))
    acc, d, thr = stats[best]
    H = np.concatenate([pos[:, best, :], neg[:, best, :]], axis=0)
    y = np.concatenate([np.ones(len(pos), bool), np.zeros(len(neg), bool)])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(int(n_perm)):
        yp = rng.permutation(y)
        hit = 0
        for i in range(len(H)):
            keep = np.arange(len(H)) != i
            p, q = keep & yp, keep & ~yp
            if not p.any() or not q.any():
                continue
            dd = paired_direction(H[p], H[q])
            tt = midpoint_threshold(H[p], H[q], dd)
            hit += int((float(H[i] @ dd) - tt > 0) == bool(yp[i]))
        if hit / len(H) >= acc - 1e-12:
            hits += 1
    return {"layer": best, "acc": float(acc), "p": (1 + hits) / (1 + n_perm),
            "direction": d, "threshold": float(thr), "n": int(len(pos)),
            "acc_per_layer": accs.tolist()}


def scores(H: np.ndarray, d: np.ndarray, thr: float) -> np.ndarray:
    """Signed margin to the decision boundary for every row of H."""
    return np.asarray(H, np.float64) @ np.asarray(d, np.float64) - float(thr)


def auroc(y: Sequence[int], s: Sequence[float]) -> float:
    """Mann-Whitney AUROC with ties counted as half."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() \
        + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt) / (len(pos) * len(neg))


def lda_scores_loo(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LOO scores for a labelled set (label 1 = positive)."""
    H = np.asarray(H, np.float64)
    y = np.asarray(y, bool)
    out = np.zeros(len(H))
    for i in range(len(H)):
        keep = np.arange(len(H)) != i
        p, q = keep & y, keep & ~y
        d = paired_direction(H[p], H[q])
        out[i] = float(H[i] @ d) - midpoint_threshold(H[p], H[q], d)
    return out
