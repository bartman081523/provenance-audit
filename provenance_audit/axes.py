"""The three axes.

  context      d_ctx  "is the evidence in the prompt?"     (C vs I)
  familiarity  d_fam  "is the entity in the training data?" (F+ vs F-)
  own text     d_s    "is the preceding text mine?"         (model vs user)

Each axis is a linear direction in the residual stream plus a decision
threshold, fitted with a leave-one-out protocol on minimal pairs. Nothing is
trained: the directions are mean differences on calibration cells.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from . import cells
from .core import loo_axis, scores


def _stack(backend: Any, rows: Sequence[Dict]) -> np.ndarray:
    """Capture the last-token hidden state of every cell: (n, layers, hidden)."""
    return np.stack([backend.capture(r["ids"]) for r in rows])


def calibrate_context(backend: Any) -> Dict[str, Any]:
    c = cells.context_cells(backend.encode, backend.first_token)
    Hc, Hi = _stack(backend, c["C"]), _stack(backend, c["I"])
    ax = loo_axis(Hc, Hi)
    ax["name"] = "context"
    ax["n"] = len(c["C"])
    ax["cells"] = {"C": c["C"], "I": c["I"]}
    return ax


def calibrate_familiarity(backend: Any) -> Dict[str, Any]:
    f = cells.familiarity_cells(backend.encode)
    Hf, Hn = _stack(backend, f["F"]), _stack(backend, f["N"])
    ax = loo_axis(Hf, Hn)
    ax["name"] = "familiarity"
    ax["n"] = len(f["F"])
    return ax


def calibrate_self(backend: Any, seeds: Sequence[str],
                   n_continuation: int = 8) -> Dict[str, Any]:
    """Author: generate a continuation, then re-read it as model vs user."""
    Hm, Hu = [], []
    pairs = []
    for seed in seeds:
        gen_ids, gen_text = backend.generate(f"user: {seed}\nmodel:", n_continuation)
        own = cells.own_cells(backend.encode, seed, gen_ids)
        Hm.append(backend.capture(own["model"]["ids"]))
        Hu.append(backend.capture(own["user"]["ids"]))
        pairs.append({"seed": seed, "continuation": gen_text})
    ax = loo_axis(np.stack(Hm), np.stack(Hu))
    ax["name"] = "self"
    ax["n"] = len(Hm)
    ax["pairs"] = pairs
    return ax


def calibrate_all(backend: Any, seeds: Sequence[str]) -> Dict[str, Any]:
    """All three axes plus a verdict per axis (acc >= .75, p < .05)."""
    ctx = calibrate_context(backend)
    fam = calibrate_familiarity(backend)
    slf = calibrate_self(backend, seeds)
    out = {"context": ctx, "familiarity": fam, "self": slf,
           "backend": getattr(backend, "name", "?")}
    out["verdict"] = {
        a["name"]: bool(a["acc"] >= 0.75 and a["p"] < 0.05)
        for a in (ctx, fam, slf)
    }
    out["all_pass"] = bool(all(out["verdict"].values()))
    return out


def export(calib: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe calibration: directions, thresholds, stats (no cells)."""
    def axis(a: Dict[str, Any]) -> Dict[str, Any]:
        return {"name": a["name"], "layer": int(a["layer"]),
                "acc": float(a["acc"]), "p": float(a["p"]),
                "n": int(a["n"]),
                "direction": np.asarray(a["direction"]).tolist(),
                "threshold": float(a["threshold"])}
    return {"backend": calib.get("backend", "?"),
            "axes": {k: axis(calib[k]) for k in ("context", "familiarity",
                                                 "self")},
            "verdict": calib["verdict"], "all_pass": calib["all_pass"]}


def axis_scores(axis: Dict[str, Any], H: np.ndarray) -> np.ndarray:
    """Score the last-token state on one axis (positive = source present)."""
    h = np.atleast_2d(H[int(axis["layer"])])
    return scores(h, np.asarray(axis["direction"]), float(axis["threshold"]))
