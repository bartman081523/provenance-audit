"""The one-pass indicator.

A claim is *justified* if the evidence is in the prompt (context axis) or the
entity is in the training data (familiarity axis). Otherwise it is *unbacked*
-- which is exactly the provenance notion of hallucination:

    hallucination = not justified by training AND not given by the prompt

This says nothing about truth. A memorised falsehood is justified by training
and will not be flagged; catching that needs external verification.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .axes import axis_scores, export


def flag(backend: Any, calib: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Read one text in one forward and return the justification verdict."""
    axes = calib["axes"] if "axes" in calib else {
        k: {"layer": calib[k]["layer"],
            "direction": calib[k]["direction"],
            "threshold": calib[k]["threshold"]}
        for k in ("context", "familiarity", "self")}
    H = backend.capture(backend.encode(text))
    ctx = float(axis_scores(axes["context"], H)[0])
    fam = float(axis_scores(axes["familiarity"], H)[0])
    slf = float(axis_scores(axes["self"], H)[0])
    ctx_on = ctx > 0.0
    fam_on = fam > 0.0
    return {"justified": bool(ctx_on or fam_on),
            "reason": "context" if ctx_on else ("familiar" if fam_on else "none"),
            "score_context": ctx, "score_familiarity": fam, "score_self": slf,
            "p_top": backend.top_prob(backend.encode(text))}


def batch(backend: Any, calib: Dict[str, Any], texts) -> Dict[str, Any]:
    rows = [{"text": t, **flag(backend, calib, t)} for t in texts]
    unbacked = [r for r in rows if not r["justified"]]
    return {"n": len(rows), "n_unbacked": len(unbacked),
            "rate": len(unbacked) / len(rows) if rows else 0.0, "rows": rows}


def load(path: str) -> Dict[str, Any]:
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
