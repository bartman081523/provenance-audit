"""The one-pass indicator.

A claim is *justified* if the evidence is in the prompt (context axis) or the
entity is in the training data (familiarity axis). Otherwise it is *unbacked*
-- which is exactly the provenance notion of hallucination:

    hallucination = not justified by training AND not given by the prompt

This says nothing about truth. A memorised falsehood is justified by training
and will not be flagged; catching that needs external verification.
"""
from __future__ import annotations

from typing import Any, Dict

from .axes import axis_scores


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
    # Only axes that passed their own calibration may justify an answer.
    # An axis that separates nothing (p ~ 1, acc ~ chance) still produces a
    # non-zero dot product, and using its sign would call every answer
    # "context-justified". Measured on Tiny 2.1, whose context axis has no
    # in-context channel at all.
    verdict = calib.get("verdict") or {}
    active = {k: bool(verdict.get(k, True))
              for k in ("context", "familiarity", "self")}
    ctx_on = active["context"] and ctx > 0.0
    fam_on = active["familiarity"] and fam > 0.0
    if not (active["context"] or active["familiarity"]):
        return {"justified": None, "reason": "no-usable-axis",
                "active": active, "score_context": ctx,
                "score_familiarity": fam, "score_self": slf,
                "p_top": backend.top_prob(backend.encode(text))}
    return {"justified": bool(ctx_on or fam_on),
            "reason": "context" if ctx_on else ("familiar" if fam_on else "none"),
            "active": active,
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
