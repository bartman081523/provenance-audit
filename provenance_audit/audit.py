"""Paraphrase-stability audit (the external ground truth).

For an entity, ask the same question in several wordings. A memorised answer
survives the rephrasing; a confabulation does not.

    stability(e) = share of wordings that assert the same content
    H(e)         = 1 - stability(e)   if the model asserts anything, else 0

Silence is not a hallucination, so an empty answer scores 0.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .cells import PARAPHRASES


def extract_claim(text: str, vocabulary: Sequence[str]) -> Optional[str]:
    """First vocabulary word mentioned in the generated text."""
    low = text.lower()
    best = None
    for w in vocabulary:
        m = re.search(r"\b" + re.escape(w.lower()) + r"\b", low)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), w)
    return best[1] if best else None


def stability(claims: Sequence[Optional[str]]) -> (Optional[str], float):
    got = [c for c in claims if c]
    if not got:
        return None, 0.0
    counts: Dict[str, int] = {}
    for c in got:
        counts[c] = counts.get(c, 0) + 1
    modal = max(counts, key=counts.get)
    return modal, counts[modal] / len(claims)


def audit_entity(backend: Any, entity: str, vocabulary: Sequence[str],
                 n_tokens: int = 12,
                 templates: Sequence[str] = PARAPHRASES,
                 extract=None) -> Dict[str, Any]:
    """Ask the same entity in several wordings; claims must survive them.

    `templates` is adapter-tunable: a model that was trained without a
    trailing "?" is out of distribution with one, and then the audit measures
    parsing rather than coverage (measured on Tiny 2.1).
    """
    take = extract or (lambda t: extract_claim(t, vocabulary))
    claims, texts = [], []
    for prompt in (t.format(X=entity) for t in templates):
        _, text = backend.generate(prompt, n_tokens)
        claims.append(take(text))
        texts.append(text)
    modal, stab = stability(claims)
    return {"entity": entity, "claims": claims, "texts": texts,
            "claim": modal, "stability": stab,
            "asserted": modal is not None,
            "H": 0.0 if modal is None or stab >= 2 / 3 else 1.0}


def audit(backend: Any, entities: Sequence[str],
          vocabulary: Sequence[str], n_tokens: int = 12,
          templates: Sequence[str] = PARAPHRASES,
          extract=None) -> Dict[str, Any]:
    rows = [audit_entity(backend, e, vocabulary, n_tokens, templates, extract)
            for e in entities]
    asserted = [r for r in rows if r["asserted"]]
    flagged = [r for r in rows if r["H"] > 0]
    return {"rows": rows, "n": len(rows), "n_asserted": len(asserted),
            "n_flagged": len(flagged),
            "flag_rate_on_assertions": (len(flagged) / len(asserted)
                                        if asserted else 0.0),
            "mean_stability": (sum(r["stability"] for r in asserted)
                               / len(asserted) if asserted else 0.0)}
