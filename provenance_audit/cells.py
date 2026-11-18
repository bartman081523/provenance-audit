"""Minimal-pair cells.

Every cell is built so that only the *source* of an answer changes:

  context      C vs I   same code, same question, only the box number in the
                        evidence sentence differs (one token)
  training     W        factual slot with a neutral prefix; the answer can only
                        come from the weights
  familiarity  F+ / F-  identical template, real vs invented entity
  own text     S / ~S   identical token string, `model:` vs `user:` in front

A cell is a dict with at least `ids` (token ids) and `target` (the token whose
probability is the claim), plus bookkeeping fields.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

CODES = (("1", "Berlin"), ("2", "Paris"), ("3", "Tokyo"), ("4", "Rome"),
         ("5", "Madrid"), ("6", "Vienna"), ("7", "Athens"), ("8", "Lisbon"))
FACTS = (("Germany", "Berlin"), ("France", "Paris"), ("Japan", "Tokyo"),
         ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
         ("Austria", "Vienna"), ("Greece", "Athens"))
FAMILIAR = ("Germany", "France", "Japan", "Italy", "Spain", "Portugal",
            "Austria", "Greece", "Norway", "Sweden", "Poland", "Hungary")
NOVEL = ("Vrandel", "Qorvath", "Zelmuth", "Brixalon", "Tavrenko", "Oxwald",
         "Mireth", "Kalvun", "Draneth", "Sovril", "Kalthen", "Brumire")
FAM_TEMPLATE = "{X} is a country. Its capital is"
NEUTRAL = "Note: nothing else is known about this case. "
PARAPHRASES = ("What is the capital of {X}?",
               "Which city is the capital of {X}?",
               "Name the capital of {X}.")
MARKERS = ("model", "user")

Encode = Callable[[str], List[int]]
FirstToken = Callable[[str], int]


def context_cells(enc: Encode, first: FirstToken) -> Dict[str, List[Dict]]:
    """C (evidence present, matching) and I (same word, wrong box)."""
    out: Dict[str, List[Dict]] = {"C": [], "I": []}
    for k, (box, word) in enumerate(CODES):
        other = CODES[(k + 1) % len(CODES)][0]
        slot = f"Therefore, the secret code of box {box} is"
        for arm, b in (("C", box), ("I", other)):
            out[arm].append({
                "ids": enc(f"In this protocol, the secret code of box {b} "
                           f"is {word}. {slot}"),
                "target": first(" " + word),
                "evidence": word, "box": box, "arm": arm,
            })
    return out


def fact_cells(enc: Encode, first: FirstToken) -> List[Dict]:
    """W: factual slot, neutral prefix, no evidence in the prompt."""
    return [{"ids": enc(NEUTRAL + f"question: the capital of {land} is"),
             "target": first(" " + cap), "entity": land, "answer": cap}
            for land, cap in FACTS]


def familiarity_cells(enc: Encode) -> Dict[str, List[Dict]]:
    """F+ real entities and F- invented ones, same template."""
    return {"F": [{"ids": enc(FAM_TEMPLATE.format(X=X)), "entity": X}
                  for X in FAMILIAR],
            "N": [{"ids": enc(FAM_TEMPLATE.format(X=X)), "entity": X}
                  for X in NOVEL]}


def paraphrase_prompts(entity: str) -> List[str]:
    return [p.format(X=entity) for p in PARAPHRASES]


def own_cells(enc: Encode, seed: str, continuation: Sequence[int]
              ) -> Dict[str, Dict]:
    """Same token string, authorship marker flipped."""
    return {m: {"ids": enc(f"user: {seed}\n{m}:") + list(continuation)}
            for m in MARKERS}
