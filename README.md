# provenance-audit

Provenance axes and a **one-pass hallucination indicator** for causal language
models.

Hallucination here is a *provenance* statement, not a truth statement:

> a claim is hallucinated if it is **not justified by training** and **not
> given by the prompt** (including the prompt's projection onto training).

So the toolkit measures **coverage**, not truth. A memorised falsehood is
covered by training and is *not* flagged — catching it needs external
verification. This boundary is deliberate and is the reason the numbers below
are honest.

## The idea

Three linear directions in the residual stream, each fitted on **minimal
pairs** so that only the source of an answer changes:

| axis | question | minimal pair |
| :--- | :--- | :--- |
| `context` | is the evidence in the prompt? | same code, same question, only the box number in the evidence differs |
| `familiarity` | is the entity in the training data? | same template, real entity vs invented entity |
| `self` | is the preceding text mine? | same token string, `model:` vs `user:` in front |

Two of them justify a claim; the third separates "my own text" from "given
text":

```
justified = context-present  OR  entity-familiar
reason    = "context" | "familiar" | "none"
```

Everything is read-only (forward hooks, `torch.no_grad`). Nothing is trained:
the directions are mean differences on calibration cells, with
leave-one-out validation and a label-permutation test.

## Install

```bash
pip install -e .            # numpy only
pip install -e ".[hf]"      # + transformers backend
pip install -e ".[nanogpt]" # + nanoGPT backend
python -m pytest tests/ -q  # model-free maths tests
```

## Use

```bash
# 1. fit the three axes (writes calibration.json)
provenance-audit calibrate --backend hf --model google/gemma-3-1b-it --device cpu

# 2. read any text in one forward
provenance-audit flag --backend hf --model google/gemma-3-1b-it \
    --text "Germany is a country. Its capital is"
#   -> {"justified": true, "reason": "familiar", ...}

# 3. audit claims externally (paraphrase stability)
provenance-audit audit --backend hf --model google/gemma-3-1b-it \
    --entities Germany,Vrandel,Qorvath --out audit.json
```

Python:

```python
from provenance_audit import calibrate_all, export, flag
from provenance_audit.adapters.hf import HFBackend

b = HFBackend("google/gemma-3-1b-it", device="cpu")
calib = export(calibrate_all(b, seeds=[...]))
print(flag(b, calib, "Vrandel is a country. Its capital is"))
# {'justified': False, 'reason': 'none', ...}
```

Any class with `encode`, `first_token`, `capture`, `generate`, `top_prob`
works as a backend — see `docs/METHOD.md`.

## Results

Axes (leave-one-out accuracy / permutation p):

| model | context | familiarity | self | one-pass flag |
| :--- | :--- | :--- | :--- | :--- |
| Argon 1B (Gemma 3 conversion) | 1.000 (p .009) | 1.000 (p .002) | 1.000 (p .025) | context / familiar / none |
| Gemma 3 1B IT | 1.000 (p .009) | 1.000 (p .002) | 0.833 | context / familiar / none |
| Gemma 4 E2B IT | 1.000 (p .009) | 1.000 (p .002) | 1.000 | context / familiar / none |
| nanoGPT / GPT-2 124M | 1.000 (p .009) | 1.000 (p .002) | 1.000 | context / familiar / none |

External audit (8 real facts, 8 invented entities with a category cue):

| model | false alarms on memorised facts | confabulations flagged | AUROC |
| :--- | :--- | :--- | :--- |
| Argon 1B | 0 / 8 | 7 / 8 | 0.938 |
| Gemma 3 1B | 0 / 8 | 8 / 8 | 1.000 |

The one miss is a *stable* confabulation (same invented capital in 3 of 4
wordings) — the registered failure mode of a stability-based audit.

## What it cannot do

| level | missing | checkable by |
| :--- | :--- | :--- |
| **U unbacked** | neither weights nor prompt support it | **this toolkit, one forward** |
| **W-false** | the weights support it, it is wrong | external only (retrieval, fact DB) |
| **C-false** | the prompt supports it, the prompt is wrong | check the source |

Two measured caveats:

- The familiarity axis sits at **layer 0**: it reads "is this a known word",
  not "do I know this fact". Sufficient for invented entities, not for rare
  real ones — there the paraphrase audit stays the ground truth.
- The internal context reader is **bound to the prompt family** it was
  calibrated on; a different wording is a different coordinate system
  (measured: 8/8 → 3/8, then 8/8 again after aligning the wording).

## Layout

```
provenance_audit/      core maths, cells, axes, audit, indicator, CLI
provenance_audit/adapters/   hf (transformers), nanogpt
tests/                 model-free checks of the measurement maths
docs/METHOD.md         minimal pairs, LOO, permutation, correction log
docs/METRICS.md        metric taxonomy and why coverage != truth
docs/PORTS.md          adapters on Gemma 3/4 and nanoGPT
results/               frozen summaries
```

## Licence

MIT.
