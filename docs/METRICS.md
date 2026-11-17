# Metrics: what can be measured, and what cannot

The definition in use:

> a hallucination is what training does not justify **and** what the prompt
> (including its projection onto training) does not provide.

That is a **provenance** statement. It has a consequence that must be stated
openly: a **memorised falsehood** is justified by training and is *not* a
hallucination under this definition. "Hallucination-free" therefore means
"coverage-free", not "true".

## Three levels, three different checks

| level | what is missing | checkable by |
| :--- | :--- | :--- |
| **U unbacked** | neither weights nor prompt support the claim | counterfactual provenance audit, no external knowledge needed |
| **W-false** | the weights support it, it is wrong | external only (retrieval, fact DB, NLI) |
| **C-false** | the prompt supports it, the prompt is wrong | check the source itself |

No internal signal can decide level 2 or 3. The provenance audit is the
*necessary* condition, not the sufficient one.

## Candidates, with their failure mode

1. **Provenance coverage (this package).**
   `H = 1 − max(stability over wordings, evidence match)`.
   Cheap, no labels, no second model.
   *Failure:* **stable confabulation** — the same invented answer in every
   wording. Measured at 1/8 on the reference model; that is why the audit
   keeps it as a separate, flagged case.

2. **Counterfactual entity dependence.**
   Swap the subject; a covered claim moves with it.
   *Failure:* invented entities are also subject-dependent (each one yields a
   different invention), so this alone is useless — it is a control, not a
   metric.

3. **Evidence attribution (retrieval / NLI).**
   Is the claim entailed by the context? The strongest form of the context
   axis.
   *Failure:* needs a second model, i.e. a second error source.

4. **Self-consistency under sampling.**
   *Failure:* measures confidence, not coverage; confounded with temperature.

5. **Calibration error (ECE, logit gap).**
   *Failure:* measures truth-aptitude, not provenance — exactly the
   confusion the definition avoids.

6. **Internal provenance readout (this package).**
   One forward, position on the axes.
   *Failure:* the axes are marker-borne; they read "where does the state come
   from", not "is the claim covered". They only become a metric once
   validated against an external audit.

## What the numbers say

| band | Argon 1B | Gemma 3 1B |
| :--- | :--- | :--- |
| false alarms on memorised facts | 0 / 8 | 0 / 8 |
| confabulations flagged | 7 / 8 | 8 / 8 |
| context coverage (behavioural / internal) | 8/8 / 8/8 | 8/8 / 8/8 |
| AUROC over claims | 0.938 | 1.000 |

The single miss is the registered failure mode: an invented country answered
with the same capital in 3 of 4 wordings.
