# Method

## 1. What is measured

The residual stream is a bus, not a carrier: the same token goes through the
same embedding matrix whether a user or the model wrote it. There is no
provenance bit. What can be separated is

1. **counterfactual dependence** — what happens if the evidence is removed,
2. **a linear direction** — decodable, possibly causal,
3. **a behavioural policy** — does the model treat its own text as its own?

This toolkit measures (1) and (2) and reports (3) as the external audit.

## 2. Minimal pairs

Every claim in the toolkit rests on a pair in which **only the source**
changes. Attention is causal, so anything after a token cannot influence it;
that rules out "last turn vs middle turn" as a factor and leaves only
what precedes.

| axis | pair | what is held constant |
| :--- | :--- | :--- |
| context | C vs I | same code word, same question, same length; the box number in the evidence differs by one token |
| familiarity | F+ vs F− | identical template; only the entity name differs |
| self | model vs user | identical token ids after the marker |

The **I arm is the crucial control**: it also contains a fact-shaped
sentence, so a reader that only sees "there is evidence here" will fail to
distinguish it from C. The C-vs-I contrast therefore measures *relevance*,
not surface form.

## 3. Fitting

- `d = unit(mean(H_pos) − mean(H_neg))` per layer.
- Decision rule: LDA, i.e. the threshold is the **midpoint of the two
  projected means**. A threshold at 0 is wrong: residual streams carry a
  large common offset, a 0-cut makes the accuracy bimodal and every
  permutation test meaningless (we hit this; see the correction log).
- Validation: leave-one-out (direction and threshold from n−1 items), then
  a **label permutation** test. Only the labels are shuffled, never rows and
  labels together — that is the identity and yields p = 1.0 always (we hit
  this too).
- Significance is reported with a max-statistic over layers when a layer
  search is involved (look-elsewhere correction).

## 4. The external audit

```
stability(e) = share of wordings that assert the same content
H(e)         = 1 − stability(e)   if the model asserts anything, else 0
```

Silence is not a hallucination. This is the ground truth against which the
one-pass indicator is validated, and it is where the honest failure mode
lives: a **stable** confabulation is scored as covered.

## 5. The indicator

```
justified = context-axis positive  OR  familiarity-axis positive
```

One forward, no generation, no second model.

## 6. Correction log (bugs that were measured, not guessed)

| # | bug | symptom | fix |
| :--- | :--- | :--- | :--- |
| 1 | permutation shuffled rows *and* labels together | p always 1.0 | shuffle labels only |
| 2 | threshold at 0 | accuracy bimodal (0.0/1.0), null test worthless | LDA midpoint |
| 3 | 16 cells × 26 layers | look-elsewhere correction swallows the signal | 20 + 20 cells |
| 4 | missing BOS | p(" Paris") = 0.0000 vs 0.4873 with BOS | prepend BOS |
| 5 | evidence arm used counterfactual *real* facts | single contradicting sentence does not override weights (0/6) | invented facts / protocol codes |
| 6 | claim read at token 1 | the claim sits 5–7 tokens deep ("The capital of France is **Paris**.") | generate, then extract |
| 7 | internal context reader applied to a different wording | 3/8 instead of 8/8 | calibrate per prompt family |
| 8 | 3-class nearest centroid for the flag | flipped on one model (classes W/C/S overlap in the second axis) | 1-D rule on the context axis, threshold = W/C midpoint |

## 7. Bands used in development

Frozen before each run, never widened afterwards:

- axis: LOO accuracy ≥ 0.75, permutation p < 0.05
- audit: no false alarm on memorised facts (≥ 6/8 stable), confabulation
  flagged (≥ 5/8 unstable or silent), AUROC over claims ≥ 0.80
- purity: bit-identical weights before/after, zero hooks left, deterministic
  double run
