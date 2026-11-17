# Ports

The same three axes were ported to external codebases as read-only adapters.
No weights are modified and no training happens anywhere.

| target | repo | backend | layer list |
| :--- | :--- | :--- | :--- |
| Gemma 3 / Gemma 4 | `google-deepmind/gemma` (HF weights) | `HFBackend` | `model.layers` / `model.language_model.layers` |
| nanoGPT | `karpathy/nanoGPT` (GPT-2 124M checkpoint) | `NanoGPTBackend` | `transformer.h` |

The standalone port modules (`provenance_axes.py` in each upstream repo) are
thin wrappers around the same maths; this package is the extracted core.

## Numbers

| target | context | self | familiarity | bands | flag demo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma 3 1B IT | 1.000 (p .009) | 0.833 | 1.000 (p .002, L0) | 5/7 | context / familiar / none |
| Gemma 4 E2B IT | 1.000 (p .009) | 1.000 | 1.000 (p .002, L0) | **7/7** | context / familiar / none |
| nanoGPT / GPT-2 | 1.000 (p .009) | 1.000 | 1.000 (p .002, L3) | 5/7 | context / familiar / none |

The behavioural bands that fail on Gemma 3 and GPT-2 are checkpoint
properties, not instrument failures:

- Gemma 3 1B: context copying in the completion form degenerates for some
  codes (" is is is is"), and only 4/8 factual slots are answered confidently.
- GPT-2 124M: a salience copier (it emits the code even for the wrong box,
  0/8 relevance-selective) that does not know the facts at all
  (p(" Berlin") = 0.058 even at the naked slot).

The axes themselves separate in all three architectures with LOO 1.000, and
`cos(context, self)` is between −0.018 and −0.084 — orthogonal in a JAX
Gemma, in a multimodal Gemma 4 config and in a 12-layer nanoGPT. That is the
portable result: **context and own text are two axes, not one category.**

## Layers

The familiarity axis lands on layer 0 for both Gemma models and layer 3 of 12
for GPT-2. It reads lexical/semantic presence ("is this a known word"), not
fact coverage. For invented entities that is enough; for rare real entities
the paraphrase audit remains the ground truth.

## Reproducing

```bash
# Gemma (HF weights, local snapshots)
provenance-audit calibrate --backend hf --model <snapshot> --local --device cpu
provenance-audit flag --backend hf --model <snapshot> --local \
    --calib calibration.json --text "Vrandel is a country. Its capital is"

# nanoGPT (GPT-2 checkpoint converted from transformers, not trained)
provenance-audit calibrate --backend nanogpt --ckpt out-gpt2/ckpt.pt \
    --nanogpt-dir path/to/nanoGPT
```
