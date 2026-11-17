# Frozen results

`summary.json` holds the numbers quoted in the top-level README: axis
accuracies, permutation p-values, audit counts and the known failure mode.

They come from the development devices on a CPU with float32 weights. The
per-run raw files (cell text, claims, stability per entity) are kept with the
devices; this folder is the stable subset that the documentation cites.

Nothing here is a benchmark score. It is a measurement of whether a linear
readout separates provenance at all, and how often the external audit agrees.
