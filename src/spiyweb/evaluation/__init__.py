"""The Phase 1 product: the MuSiQue evaluation harness.

This package holds the dataset loader, the two baselines (`top-k` and
IRCoT-style iterative retrieval), the metrics of the weighted objective and
the index/evaluate/report pipeline. It later becomes the regression suite -
it is never throwaway code (boundary rule 3).

Deliberately NO eager imports here: the harness needs faiss/numpy at runtime,
while `import spiyweb` (and `spiyweb.evaluation` as a namespace) must keep
working with zero dependencies. Import the submodules directly.
"""
