"""Real-NLI wrapper logic: label resolution, batching, backend contract.

The claim under test: `contradiction_label_index` locates the contradiction
class from the model's OWN `id2label` (a hardcoded index would silently score
entailment as contradiction on a differently-ordered head), `batched` splits
without losing or reordering pairs, and `TransformersNLIModel` is a faithful
pass-through over any injected backend - the heavy transformers path never
runs in CI.
"""

from __future__ import annotations

import pytest

from spiyweb import TransformersNLIModel, contradiction_label_index
from spiyweb.nli import batched


def test_contradiction_label_index_reads_the_models_own_mapping() -> None:
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    assert contradiction_label_index(id2label) == 2, (
        "the standard XNLI head puts contradiction at 2"
    )


def test_contradiction_label_index_is_case_and_prefix_tolerant() -> None:
    id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    assert contradiction_label_index(id2label) == 0, (
        "checkpoints disagree on label case and order; the mapping decides"
    )


def test_contradiction_label_index_rejects_a_headless_mapping() -> None:
    with pytest.raises(ValueError, match="contradiction"):
        contradiction_label_index({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_contradiction_label_index_rejects_an_ambiguous_mapping() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        contradiction_label_index({0: "contradiction", 1: "contradiction_b"})


def test_batched_splits_without_losing_or_reordering() -> None:
    pairs = [(str(i), str(i)) for i in range(5)]
    batches = batched(pairs, 2)
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [pair for batch in batches for pair in batch] == pairs, (
        "scores are matched to pairs by position; order must survive batching"
    )


def test_batched_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        batched([("a", "b")], 0)


class FakeScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return self.scores[: len(pairs)]


def test_wrapper_passes_scores_through_unchanged() -> None:
    scorer = FakeScorer([0.9, 0.1])
    model = TransformersNLIModel(scorer=scorer)
    pairs = [("wet", "dry"), ("wet", "wet")]
    assert list(model.contradiction_scores(pairs)) == [0.9, 0.1]
    assert scorer.calls == [pairs], "one backend call, pairs untouched"


def test_wrapper_short_circuits_on_no_pairs() -> None:
    scorer = FakeScorer([])
    model = TransformersNLIModel(scorer=scorer)
    assert list(model.contradiction_scores([])) == []
    assert scorer.calls == [], "an empty candidate set must not touch the model"


def test_wrapper_rejects_a_miscounting_backend() -> None:
    model = TransformersNLIModel(scorer=FakeScorer([0.5]))
    with pytest.raises(ValueError, match="1 scores for 2 pairs"):
        model.contradiction_scores([("a", "b"), ("c", "d")])
