"""Hand-traced values for every metric of the Phase 1 objective.

The claim under test: the exact formulas the owner fixed on 2026-08-14.
Support recall and Novelty share the |gold| denominator (that shared scale IS
the 65/35 normalisation rule), and Novelty subtracts the reference's ENTIRE
top-k - a document the dense baseline already returned is not novel, however
it was ranked there.
"""

from __future__ import annotations

import pytest

from spiyweb.config import EvaluationConfig
from spiyweb.evaluation.metrics import (
    bridge_recall_at_k,
    novelty_at_k,
    passages_at_k,
    support_recall_at_k,
    weighted_objective,
)

GOLD = {"g1", "g2", "g3"}


def test_a_chunk_only_ranking_is_returned_unchanged() -> None:
    """The fold must not move a single existing number."""
    assert passages_at_k(["a", "b", "c", "d"], k=3) == ["a", "b", "c"]
    assert passages_at_k(["a", "b"], k=5) == ["a", "b"]


def test_propositions_are_scored_through_their_parent_passage() -> None:
    """Gold is per passage; a proposition id can never match it directly.

    Without the fold this ranking scores 0.0 against gold made of chunk ids,
    which reads as a broken retriever instead of a unit mismatch.
    """
    ranked = ["g1#p2", "g1#p5", "x#p0", "g2"]
    assert passages_at_k(ranked, k=3) == ["g1", "x", "g2"], (
        "two propositions of one passage take ONE slot, not two"
    )
    assert support_recall_at_k(ranked, GOLD, k=3) == pytest.approx(2 / 3)
    assert bridge_recall_at_k(ranked, {"g1"}, k=1) == pytest.approx(1.0)


def test_novelty_folds_both_sides_so_the_reference_stays_comparable() -> None:
    """A passage the dense reference already surfaced is not novel.

    Folding only the system's side would count `g1` as novel whenever the
    reference reached it through the chunk and the web through a proposition.
    """
    ranked = ["g1#p0", "g2"]
    reference = ["g1", "z"]
    assert novelty_at_k(ranked, reference, GOLD, k=2) == pytest.approx(1 / 3)


def test_support_recall_counts_gold_hits_within_the_cutoff() -> None:
    retrieved = ["g1", "x", "g2", "g3"]
    assert support_recall_at_k(retrieved, GOLD, k=3) == pytest.approx(2 / 3)
    assert support_recall_at_k(retrieved, GOLD, k=4) == pytest.approx(1.0)
    assert support_recall_at_k(["x", "y"], GOLD, k=2) == pytest.approx(0.0)


def test_novelty_subtracts_the_references_entire_top_k() -> None:
    web = ["g1", "g2", "x"]
    reference = ["g1", "y", "z"]
    # g1 is in the reference's top-k, so only g2 is novel: 1/3.
    assert novelty_at_k(web, reference, GOLD, k=3) == pytest.approx(1 / 3)

    # g2 ranked LAST in the reference is still returned by it - not novel.
    reference_with_late_g2 = ["y", "z", "g2"]
    assert novelty_at_k(web, reference_with_late_g2, GOLD, k=3) == pytest.approx(
        1 / 3
    ), (
        "the reference's whole top-k is subtracted, not just its gold hits - "
        "a document the baseline already showed the reader is not novel"
    )


def test_the_reference_is_never_novel_against_itself() -> None:
    reference = ["g1", "g2", "x"]
    assert novelty_at_k(reference, reference, GOLD, k=3) == pytest.approx(0.0)


def test_novelty_respects_the_cutoff_on_both_sides() -> None:
    web = ["x", "g1"]
    reference = ["g1", "y"]
    # At k=1 the web finds nothing gold: novelty 0 despite g1 at rank 2.
    assert novelty_at_k(web, reference, GOLD, k=1) == pytest.approx(0.0)
    # At k=2 the web finds g1, but the reference also returned it: still 0.
    assert novelty_at_k(web, reference, GOLD, k=2) == pytest.approx(0.0)


def test_bridge_recall_scores_only_the_intermediate_documents() -> None:
    bridge_gold = {"g1", "g2"}
    retrieved = ["g3", "g1", "x"]
    # g3 is gold but NOT bridge gold - it must not count here.
    assert bridge_recall_at_k(retrieved, bridge_gold, k=3) == pytest.approx(1 / 2)


def test_the_weighted_objective_is_a_plain_65_35_sum() -> None:
    assert weighted_objective(0.8, 0.2) == pytest.approx(0.65 * 0.8 + 0.35 * 0.2)
    assert weighted_objective(1.0, 1.0) == pytest.approx(1.0)
    assert weighted_objective(0.0, 0.0) == pytest.approx(0.0)


def test_the_weights_come_from_config_not_from_code() -> None:
    config = EvaluationConfig(accuracy_weight=0.5, novelty_weight=0.5)
    assert weighted_objective(0.6, 0.2, config) == pytest.approx(0.4)


def test_empty_gold_is_loader_corruption_not_a_zero() -> None:
    with pytest.raises(ValueError, match="gold"):
        support_recall_at_k(["x"], set(), k=1)
    with pytest.raises(ValueError, match="gold"):
        novelty_at_k(["x"], ["y"], set(), k=1)
    with pytest.raises(ValueError, match="bridge_gold"):
        bridge_recall_at_k(["x"], set(), k=1)


def test_out_of_range_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="k must be"):
        support_recall_at_k(["x"], GOLD, k=0)
    with pytest.raises(ValueError, match="recall"):
        weighted_objective(1.5, 0.0)
    with pytest.raises(ValueError, match="novelty"):
        weighted_objective(0.5, -0.1)


def test_evaluation_config_validation_guards_the_experiment_identity() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        EvaluationConfig(accuracy_weight=0.9, novelty_weight=0.3)
    with pytest.raises(ValueError, match="ascending"):
        EvaluationConfig(k_values=(5, 2))
    with pytest.raises(ValueError, match="ascending"):
        EvaluationConfig(k_values=(2, 2, 5))
    with pytest.raises(ValueError, match="at least 1"):
        EvaluationConfig(k_values=(0, 2))
    with pytest.raises(ValueError, match="sample_size"):
        EvaluationConfig(sample_size=-1)
