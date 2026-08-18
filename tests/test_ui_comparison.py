"""The side-by-side panel: the web against plain top-k, cut at the same k."""

from __future__ import annotations

import pytest

from graph_view import build_comparison
from spiyweb.core.graph import Graph, Node
from spiyweb.core.propagate import Activation

_WEB: list[tuple[str, float]] = [("a", 5.0), ("d", 2.9), ("b", 2.2)]
_BASELINE = ["a", "x", "y"]


def _graph() -> Graph:
    nodes = [
        Node(id=node, layer="chunk", source_id=f"doc_{node}", length=50)
        for node in ("a", "b", "d", "x", "y")
    ]
    return Graph.from_edges([("a", "b", 1.0)], nodes=nodes)


def _activations() -> dict[str, Activation]:
    return {
        "a": Activation(energy=5.0, hop=0, contributors=()),
        "d": Activation(energy=2.9, hop=2, contributors=("a", "b")),
        "b": Activation(energy=2.2, hop=1, contributors=("a",)),
    }


def _comparison(**overrides: object) -> object:
    defaults: dict[str, object] = {
        "web_ranked": _WEB,
        "baseline_ids": _BASELINE,
        "k": 3,
        "activations": _activations(),
        "graph": _graph(),
    }
    defaults.update(overrides)
    return build_comparison(**defaults)  # type: ignore[arg-type]


def test_ranks_are_one_based_and_ordered() -> None:
    comparison = _comparison()
    assert [row.rank for row in comparison.web] == [1, 2, 3]
    assert [row.node_id for row in comparison.web] == ["a", "d", "b"]


def test_set_differences_are_the_novelty_the_user_can_see() -> None:
    comparison = _comparison()
    assert comparison.only_in_web == ("d", "b")
    assert comparison.only_in_baseline == ("x", "y")
    assert comparison.overlap == ("a",)


def test_both_columns_are_cut_at_the_same_k() -> None:
    """Context-budget parity: the web may not win by returning a longer list."""
    comparison = _comparison(k=2)
    assert len(comparison.web) == 2
    assert len(comparison.baseline) == 2


def test_in_other_marks_the_shared_passages() -> None:
    comparison = _comparison()
    assert comparison.web[0].in_other is True
    assert comparison.web[1].in_other is False
    assert comparison.baseline[0].in_other is True


def test_hop_is_a_web_only_column() -> None:
    comparison = _comparison()
    assert comparison.web[1].hop == 2
    assert all(row.hop is None for row in comparison.baseline)


def test_votes_come_from_the_merged_two_stage_ledger() -> None:
    """`RetrievalResult.votes()` merges contact and propagation suppression."""
    comparison = _comparison(votes={"doc_d": 3})
    assert comparison.web[1].votes == 3
    assert "voted x3" in comparison.web[1].badges


def test_badges_report_seed_bridge_and_dispute() -> None:
    comparison = _comparison(seeds=("a",), bridges=("d",), disputed=("b",))
    badges = {row.node_id: row.badges for row in comparison.web}
    assert badges["a"] == ("seed",)
    assert badges["d"] == ("bridge",)
    assert badges["b"] == ("disputed",)


def test_missing_texts_fall_back_to_ids_instead_of_raising() -> None:
    comparison = _comparison()
    assert comparison.web[0].title == "a"
    assert comparison.web[0].snippet == ""


def test_titles_and_snippets_are_used_when_present() -> None:
    comparison = _comparison(
        titles={"a": "Marie Curie"},
        texts={"a": "  Marie   Curie was a physicist.  "},
    )
    assert comparison.web[0].title == "Marie Curie"
    assert comparison.web[0].snippet == "Marie Curie was a physicist."


def test_snippet_is_ellipsised() -> None:
    comparison = _comparison(texts={"a": "x" * 500}, snippet_chars=10)
    assert comparison.web[0].snippet.endswith("…")
    assert len(comparison.web[0].snippet) == 10


def test_contact_tau_travels_into_the_view_model() -> None:
    """The design requires the computed duplicate cut to be visible."""
    comparison = _comparison(contact_tau=0.962, dedup_enabled=True)
    assert comparison.contact_tau == pytest.approx(0.962)
    assert comparison.dedup_enabled is True


def test_dedup_off_reports_a_missing_tau_explicitly() -> None:
    comparison = _comparison()
    assert comparison.contact_tau is None
    assert comparison.dedup_enabled is False


def test_baseline_scores_are_carried_when_supplied() -> None:
    comparison = _comparison(baseline_scores={"a": 0.83})
    assert comparison.baseline[0].score == pytest.approx(0.83)


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _comparison(k=0)
