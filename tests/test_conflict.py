"""Negative charge (D15/D16/D26): formula, per-hop firing, surfacing, NLI.

The claim under test: contradiction is the ONE mechanism besides negative
seeds allowed to destroy energy, it fires per hop inside propagation (a
neutralised atom stops spreading), every event lands in an auditable ledger,
and the library ships the ready-made user question instead of silently
picking a winner.
"""

from __future__ import annotations

import pytest

from spiyweb import (
    ConflictConfig,
    Graph,
    NegativeEdge,
    NLIEdgeConfig,
    PropagationConfig,
    build_conflict_question,
    conflict_adjacency,
    neutralize,
    propagate,
    propagate_colored,
    retrieve,
)
from spiyweb.edges.nli import build_nli_edges
from spiyweb.questions import KEEP_BOTH_OPTION_INDEX

# ---------------------------------------------------------------- formula


def test_neutralize_matches_the_owner_worked_example() -> None:
    # E_a=5.0, E_b=2.0, s=0.9, k=1.0 -> absorbed 1.8 each, 3.6 destroyed.
    new_a, new_b, absorbed = neutralize(5.0, 2.0, 0.9, 1.0)
    assert new_a == pytest.approx(3.2)
    assert new_b == pytest.approx(0.2)
    assert absorbed == pytest.approx(1.8)


def test_neutralize_never_pushes_an_energy_negative() -> None:
    new_a, new_b, absorbed = neutralize(5.0, 2.0, 1.0, 1.0)
    assert absorbed == pytest.approx(2.0), "absorption is capped by the weaker side"
    assert new_a == pytest.approx(3.0)
    assert new_b == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("source", "target", "strength"),
    [
        ("a", "a", 0.5),
        ("", "b", 0.5),
        ("a", "b", 0.0),
        ("a", "b", 1.2),
    ],
)
def test_negative_edge_rejects_broken_inputs(
    source: str, target: str, strength: float
) -> None:
    with pytest.raises(ValueError):
        NegativeEdge(source=source, target=target, strength=strength)


def test_conflict_adjacency_is_symmetric_and_keeps_strongest_evidence() -> None:
    adjacency = conflict_adjacency(
        [
            NegativeEdge("a", "b", 0.6),
            NegativeEdge("b", "a", 0.9),  # same pair, stronger evidence
        ]
    )
    assert adjacency["a"]["b"] == pytest.approx(0.9)
    assert adjacency["b"]["a"] == pytest.approx(0.9)


@pytest.mark.parametrize("value", [0.0, 1.5])
def test_conflict_config_rejects_out_of_range_coefficient(value: float) -> None:
    with pytest.raises(ValueError, match="coefficient"):
        ConflictConfig(coefficient=value)


@pytest.mark.parametrize("value", [0.0, 1.1])
def test_nli_config_rejects_out_of_range_threshold(value: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        NLIEdgeConfig(contradiction_threshold=value)


# ---------------------------------------------------------------- propagation

NEGATIVE_PN = conflict_adjacency([NegativeEdge("P", "N", 1.0)])


def test_opposing_seeds_neutralise_and_the_ledger_records_it() -> None:
    graph = Graph.from_edges([("P", "X", 0.8)])
    result = propagate(
        graph,
        {"P": 0.9, "N": 0.7},  # P = 5.625, N = 4.375
        PropagationConfig(),
        negative=NEGATIVE_PN,
        conflict=ConflictConfig(),
    )
    assert result.energy_of("P") == pytest.approx(1.25)
    assert result.energy_of("N") == pytest.approx(0.0)
    (record,) = result.conflicts
    assert (record.node_a, record.node_b) == ("N", "P")
    assert record.hop == 0
    assert record.absorbed_each == pytest.approx(4.375)
    assert record.absorbed_total == pytest.approx(8.75), (
        "the ledger's destroyed total is exactly what both sides lost"
    )


def test_a_neutralised_atom_stops_spreading() -> None:
    # Without conflict, P forwards 5.625 * .6 = 3.375 to X. With the full
    # neutralisation P drops to 1.25 < threshold 1.5 and X must never light.
    graph = Graph.from_edges([("P", "X", 0.8)])
    seeds = {"P": 0.9, "N": 0.7}
    plain = propagate(graph, seeds, PropagationConfig())
    assert plain.energy_of("X") == pytest.approx(3.375)

    damped = propagate(
        graph,
        seeds,
        PropagationConfig(),
        negative=NEGATIVE_PN,
        conflict=ConflictConfig(),
    )
    assert "X" not in damped.activations, (
        "per-hop application means the damped atom loses its next hop too"
    )


def test_a_conflict_pair_fires_at_most_once_per_run() -> None:
    # A and B damp each other at hop 0 (strength .5 -> 2.5 left each); the
    # run continues for another hop, but the settled pair must not re-fire.
    graph = Graph.from_edges([("A", "X", 1.0)])
    result = propagate(
        graph,
        {"A": 0.5, "B": 0.5},
        PropagationConfig(),
        negative=conflict_adjacency([NegativeEdge("A", "B", 0.5)]),
        conflict=ConflictConfig(),
    )
    assert len(result.conflicts) == 1
    assert result.energy_of("X") == pytest.approx(1.5), (
        "A spreads its REMAINING 2.5 * .6 after the hop-0 neutralisation"
    )


def test_a_later_hop_conflict_fires_with_the_hop_recorded() -> None:
    # H activates at hop 1 (T's neighbour) and only then meets its opponent S.
    graph = Graph.from_edges([("T", "H", 1.0), ("H", "H2", 1.0)])
    result = propagate(
        graph,
        {"S": 0.5, "T": 0.5},  # 5.0 each; H arrives with 3.0 at hop 1
        PropagationConfig(),
        negative=conflict_adjacency([NegativeEdge("S", "H", 1.0)]),
        conflict=ConflictConfig(),
    )
    (record,) = result.conflicts
    assert record.hop == 1
    assert record.energy_a_before == pytest.approx(3.0)  # H
    assert record.energy_b_before == pytest.approx(5.0)  # S
    assert result.energy_of("H") == pytest.approx(0.0)
    assert result.energy_of("S") == pytest.approx(2.0)
    assert "H2" not in result.activations, "the dead atom must not chain onward"


def test_an_unrelated_conflict_leaves_untouched_seeds_spreading() -> None:
    # B enters below the threshold (seeds are never threshold-checked); an
    # enabled conflict pass that fired NOTHING must not steal its spread.
    graph = Graph.from_edges([("A", "Y", 1.0), ("B", "Y", 1.0)])
    seeds = {"A": 0.9, "B": 0.1}  # A = 9.0, B = 1.0 < threshold 1.5
    unrelated = conflict_adjacency([NegativeEdge("C", "D", 1.0)])
    with_conflict = propagate(
        graph,
        seeds,
        PropagationConfig(),
        negative=unrelated,
        conflict=ConflictConfig(),
    )
    plain = propagate(graph, seeds, PropagationConfig())
    assert with_conflict.conflicts == ()
    assert with_conflict.energy_of("Y") == pytest.approx(plain.energy_of("Y"))
    assert with_conflict.energy_of("Y") == pytest.approx(6.0)  # 5.4 + 0.6


def test_disabled_config_behaves_exactly_as_no_negative_edges() -> None:
    graph = Graph.from_edges([("P", "X", 0.8)])
    seeds = {"P": 0.9, "N": 0.7}
    ablated = propagate(
        graph,
        seeds,
        PropagationConfig(),
        negative=NEGATIVE_PN,
        conflict=ConflictConfig(enabled=False),
    )
    plain = propagate(graph, seeds, PropagationConfig())
    assert ablated.activations == plain.activations
    assert ablated.conflicts == ()


def test_colors_never_mix_but_neutralise_within_a_color() -> None:
    graph = Graph.from_edges([("P", "X", 0.2)])
    # Opponents in DIFFERENT colours never meet: no conflict may fire.
    separated = propagate_colored(
        graph,
        {"c0": {"P": 1.0}, "c1": {"N": 1.0}},
        PropagationConfig(),
        negative=NEGATIVE_PN,
        conflict=ConflictConfig(),
    )
    assert all(result.conflicts == () for result in separated.per_color.values())

    # Both opponents inside ONE colour neutralise as in a plain run.
    together = propagate_colored(
        graph,
        {"c0": {"P": 0.5, "N": 0.5}},
        PropagationConfig(),
        negative=NEGATIVE_PN,
        conflict=ConflictConfig(),
    )
    assert len(together.per_color["c0"].conflicts) == 1


# ---------------------------------------------------------------- retrieve


class FakeIndex:
    def __init__(self, contacts: list[tuple[str, float]]) -> None:
        self._contacts = contacts

    def search(self, query: list[float], k: int) -> list[tuple[str, float]]:
        return self._contacts[:k]


def test_retrieve_surfaces_conflicts_and_flags_only_survivors_disputed() -> None:
    graph = Graph.from_edges([("P", "X", 0.2)])
    result = retrieve(
        [1.0, 0.0],
        FakeIndex([("P", 0.9), ("N", 0.7)]),
        graph,
        negative=NEGATIVE_PN,
        conflict=ConflictConfig(),
    )
    assert len(result.conflicts) == 1
    assert result.disputed == frozenset({"P"}), (
        "the side neutralised to zero is gone from the ranking and carries "
        "nothing to flag"
    )


# ---------------------------------------------------------------- questions


def test_conflict_question_is_template_built_with_label_fallback() -> None:
    graph = Graph.from_edges([("P", "X", 0.2)])
    result = retrieve(
        [1.0, 0.0],
        FakeIndex([("P", 0.9), ("N", 0.7)]),
        graph,
        negative=conflict_adjacency([NegativeEdge("P", "N", 0.5)]),
        conflict=ConflictConfig(),
    )
    (record,) = result.conflicts
    question = build_conflict_question(record, labels={"P": "Doc Plus"})
    assert "Doc Plus" in question.text
    assert "N" in question.text, "a node without a label falls back to its id"
    assert len(question.options) == 3
    assert question.options[KEEP_BOTH_OPTION_INDEX] == ("Keep both, marked as disputed")
    assert question.record is record


# ---------------------------------------------------------------- NLI edges


class FakeNLI:
    """Scripted contradiction scores keyed by (premise, hypothesis)."""

    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self._scores = scores

    def contradiction_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._scores.get(pair, 0.0) for pair in pairs]


TEXTS = {"n1": "the sky is blue", "n2": "the sky is not blue", "n3": "grass"}


def test_nli_builder_takes_the_max_over_both_directions() -> None:
    model = FakeNLI(
        {
            (TEXTS["n1"], TEXTS["n2"]): 0.3,
            (TEXTS["n2"], TEXTS["n1"]): 0.95,  # found only in this direction
        }
    )
    edges = build_nli_edges([("n2", "n1")], TEXTS, model)
    (edge,) = edges
    assert (edge.source, edge.target) == ("n1", "n2"), "ids come out sorted"
    assert edge.strength == pytest.approx(0.95)


def test_nli_builder_threshold_is_inclusive_and_filters_below() -> None:
    model = FakeNLI(
        {
            (TEXTS["n1"], TEXTS["n2"]): 0.9,
            (TEXTS["n1"], TEXTS["n3"]): 0.89,
        }
    )
    edges = build_nli_edges(
        [("n1", "n2"), ("n1", "n3")],
        TEXTS,
        model,
        NLIEdgeConfig(contradiction_threshold=0.9),
    )
    assert [(edge.source, edge.target) for edge in edges] == [("n1", "n2")]


def test_nli_builder_rejects_broken_candidates_loudly() -> None:
    model = FakeNLI({})
    with pytest.raises(ValueError, match="self-pair"):
        build_nli_edges([("n1", "n1")], TEXTS, model)
    with pytest.raises(ValueError, match="no text"):
        build_nli_edges([("n1", "missing")], TEXTS, model)


def test_nli_builder_rejects_a_model_returning_the_wrong_score_count() -> None:
    class ShortModel:
        def contradiction_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.5]

    with pytest.raises(ValueError, match="scores"):
        build_nli_edges([("n1", "n2")], TEXTS, ShortModel())
