"""Multi-layer edge merging: weighted sum, layer disabling, registry carriage."""

from __future__ import annotations

import pytest

from spiyweb import Graph, LayerWeights, Node, propagate


def test_from_layers_sums_weighted_contributions_across_layers() -> None:
    graph = Graph.from_layers(
        {
            "semantic": [("A", "B", 0.8)],
            "entity": [("A", "B", 0.5)],
        }
    )
    # 0.5 * 0.8 + 1.0 * 0.5 with the Phase 1 default weights.
    assert graph.neighbors("A")["B"] == pytest.approx(0.9)


def test_from_layers_edge_in_one_layer_keeps_that_layers_scaled_weight() -> None:
    graph = Graph.from_layers(
        {
            "semantic": [("A", "B", 0.8)],
            "structural": [("A", "C", 1.0)],
        }
    )
    assert graph.neighbors("A")["B"] == pytest.approx(0.4), "no averaging over layers"
    assert graph.neighbors("A")["C"] == pytest.approx(0.3)


def test_from_layers_zero_layer_weight_leaves_no_trace_of_the_layer() -> None:
    graph = Graph.from_layers(
        {
            "entity": [("A", "B", 1.0)],
            "learned": [("A", "L", 0.9), ("L", "M", 0.9)],
        }
    )
    assert graph.nodes == {"A", "B"}, "disabled layer must not even add its nodes"


def test_from_layers_zero_edge_weight_stays_suppressed() -> None:
    graph = Graph.from_layers({"entity": [("A", "A_dup", 0.0)]})
    assert "A_dup" in graph.nodes
    assert graph.neighbors("A")["A_dup"] == 0.0


def test_from_layers_other_layer_can_revive_a_suppressed_edge() -> None:
    graph = Graph.from_layers(
        {
            "semantic": [("A", "B", 0.0)],
            "entity": [("A", "B", 0.4)],
        }
    )
    assert graph.neighbors("A")["B"] == pytest.approx(0.4)


def test_from_layers_negative_edge_weight_raises_value_error() -> None:
    with pytest.raises(ValueError, match="negative charge"):
        Graph.from_layers({"entity": [("A", "B", -0.2)]})


def test_from_layers_undirected_mirrors_both_directions() -> None:
    graph = Graph.from_layers({"entity": [("A", "B", 0.6)]})
    assert graph.neighbors("B")["A"] == pytest.approx(0.6)


def test_from_layers_directed_stays_one_way() -> None:
    graph = Graph.from_layers({"entity": [("A", "B", 0.6)]}, undirected=False)
    assert graph.neighbors("B") == {}


def test_from_layers_equals_premerged_from_edges_under_propagation() -> None:
    weights = LayerWeights()
    layered = Graph.from_layers(
        {
            "semantic": [("Q", "A", 0.9), ("A", "B", 0.8)],
            "entity": [("A", "B", 0.5), ("B", "C", 0.7)],
            "structural": [("Q", "A", 1.0)],
        },
        weights=weights,
    )
    premerged = Graph.from_edges(
        [
            ("Q", "A", 0.9 * weights.semantic + 1.0 * weights.structural),
            ("A", "B", 0.8 * weights.semantic + 0.5 * weights.entity),
            ("B", "C", 0.7 * weights.entity),
        ]
    )
    ranked_layered = propagate(layered, {"Q": 1.0}).ranked()
    ranked_premerged = propagate(premerged, {"Q": 1.0}).ranked()
    assert ranked_layered == ranked_premerged, (
        "merging must be pure pre-processing, invisible to the propagation"
    )


def test_graph_carries_optional_node_registry() -> None:
    node = Node(id="A", layer="chunk", source_id="doc-1", length=100)
    graph = Graph.from_edges([("A", "B", 0.5)], nodes=[node])
    assert graph.node("A") is node
    assert graph.node("B") is None
    assert graph.nodes == {"A", "B"}, "the id set property must stay metadata-free"


def test_from_layers_carries_the_node_registry_too() -> None:
    node = Node(id="A", layer="proposition", source_id="doc-1", length=12)
    graph = Graph.from_layers({"entity": [("A", "B", 0.5)]}, nodes=[node])
    assert graph.node("A") is node


def test_node_registry_duplicate_ids_raise_value_error() -> None:
    nodes = [
        Node(id="A", layer="chunk", source_id="doc-1", length=100),
        Node(id="A", layer="proposition", source_id="doc-2", length=10),
    ]
    with pytest.raises(ValueError, match="duplicate node id"):
        Graph.from_edges([("A", "B", 0.5)], nodes=nodes)
