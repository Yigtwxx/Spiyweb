"""Builders -> from_layers -> propagate: the layered-hybrid claim in miniature."""

from __future__ import annotations

import pytest

from spiyweb import Graph, LayerWeights, propagate
from spiyweb.edges import ChunkRef, build_semantic_edges, build_structural_edges

# Two documents. `a1` is the seed's semantic contact; `a2` is its structurally
# adjacent neighbour but semantically orthogonal to everything - only the
# structural layer can ever reach it. `b1` (other document) is a semantic twin
# of `a1`.
CHUNKS = [
    ChunkRef(id="a1", source_id="doc-a", position=0),
    ChunkRef(id="a2", source_id="doc-a", position=1),
    ChunkRef(id="b1", source_id="doc-b", position=0),
]
IDS = ["a1", "a2", "b1"]
EMBEDDINGS = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]


def build_layered_graph(weights: LayerWeights) -> Graph:
    return Graph.from_layers(
        {
            "structural": build_structural_edges(CHUNKS),
            "semantic": build_semantic_edges(IDS, EMBEDDINGS),
        },
        weights=weights,
    )


def test_structural_layer_activates_the_semantically_invisible_neighbour() -> None:
    graph = build_layered_graph(LayerWeights())
    result = propagate(graph, {"a1": 1.0})
    # a1 forwards 6.0 across {a2: 0.3, b1: 0.5}: a2 gets 2.25, b1 gets 3.75.
    assert result.energy_of("a2") == pytest.approx(2.25), (
        "a structural hop the semantic layer alone could never produce"
    )
    assert result.energy_of("b1") == pytest.approx(3.75)


def test_builders_equal_a_hand_premerged_graph_under_propagation() -> None:
    weights = LayerWeights()
    layered = build_layered_graph(weights)
    premerged = Graph.from_edges(
        [
            ("a1", "a2", 1.0 * weights.structural),
            ("a1", "b1", 1.0 * weights.semantic),
        ]
    )
    ranked_layered = propagate(layered, {"a1": 1.0}).ranked()
    ranked_premerged = propagate(premerged, {"a1": 1.0}).ranked()
    assert ranked_layered == ranked_premerged, (
        "builders must be pure pre-processing, invisible to the core"
    )


def test_disabling_the_structural_layer_silences_the_structural_hop() -> None:
    graph = build_layered_graph(LayerWeights(structural=0.0))
    result = propagate(graph, {"a1": 1.0})
    assert result.energy_of("a2") == 0.0, (
        "with the layer ablated, the web has no path to the adjacent chunk"
    )
    assert result.energy_of("b1") > 0.0, "the semantic contact must survive"
