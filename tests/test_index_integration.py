"""Chunker + three edge layers + propagation: the entity hop in miniature.

The scenario the whole project exists for, at its smallest: the answer chunk
lives in ANOTHER document, is semantically invisible from the seed (orthogonal
embedding), and is reachable only because it shares one rare entity with the
seed's structural neighbour. Semantic alone can never produce that link -
entity edges are the main hop fuel.
"""

from __future__ import annotations

import pytest

from spiyweb import EntityEdgeConfig, Graph, LayerWeights, propagate
from spiyweb.edges import (
    build_entity_edges,
    build_semantic_edges,
    build_structural_edges,
)
from spiyweb.nodes import DocumentInput, TextUnit, chunk_documents

DOCUMENTS = [
    DocumentInput(
        source_id="doc-a",
        units=(
            TextUnit("the seed topic, as the query phrases it"),
            TextUnit("same topic, and it mentions the Wardenclyffe tower"),
        ),
    ),
    DocumentInput(
        source_id="doc-b",
        units=(
            TextUnit("the answer: what happened at Wardenclyffe"),
            TextUnit("unrelated trailing paragraph of doc-b"),
        ),
    ),
]

# doc-a chunks point one way, doc-b chunks the other: the seed's cosine
# neighbourhood NEVER crosses the document boundary.
EMBEDDINGS = {
    "doc-a:0": [1.0, 0.0],
    "doc-a:1": [1.0, 0.0],
    "doc-b:0": [0.0, 1.0],
    "doc-b:1": [0.0, 1.0],
}

# Prepared extraction output (the extractor's own tests live elsewhere): one
# rare entity shared across the boundary, df=2 over 4 chunks.
ENTITIES: dict[str, list[str]] = {
    "doc-a:0": [],
    "doc-a:1": ["wardenclyffe"],
    "doc-b:0": ["wardenclyffe"],
    "doc-b:1": [],
}


def build_web(weights: LayerWeights) -> Graph:
    chunks = chunk_documents(DOCUMENTS)
    ids = [chunk.node.id for chunk in chunks]
    return Graph.from_layers(
        {
            "structural": build_structural_edges([chunk.ref for chunk in chunks]),
            "semantic": build_semantic_edges(
                ids, [EMBEDDINGS[chunk_id] for chunk_id in ids]
            ),
            # Ratio 1.0 disables the df guard: the measured 0.02 default
            # assumes a corpus of thousands, not this 4-chunk miniature.
            "entity": build_entity_edges(
                ENTITIES, config=EntityEdgeConfig(max_df_ratio=1.0)
            ),
        },
        weights=weights,
        nodes=[chunk.node for chunk in chunks],
    )


def test_the_cross_document_answer_activates_through_the_entity_hop() -> None:
    result = propagate(build_web(LayerWeights()), {"doc-a:0": 1.0})

    # Hand-traced: seed 10.0 -> a:1 gets the full damped 6.0 (only
    # neighbour), a:1 forwards 3.6 entirely to b:0 (a:0 is already active and
    # drops from the denominator), b:0 forwards 2.16 to b:1.
    assert result.energy_of("doc-b:0") > 0.0, (
        "the answer chunk is semantically invisible; only the shared rare "
        "entity can carry energy across the document boundary"
    )
    assert result.energy_of("doc-b:0") == pytest.approx(6.0 * 0.6)
    ranking = [node for node, _ in result.ranked()]
    assert ranking == ["doc-a:0", "doc-a:1", "doc-b:0", "doc-b:1"]


def test_ablating_the_entity_layer_turns_the_answer_dark() -> None:
    result = propagate(build_web(LayerWeights(entity=0.0)), {"doc-a:0": 1.0})
    assert result.energy_of("doc-b:0") == 0.0, (
        "with the entity layer off, no path crosses the document boundary - "
        "the ablation flip is the proof that the layer earns its keep"
    )
    assert result.energy_of("doc-a:1") > 0.0, (
        "the structural neighbour must stay alive; only the hop fuel is gone"
    )
