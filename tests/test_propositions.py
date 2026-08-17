"""Proposition layer (D10): atomic facts as second-layer nodes, linked.

The claim under test: extraction produces deterministic `{chunk}#p{n}` ids
on the KEPT lines only, inherits source and timestamp from the parent
chunk, filters fragments and echoes, and the derivation layer is the
switchable bridge that lets energy flow from a chunk into its propositions.
"""

from __future__ import annotations

import pytest

from spiyweb import (
    Graph,
    LayerWeights,
    PropagationConfig,
    PropositionConfig,
    build_derivation_edges,
    extract_propositions,
    propagate,
)
from spiyweb.nodes import DocumentInput, TextUnit, chunk_documents

CHUNKS = chunk_documents(
    [
        DocumentInput(
            source_id="doc",
            units=(TextUnit(text="Marie Curie won two Nobel Prizes."),),
            timestamp=123.0,
        )
    ]
)


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_extraction_builds_second_layer_nodes_with_inherited_fields() -> None:
    llm = FakeLLM(
        "Marie Curie won the Nobel Prize in Physics.\n"
        "Marie Curie won the Nobel Prize in Chemistry.\n"
    )
    propositions = extract_propositions(CHUNKS, llm)
    assert [p.node.id for p in propositions] == ["doc:0#p0", "doc:0#p1"]
    first = propositions[0]
    assert first.node.layer == "proposition"
    assert first.node.source_id == "doc", "vote granularity follows the parent"
    assert first.node.timestamp == pytest.approx(123.0)
    assert first.node.length == len(first.text), "characters, the chunker's unit"
    assert first.chunk_id == "doc:0"
    assert len(llm.prompts) == 1, "one LLM call per chunk - the cost contract"


def test_fragments_echoes_and_overflow_are_filtered() -> None:
    lines = [
        "- Marie Curie won the Nobel Prize in Physics.",  # bullet tolerated
        "Too short.",  # under min_chars
        "MARIE CURIE   won the Nobel Prize in Physics.",  # case/space echo
        "1. Marie Curie was born in Warsaw in the year 1867.",  # numbering
        "Marie Curie discovered the elements polonium and radium.",  # overflow
    ]
    propositions = extract_propositions(
        CHUNKS,
        FakeLLM("\n".join(lines)),
        PropositionConfig(max_per_chunk=2, min_chars=15),
    )
    assert [p.text for p in propositions] == [
        "Marie Curie won the Nobel Prize in Physics.",
        "Marie Curie was born in Warsaw in the year 1867.",
    ], "kept ids leave no numbering holes: filtered lines never claim a slot"


def test_an_empty_completion_is_a_legitimate_outcome() -> None:
    assert extract_propositions(CHUNKS, FakeLLM("")) == []


def test_texts_override_feeds_the_prompt_not_the_node() -> None:
    llm = FakeLLM("Marie Curie won the Nobel Prize in Physics.")
    extract_propositions(CHUNKS, llm, texts={"doc:0": "TITLE\nbody"})
    assert "TITLE\nbody" in llm.prompts[0], (
        "the harness's composed string reaches the LLM, as with entities"
    )


def test_derivation_layer_is_the_switchable_bridge() -> None:
    propositions = extract_propositions(
        CHUNKS, FakeLLM("Marie Curie won the Nobel Prize in Physics.")
    )
    derivation = build_derivation_edges(propositions)
    assert derivation == [("doc:0", "doc:0#p0", 1.0)]
    layers = {"derivation": derivation}
    seeds = {"doc:0": 1.0}
    dark = propagate(
        Graph.from_layers(layers, weights=LayerWeights(derivation=0.0)),
        seeds,
        PropagationConfig(),
    )
    lit = propagate(
        Graph.from_layers(layers, weights=LayerWeights(derivation=1.0)),
        seeds,
        PropagationConfig(),
    )
    assert "doc:0#p0" not in dark.activations, "weight 0 cuts the layers apart"
    assert lit.energy_of("doc:0#p0") == pytest.approx(6.0), (
        "the derivation layer is how a chunk's energy reaches its propositions"
    )


def test_config_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="max_per_chunk"):
        PropositionConfig(max_per_chunk=0)
    with pytest.raises(ValueError, match="min_chars"):
        PropositionConfig(min_chars=0)
