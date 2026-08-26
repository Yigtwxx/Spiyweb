"""Golden artifacts: what the index pipeline writes, pinned byte for byte.

This file exists for one job. `build_index` is being split into a
corpus-agnostic core and a benchmark adapter, and every sealed Phase 1 number
was produced by the version before the split. A refactor that quietly changes
an edge weight, a node length or an id order does not fail any other test -
it just makes the sealed table incomparable with everything measured after
it. So the artifacts are pinned here first, and the refactor has to keep them.

The fixture is deliberately not a MuSiQue miniature. Every passage in
MuSiQue, HotpotQA and 2Wiki is its own single-unit document, which is why
`edges_structural.json` came out EMPTY in all four sealed indexes. Here the
documents carry two units each, so the structural layer is non-empty and the
refactor is checked against a shape the benchmark corpora never produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spiyweb.config import EntityEdgeConfig, EntityExtractionConfig, SemanticEdgeConfig
from spiyweb.evaluation.datasets import MusiqueDataset
from spiyweb.evaluation.index import IndexPaths, build_index
from spiyweb.nodes import DocumentInput, TextUnit

DOCUMENTS = (
    DocumentInput(
        source_id="alpha",
        units=(
            TextUnit(text="The tower was raised on the shore."),
            TextUnit(text="Wardenclyffe drew power from the ground."),
        ),
        timestamp=1000.0,
    ),
    DocumentInput(
        source_id="beta",
        units=(
            TextUnit(text="What happened at Wardenclyffe afterwards."),
            TextUnit(text="An unrelated closing paragraph."),
        ),
    ),
)
TITLES = {
    "alpha:0": "Tower",
    "alpha:1": "Ground",
    "beta:0": "Aftermath",
    "beta:1": "Colophon",
}
TEXTS = {
    "alpha:0": "The tower was raised on the shore.",
    "alpha:1": "Wardenclyffe drew power from the ground.",
    "beta:0": "What happened at Wardenclyffe afterwards.",
    "beta:1": "An unrelated closing paragraph.",
}

# Keyed by the COMPOSED string (title + newline + text), because that is what
# the pipeline hands the embedder - a fake keyed by the bare text would pass
# while the composition silently broke.
DIRECTIONS = {
    "Tower\nThe tower was raised on the shore.": [1.0, 0.0, 0.0],
    "Ground\nWardenclyffe drew power from the ground.": [0.8, 0.6, 0.0],
    "Aftermath\nWhat happened at Wardenclyffe afterwards.": [0.0, 1.0, 0.0],
    "Colophon\nAn unrelated closing paragraph.": [0.0, 0.0, 1.0],
}


class FixedEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [DIRECTIONS[text] for text in texts]


class KeywordPipeline:
    """spaCy stand-in: two rare names, one of them crossing the documents."""

    def pipe(self, texts: list[str]) -> list[object]:
        class Span:
            def __init__(self, text: str) -> None:
                self.text = text
                self.label_ = "ORG"

        class Doc:
            def __init__(self, ents: tuple[object, ...]) -> None:
                self.ents = ents

        docs: list[object] = []
        for text in texts:
            found: list[object] = []
            if "Wardenclyffe" in text:
                found.append(Span("Wardenclyffe"))
            if "tower" in text.lower():
                found.append(Span("tower"))
            docs.append(Doc(tuple(found)))
        return docs


GOLDEN_NODES = [
    {
        "id": "alpha:0",
        "layer": "chunk",
        "source_id": "alpha",
        "length": 34,
        "timestamp": 1000.0,
        "cluster_id": None,
        "polarity": 1,
    },
    {
        "id": "alpha:1",
        "layer": "chunk",
        "source_id": "alpha",
        "length": 40,
        "timestamp": 1000.0,
        "cluster_id": None,
        "polarity": 1,
    },
    {
        "id": "beta:0",
        "layer": "chunk",
        "source_id": "beta",
        "length": 41,
        "timestamp": None,
        "cluster_id": None,
        "polarity": 1,
    },
    {
        "id": "beta:1",
        "layer": "chunk",
        "source_id": "beta",
        "length": 31,
        "timestamp": None,
        "cluster_id": None,
        "polarity": 1,
    },
]
GOLDEN_ENTITIES = {
    "alpha:0": ["tower"],
    "alpha:1": ["wardenclyffe"],
    "beta:0": ["wardenclyffe"],
    "beta:1": [],
}
GOLDEN_SEMANTIC = [
    ["alpha:0", "alpha:1", 0.7999999928474427],
    ["alpha:1", "beta:0", 0.6000000095367428],
]
GOLDEN_ENTITY = [["alpha:1", "beta:0", 0.5]]
GOLDEN_STRUCTURAL = [["alpha:0", "alpha:1", 1.0], ["beta:0", "beta:1", 1.0]]
GOLDEN_VECTOR_IDS = ["alpha:0", "alpha:1", "beta:0", "beta:1"]


def _build(root: Path) -> IndexPaths:
    paths = IndexPaths(root=root)
    build_index(
        MusiqueDataset(documents=DOCUMENTS, titles=TITLES, texts=TEXTS, questions=()),
        paths,
        embedder=FixedEmbedder(),
        entity_pipeline=KeywordPipeline(),
        # min_entities=0 keeps the LLM fallback out of the fixture; the
        # df guard is off because a four-chunk corpus makes every name
        # corpus-wide by definition.
        extraction_config=EntityExtractionConfig(min_entities=0),
        semantic_config=SemanticEdgeConfig(k=2, min_similarity=0.0),
        entity_config=EntityEdgeConfig(max_df_ratio=1.0),
        log=lambda message: None,
    )
    return paths


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_node_registry_is_unchanged(tmp_path: Path) -> None:
    """Ids, layer, source, LENGTH and timestamp - the mass and vote inputs."""
    paths = _build(tmp_path)
    assert _read(paths.nodes_json) == GOLDEN_NODES


def test_the_entity_artifact_is_unchanged(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    assert _read(paths.entities_json) == GOLDEN_ENTITIES


def test_every_edge_layer_is_unchanged(tmp_path: Path) -> None:
    """Weights to full float precision: a rounding change is a ranking change."""
    paths = _build(tmp_path)
    for layer, golden in (
        ("semantic", GOLDEN_SEMANTIC),
        ("entity", GOLDEN_ENTITY),
        ("structural", GOLDEN_STRUCTURAL),
        ("derivation", []),
    ):
        actual = _read(paths.edges_json(layer))
        assert isinstance(actual, list)
        assert [[u, v] for u, v, _ in actual] == [[u, v] for u, v, _ in golden], layer
        assert [w for _, _, w in actual] == pytest.approx(
            [w for _, _, w in golden], rel=0, abs=0
        ), layer


def test_the_vector_store_is_unchanged(tmp_path: Path) -> None:
    """Id ORDER is load-bearing: the NLI stage indexes into it positionally."""
    import numpy as np

    paths = _build(tmp_path)
    with np.load(paths.vectors_npz) as payload:
        ids = [str(node_id) for node_id in payload["ids"]]
        vectors = payload["vectors"].tolist()
    assert ids == GOLDEN_VECTOR_IDS
    golden = [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for row, expected in zip(vectors, golden, strict=True):
        assert row == pytest.approx(expected, abs=1e-6)


def test_the_receipt_still_records_every_config(tmp_path: Path) -> None:
    """`meta.json` may GAIN fields; losing one silently breaks comparability."""
    paths = _build(tmp_path)
    meta = _read(paths.meta_json)
    assert isinstance(meta, dict)
    for key in (
        "corpus_chunks",
        "propositions",
        "proposition_config",
        "questions",
        "entity_llm",
        "llm_model",
        "nli_model",
        "nli_edges",
        "nli_config",
        "nli_candidates",
        "llm_cache",
        "extraction_config",
        "semantic_config",
        "structural_config",
        "entity_config",
    ):
        assert key in meta, key
    assert meta["corpus_chunks"] == 4
    assert meta["questions"] == 0
    assert meta["llm_cache"] == "llm_cache.jsonl"
    assert meta["semantic_config"] == {"k": 2, "min_similarity": 0.0}
    assert meta["structural_config"] == {
        "adjacent": 1.0,
        "same_section": 0.6,
        "same_document": 0.0,
    }
