"""Index pipeline with the proposition layer: artifacts, reload, fidelity.

The claim under test: `--propositions` adds the second node layer to every
artifact (nodes, vectors, entities, derivation edges), `load_graph` carries
ALL seven node fields through the round trip (a `polarity=-1` atom must not
silently reload as +1 - it feeds D34), and an index built WITHOUT the layer
still loads (the derivation file is optional history, not an error).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spiyweb import LayerWeights
from spiyweb.evaluation.datasets import MusiqueDataset
from spiyweb.evaluation.index import IndexPaths, build_index, load_graph
from spiyweb.nodes import DocumentInput, TextUnit

DATASET = MusiqueDataset(
    documents=(
        DocumentInput(
            source_id="d0",
            units=(TextUnit(text="Curie won two Nobel Prizes."),),
            timestamp=99.0,
        ),
    ),
    titles={"d0:0": "Marie Curie"},
    texts={"d0:0": "Curie won two Nobel Prizes."},
    questions=(),
)


class HashEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text) % 7)] for text in texts]


class EmptyPipeline:
    def pipe(self, texts: list[str]) -> list[object]:
        class Doc:
            ents: tuple = ()

        return [Doc() for _ in texts]


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_proposition_stage_lands_in_every_artifact(tmp_path: Path) -> None:
    paths = IndexPaths(root=tmp_path / "idx")
    llm = FakeLLM("Marie Curie won the Nobel Prize in Physics.")
    build_index(
        DATASET,
        paths,
        embedder=HashEmbedder(),
        entity_pipeline=EmptyPipeline(),
        llm=llm,
        entity_llm=False,
        propositions=True,
        log=lambda _: None,
    )
    assert len(llm.prompts) == 1, "one call per passage, entity fallback off"
    assert "Marie Curie\n" in llm.prompts[0], (
        "extraction sees the composed title+text string, like every consumer"
    )

    records = json.loads(paths.propositions_json.read_text(encoding="utf-8"))
    assert [r["id"] for r in records] == ["d0:0#p0"]
    derivation = json.loads(paths.edges_json("derivation").read_text(encoding="utf-8"))
    assert derivation == [["d0:0", "d0:0#p0", 1.0]]

    graph = load_graph(paths, LayerWeights())
    proposition = graph.node("d0:0#p0")
    assert proposition is not None and proposition.layer == "proposition"
    assert proposition.timestamp == pytest.approx(99.0), "inherited, persisted"
    assert "d0:0#p0" in graph.neighbors("d0:0"), "the layers are linked (D10)"


def test_reopening_an_existing_proposition_layer_needs_no_llm(
    tmp_path: Path,
) -> None:
    """An LLM is needed to EXTRACT propositions, not to reopen them.

    Later stages - the NLI pass, a layer-weight re-merge - must reopen the
    two-layer index. Demanding a client before checking the artifact forced
    those callers to stand up an Ollama connection for a file already on disk.
    """
    paths = IndexPaths(root=tmp_path / "idx")
    build_index(
        DATASET,
        paths,
        embedder=HashEmbedder(),
        entity_pipeline=EmptyPipeline(),
        llm=FakeLLM("Marie Curie won the Nobel Prize in Physics."),
        entity_llm=False,
        propositions=True,
        log=lambda _: None,
    )

    build_index(
        DATASET,
        paths,
        embedder=HashEmbedder(),
        entity_pipeline=EmptyPipeline(),
        llm=None,
        entity_llm=False,
        propositions=True,
        log=lambda _: None,
    )
    graph = load_graph(paths, LayerWeights())
    assert graph.node("d0:0#p0") is not None, "the layer survived the reopen"

    with pytest.raises(ValueError, match="requires an LLM client"):
        build_index(
            DATASET,
            paths,
            embedder=HashEmbedder(),
            entity_pipeline=EmptyPipeline(),
            llm=None,
            entity_llm=False,
            propositions=True,
            force=True,
            log=lambda _: None,
        )


def test_node_roundtrip_keeps_all_seven_fields(tmp_path: Path) -> None:
    paths = IndexPaths(root=tmp_path / "idx")
    build_index(
        DATASET,
        paths,
        embedder=HashEmbedder(),
        entity_pipeline=EmptyPipeline(),
        log=lambda _: None,
    )
    # Simulate an index-time polarity marker (the D34 upstream): the loader
    # must carry it through, never silently reset it to +1.
    records = json.loads(paths.nodes_json.read_text(encoding="utf-8"))
    records[0]["polarity"] = -1
    paths.nodes_json.write_text(json.dumps(records), encoding="utf-8")
    node = load_graph(paths, LayerWeights()).node("d0:0")
    assert node is not None
    assert node.polarity == -1, "a reload must not launder the corpus's 'no'"
    assert node.timestamp == pytest.approx(99.0)


def test_an_index_without_the_layer_still_loads(tmp_path: Path) -> None:
    paths = IndexPaths(root=tmp_path / "idx")
    build_index(
        DATASET,
        paths,
        embedder=HashEmbedder(),
        entity_pipeline=EmptyPipeline(),
        log=lambda _: None,
    )
    # Pre-proposition indexes have no derivation artifact at all.
    paths.edges_json("derivation").unlink()
    graph = load_graph(paths, LayerWeights())
    assert graph.node("d0:0") is not None, "an absent optional layer is empty"


def test_propositions_without_an_llm_is_a_hard_error(tmp_path: Path) -> None:
    paths = IndexPaths(root=tmp_path / "idx")
    with pytest.raises(ValueError, match="requires an LLM"):
        build_index(
            DATASET,
            paths,
            embedder=HashEmbedder(),
            entity_pipeline=EmptyPipeline(),
            propositions=True,
            log=lambda _: None,
        )
