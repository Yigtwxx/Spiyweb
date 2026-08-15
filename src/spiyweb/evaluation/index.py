"""Index building: corpus -> vectors + entities + edge layers, as artifacts.

Every stage writes one file and skips itself when that file exists (the
resume mechanism; `force=True` rebuilds). The graph is deliberately NOT an
artifact: raw per-layer edges are, and `load_graph` re-merges them through
`Graph.from_layers` on every load - the same "one source of truth, rebuild
the derived thing" philosophy as the vector store, and it makes
`LayerWeights` ablations free (re-merge, never re-index).

Passages are embedded and entity-extracted over `title + newline + text`:
MuSiQue titles are entity-dense (they name the article's subject) and both
consumers must see the same string.

`meta.json` is the experiment receipt: configs, sample identity, counts and
the LLM model behind the entity fallback. A measurement whose settings are
not recorded is not a measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from spiyweb.config import (
    EdgeLayer,
    EntityEdgeConfig,
    EntityExtractionConfig,
    LayerWeights,
    NLICandidateConfig,
    NLIEdgeConfig,
    PropositionConfig,
    SemanticEdgeConfig,
    StructuralEdgeConfig,
)
from spiyweb.core.conflict import NegativeEdge
from spiyweb.core.graph import Graph, Node
from spiyweb.edges import (
    build_derivation_edges,
    build_entity_edges,
    build_nli_edges,
    build_structural_edges,
)
from spiyweb.entities import extract_entities
from spiyweb.nodes import Proposition, chunk_documents, extract_propositions
from spiyweb.store import VectorStore, build_semantic_edges_fast

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spiyweb.edges import NLIModel
    from spiyweb.embedding import Embedder
    from spiyweb.entities import EntityPipeline
    from spiyweb.evaluation.datasets import MusiqueDataset
    from spiyweb.llm import LLMClient

_EDGE_LAYER_FILES: tuple[EdgeLayer, ...] = (
    "semantic",
    "entity",
    "structural",
    "derivation",
)


@dataclass(frozen=True)
class IndexPaths:
    """Path schema of one experiment directory - names live here only."""

    root: Path

    @property
    def dataset_jsonl(self) -> Path:
        return self.root / "musique_ans_v1.0_dev.jsonl"

    @property
    def twowiki_dev_json(self) -> Path:
        return self.root / "2wiki_dev.json"

    @property
    def hotpot_dev_json(self) -> Path:
        return self.root / "hotpot_dev_distractor.json"

    @property
    def vectors_npz(self) -> Path:
        return self.root / "vectors.npz"

    @property
    def entities_json(self) -> Path:
        return self.root / "entities.json"

    @property
    def nodes_json(self) -> Path:
        return self.root / "nodes.json"

    @property
    def propositions_json(self) -> Path:
        return self.root / "propositions.json"

    def edges_json(self, layer: str) -> Path:
        return self.root / f"edges_{layer}.json"

    @property
    def nli_json(self) -> Path:
        # NOT an EdgeLayer artifact: negative edges never enter the graph
        # merge; `load_nli_edges` hands them to the conflict mechanism.
        return self.root / "edges_nli.json"

    @property
    def meta_json(self) -> Path:
        return self.root / "meta.json"

    @property
    def llm_cache_jsonl(self) -> Path:
        return self.root / "llm_cache.jsonl"

    @property
    def results_json(self) -> Path:
        return self.root / "results.json"

    @property
    def per_query_jsonl(self) -> Path:
        return self.root / "per_query.jsonl"


def composed_text(title: str, text: str) -> str:
    """The one string both the embedder and the extractor see for a passage."""
    return f"{title}\n{text}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(
    dataset: MusiqueDataset,
    paths: IndexPaths,
    *,
    embedder: Embedder,
    entity_pipeline: EntityPipeline,
    llm: LLMClient | None = None,
    extraction_config: EntityExtractionConfig | None = None,
    semantic_config: SemanticEdgeConfig | None = None,
    structural_config: StructuralEdgeConfig | None = None,
    entity_config: EntityEdgeConfig | None = None,
    llm_model: str | None = None,
    entity_llm: bool = True,
    propositions: bool = False,
    proposition_config: PropositionConfig | None = None,
    nli_model: NLIModel | None = None,
    nli_model_name: str | None = None,
    nli_config: NLIEdgeConfig | None = None,
    nli_candidates: NLICandidateConfig | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """Build (or resume) every index artifact for `dataset` under `paths`.

    `llm_model` is receipt data only (the model name behind `llm`, recorded
    in `meta.json`); the `LLMClient` protocol deliberately hides it.
    `entity_llm=False` keeps the entity fallback off even when `llm` is
    supplied - the two LLM consumers (entity fallback, proposition
    extraction) are independently switchable ablations.

    `propositions=True` adds the second node layer (D10): one LLM call per
    chunk extracts atomic propositions, which then join every downstream
    stage - embedding, entity extraction, and the `derivation` edge layer.
    Opt-in because the cost is an open question (#2); the call count is
    logged BEFORE the first call, as ever.

    `nli_model` (opt-in, D26) adds the index-time contradiction stage: the
    corpus's high-cosine pairs - propositions when the index has them, chunks
    otherwise - are scored by the model and the survivors land in
    `edges_nli.json` as negative edges for the conflict mechanism, loaded
    back by `load_nli_edges`. The candidate count is logged BEFORE the model
    runs. `nli_model_name` is receipt data for `meta.json`, like `llm_model`.
    """
    extraction_cfg = (
        extraction_config if extraction_config is not None else EntityExtractionConfig()
    )

    chunks = chunk_documents(list(dataset.documents))
    ids = [chunk.node.id for chunk in chunks]
    composed = {
        chunk_id: composed_text(dataset.titles[chunk_id], dataset.texts[chunk_id])
        for chunk_id in ids
    }

    extracted: list[Proposition] = []
    if propositions:
        if llm is None:
            raise ValueError("proposition extraction requires an LLM client")
        if force or not paths.propositions_json.exists():
            # The cost is visible up front: exactly one call per passage.
            log(f"extracting propositions: {len(ids)} passages, one LLM call each ...")
            fresh = extract_propositions(
                chunks, llm, proposition_config, texts=composed
            )
            _write_json(
                paths.propositions_json,
                [
                    {
                        "id": p.node.id,
                        "chunk_id": p.chunk_id,
                        "source_id": p.node.source_id,
                        "length": p.node.length,
                        "timestamp": p.node.timestamp,
                        "polarity": p.node.polarity,
                        "text": p.text,
                    }
                    for p in fresh
                ],
            )
        else:
            log("propositions exist, skipping the extraction stage")
        extracted = _load_propositions(paths)
        log(f"proposition layer: {len(extracted)} nodes")

    all_ids = ids + [p.node.id for p in extracted]
    texts_all = {**composed, **{p.node.id: p.text for p in extracted}}
    all_nodes = [chunk.node for chunk in chunks] + [p.node for p in extracted]

    if force or not paths.nodes_json.exists():
        _write_json(
            paths.nodes_json,
            [
                {
                    "id": node.id,
                    "layer": node.layer,
                    "source_id": node.source_id,
                    "length": node.length,
                    "timestamp": node.timestamp,
                    "cluster_id": node.cluster_id,
                    "polarity": node.polarity,
                }
                for node in all_nodes
            ],
        )

    if force or not paths.vectors_npz.exists():
        log(f"embedding {len(all_ids)} passages ...")
        vectors = embedder.embed_passages([texts_all[node_id] for node_id in all_ids])
        store = VectorStore(len(vectors[0]))
        store.add(all_ids, vectors)
        paths.root.mkdir(parents=True, exist_ok=True)
        store.save(paths.vectors_npz)
    else:
        log("vectors exist, skipping the embed stage")

    if force or not paths.entities_json.exists():
        log(f"extracting entities from {len(all_ids)} passages (spaCy bulk) ...")
        entities = extract_entities(texts_all, entity_pipeline, extraction_cfg)
        pending = {
            chunk_id: texts_all[chunk_id]
            for chunk_id, found in entities.items()
            if len(found) < extraction_cfg.min_entities
        }
        # The count is logged BEFORE any LLM call: the cost must be visible
        # up front, never discovered from a stalled progress bar.
        fallback = llm is not None and entity_llm
        log(
            f"{len(pending)} of {len(all_ids)} passages fall below "
            f"min_entities={extraction_cfg.min_entities}"
            + (" and go to the LLM" if fallback else "; LLM fallback is OFF")
        )
        if fallback and pending:
            refreshed = extract_entities(
                pending, entity_pipeline, extraction_cfg, llm=llm
            )
            entities.update(refreshed)
        _write_json(paths.entities_json, entities)
    else:
        log("entities exist, skipping the extraction stage")

    edge_builders: dict[str, Callable[[], list[tuple[str, str, float]]]] = {
        "semantic": lambda: build_semantic_edges_fast(
            *_load_vectors(paths), config=semantic_config
        ),
        "entity": lambda: build_entity_edges(
            _load_entities(paths), config=entity_config
        ),
        "structural": lambda: build_structural_edges(
            [chunk.ref for chunk in chunks], config=structural_config
        ),
        # Written even when empty, so `load_graph` never guesses whether an
        # index was built with the proposition layer.
        "derivation": lambda: build_derivation_edges(extracted),
    }
    for layer, builder in edge_builders.items():
        target = paths.edges_json(layer)
        if not force and target.exists():
            log(f"{layer} edges exist, skipping")
            continue
        edges = builder()
        # Per-layer counts make an entity-clique blow-up visible immediately;
        # the fix is config (max_df_ratio), never code.
        log(f"{layer} layer: {len(edges)} edges")
        _write_json(target, [[u, v, w] for u, v, w in edges])

    nli_edge_count: int | None = None
    if nli_model is not None:
        if not force and paths.nli_json.exists():
            log("nli edges exist, skipping the contradiction stage")
            nli_edge_count = len(load_nli_edges(paths))
        else:
            candidate_cfg = (
                nli_candidates if nli_candidates is not None else NLICandidateConfig()
            )
            # Contradiction is sharp on propositions and blurry on chunks
            # (D26); pair within whichever layer this index actually has.
            subset = [p.node.id for p in extracted] if extracted else list(ids)
            positions = {node_id: index for index, node_id in enumerate(all_ids)}
            vector_ids, vectors = _load_vectors(paths)
            if vector_ids != all_ids:
                raise ValueError(
                    "vectors.npz id order does not match the node registry; "
                    "rebuild the embed stage before the NLI stage"
                )
            pairs = build_semantic_edges_fast(
                subset,
                [vectors[positions[node_id]] for node_id in subset],
                config=SemanticEdgeConfig(
                    k=candidate_cfg.top_k,
                    min_similarity=candidate_cfg.min_similarity,
                ),
            )
            pairs.sort(key=lambda edge: edge[2], reverse=True)
            dropped = max(0, len(pairs) - candidate_cfg.max_pairs)
            if dropped:
                log(f"nli candidates capped: {dropped} lowest-similarity pairs cut")
            candidates = [(u, v) for u, v, _ in pairs[: candidate_cfg.max_pairs]]
            layer_name = "proposition" if extracted else "chunk"
            # The cost is visible up front: two directed scores per pair.
            log(
                f"nli stage: scoring {len(candidates)} {layer_name} pairs "
                "(2 directed scores each) ..."
            )
            negative = build_nli_edges(candidates, texts_all, nli_model, nli_config)
            log(f"nli stage: {len(negative)} contradiction edges")
            _write_json(
                paths.nli_json,
                [[edge.source, edge.target, edge.strength] for edge in negative],
            )
            nli_edge_count = len(negative)

    _write_json(
        paths.meta_json,
        {
            "corpus_chunks": len(ids),
            "propositions": len(extracted) if propositions else None,
            "proposition_config": (
                asdict(
                    proposition_config
                    if proposition_config is not None
                    else PropositionConfig()
                )
                if propositions
                else None
            ),
            "questions": len(dataset.questions),
            "entity_llm": llm is not None and entity_llm,
            "llm_model": llm_model if llm is not None else None,
            "nli_model": nli_model_name if nli_model is not None else None,
            "nli_edges": nli_edge_count,
            "nli_config": (
                asdict(nli_config if nli_config is not None else NLIEdgeConfig())
                if nli_model is not None
                else None
            ),
            "nli_candidates": (
                asdict(
                    nli_candidates
                    if nli_candidates is not None
                    else NLICandidateConfig()
                )
                if nli_model is not None
                else None
            ),
            "llm_cache": paths.llm_cache_jsonl.name,
            "extraction_config": {
                **asdict(extraction_cfg),
                "labels": sorted(extraction_cfg.labels),
            },
            "semantic_config": asdict(
                semantic_config if semantic_config is not None else SemanticEdgeConfig()
            ),
            "structural_config": asdict(
                structural_config
                if structural_config is not None
                else StructuralEdgeConfig()
            ),
            "entity_config": asdict(
                entity_config if entity_config is not None else EntityEdgeConfig()
            ),
        },
    )


def _load_vectors(paths: IndexPaths) -> tuple[list[str], list[list[float]]]:
    """Ids and vectors straight from the store artifact (single source)."""
    with np.load(paths.vectors_npz) as payload:
        ids = [str(chunk_id) for chunk_id in payload["ids"]]
        vectors = payload["vectors"].tolist()
    return ids, vectors


def _load_propositions(paths: IndexPaths) -> list[Proposition]:
    """Rebuild `Proposition`s from the artifact (single source of truth)."""
    payload = _read_json(paths.propositions_json)
    if not isinstance(payload, list):
        raise ValueError(f"{str(paths.propositions_json)!r} is not a proposition list")
    propositions: list[Proposition] = []
    for record in payload:
        timestamp = record.get("timestamp")
        propositions.append(
            Proposition(
                node=Node(
                    id=str(record["id"]),
                    layer="proposition",
                    source_id=str(record["source_id"]),
                    length=int(record["length"]),
                    timestamp=float(timestamp) if timestamp is not None else None,
                    # Absent in pre-#11 artifacts; positive is Node's default.
                    polarity=int(record.get("polarity", 1)),  # type: ignore[arg-type]
                ),
                chunk_id=str(record["chunk_id"]),
                text=str(record["text"]),
            )
        )
    return propositions


def _load_entities(paths: IndexPaths) -> dict[str, list[str]]:
    payload = _read_json(paths.entities_json)
    if not isinstance(payload, dict):
        raise ValueError(f"{str(paths.entities_json)!r} is not an entities mapping")
    return {str(chunk_id): list(found) for chunk_id, found in payload.items()}


def load_nli_edges(paths: IndexPaths) -> list[NegativeEdge]:
    """Negative (contradiction) edges of an index, empty when never built.

    An index without the NLI stage simply has no `edges_nli.json` - like the
    derivation layer, an absent optional artifact means "none", not an error.
    Feed the result through `conflict_adjacency` into `retrieve(negative=)`.
    """
    if not paths.nli_json.exists():
        return []
    payload = _read_json(paths.nli_json)
    if not isinstance(payload, list):
        raise ValueError(f"{str(paths.nli_json)!r} is not an edge list")
    return [
        NegativeEdge(source=str(u), target=str(v), strength=float(s))
        for u, v, s in payload
    ]


def load_store(paths: IndexPaths) -> VectorStore:
    """The query-time seed source, rebuilt from the vectors artifact."""
    return VectorStore.load(paths.vectors_npz)


def load_graph(paths: IndexPaths, weights: LayerWeights | None = None) -> Graph:
    """Re-merge the raw edge layers into a graph under `weights`.

    Merging at load time (instead of persisting a merged graph) is what makes
    a `LayerWeights` ablation a re-merge instead of a re-index.
    """
    layers: dict[EdgeLayer, list[tuple[str, str, float]]] = {}
    for layer in _EDGE_LAYER_FILES:
        target = paths.edges_json(layer)
        if layer == "derivation" and not target.exists():
            # Indexes built before the proposition layer carry no derivation
            # file; an absent optional layer is empty, not an error.
            layers[layer] = []
            continue
        payload = _read_json(target)
        if not isinstance(payload, list):
            raise ValueError(f"{str(target)!r} is not an edge list")
        layers[layer] = [(str(u), str(v), float(w)) for u, v, w in payload]

    nodes_payload = _read_json(paths.nodes_json)
    if not isinstance(nodes_payload, list):
        raise ValueError(f"{str(paths.nodes_json)!r} is not a node list")
    nodes = []
    for record in nodes_payload:
        timestamp = record.get("timestamp")
        cluster_id = record.get("cluster_id")
        nodes.append(
            Node(
                id=str(record["id"]),
                layer=record["layer"],
                source_id=str(record["source_id"]),
                length=int(record["length"]),
                # Absent in pre-2026-08-14 artifacts; defaults match Node's.
                timestamp=float(timestamp) if timestamp is not None else None,
                cluster_id=str(cluster_id) if cluster_id is not None else None,
                polarity=int(record.get("polarity", 1)),  # type: ignore[arg-type]
            )
        )

    return Graph.from_layers(layers, weights=weights, nodes=nodes)
