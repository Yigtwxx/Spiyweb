"""One query, end to end: retrieve, scene, ledger, comparison, honesty outputs.

Everything the browser draws for a single question is produced here in one
`retrieve()` call. The layout, the scene assembly and the side-by-side view
model all come from `spiyweb.scene`, the module the browser face shares
uses, so the two front ends cannot drift into showing different pictures of
the same run.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from server.resources import CACHE
from server.schemas import (
    ActivationPathDto,
    ComparisonDto,
    ComparisonRowDto,
    InspectRequest,
    InspectResponse,
    InspectStats,
    LedgerDto,
    RefusalDto,
    SceneDto,
    SeedDto,
    ThemeClusterDto,
    WarningDto,
)
from server.settings import SETTINGS
from spiyweb.ledger import build_ledger

if TYPE_CHECKING:
    from pathlib import Path


class QueryProblem(ValueError):
    """The query cannot be run as asked; the message is the explanation."""


def _query_vector(request: InspectRequest, root: Path) -> tuple[list[float], str]:
    vectors = CACHE.vectors(root)
    spec = request.query
    if spec.mode == "atom":
        node = spec.node or ""
        row = vectors.position.get(node)
        if row is None:
            raise QueryProblem(f"atom {node!r} is not in this index")
        return vectors.matrix[row].tolist(), node
    text = (spec.text or "").strip()
    if not text:
        raise QueryProblem("enter a question, or pick a corpus atom instead")
    embedder = CACHE.embedder(spec.model, spec.device)
    return embedder.embed_queries([text])[0], text  # type: ignore[attr-defined]


def run(request: InspectRequest) -> InspectResponse:
    """Run one inspection and shape it for the wire."""
    from spiyweb.config import (
        ConflictConfig,
        DedupConfig,
        MassConfig,
        OutputConfig,
        PolarityConfig,
        PropagationConfig,
        RetrievalConfig,
    )
    from spiyweb.core.conflict import conflict_adjacency
    from spiyweb.evaluation.index import IndexPaths, load_nli_edges
    from spiyweb.output import (
        activation_paths,
        build_refusal_report,
        dispute_warnings,
        entity_edge_labels,
        gap_warnings,
        theme_clusters,
    )
    from spiyweb.profiles import PROFILES
    from spiyweb.retrieve import retrieve
    from spiyweb.scene import (
        EDGE_LAYER_ORDER,
        LAYER_COLORS,
        LayoutConfig,
        ViewConfig,
        build_comparison,
        build_scene,
        hop_ring_layout,
        make_similarity,
    )
    from spiyweb.viewer.payload import (
        clusters_payload,
        ledger_payload,
        paths_payload,
        refusal_payload,
        scene_payload_of,
    )

    root = CACHE.index_root(request.index)
    if request.view.max_nodes > SETTINGS.max_scene_nodes:
        raise QueryProblem(
            f"max_nodes {request.view.max_nodes} exceeds the drawable ceiling "
            f"of {SETTINGS.max_scene_nodes}"
        )

    propagation = PropagationConfig(
        seed_energy=request.propagation.seed_energy,
        damping=request.propagation.damping,
        threshold_ratio=request.propagation.threshold_ratio,
        max_hop=request.propagation.max_hop,
        max_nodes=request.propagation.max_nodes,
        split_alpha=request.propagation.split_alpha,
        mass=MassConfig(enabled=request.propagation.mass_enabled),
    )
    config = RetrievalConfig(
        seed_width=request.seed_width,
        propagation=propagation,
        contact_overfetch=request.contact_overfetch,
    )
    if request.profile:
        profile = PROFILES.get(request.profile)
        if profile is None:
            raise QueryProblem(f"unknown profile {request.profile!r}")
        config = profile.as_retrieval(config)

    graph = CACHE.graph(root, request.weights.as_tuple())
    store = CACHE.store(root)
    vectors = CACHE.vectors(root)
    layer_index = CACHE.layer_index(root)
    entities = CACHE.entities(root)
    corpus = CACHE.corpus(root, request.sample_size, request.sample_seed)
    titles, texts = corpus["titles"], corpus["texts"]

    query, label = _query_vector(request, root)

    # Dedup needs BOTH halves; supplying only the config leaves the mechanism
    # silently off, which is the failure this page must never hide.
    dedup = (
        DedupConfig(
            enabled=True,
            sigma=request.ablations.dedup_sigma,
            floor=request.ablations.dedup_floor,
            include_seeds=request.ablations.dedup_include_seeds,
        )
        if request.ablations.dedup
        else None
    )
    similarity = make_similarity(vectors) if dedup is not None else None

    marked = load_nli_edges(IndexPaths(root=root))
    negative = conflict_adjacency(marked) if marked else None
    conflict = ConflictConfig() if (negative and request.ablations.conflict) else None
    polarity = PolarityConfig() if request.ablations.polarity else None
    source_of = {node_id: node.source_id for node_id, node in graph.node_data.items()}

    started = time.perf_counter()
    result = retrieve(
        query,
        store,  # type: ignore[arg-type]
        graph,
        config,
        similarity=similarity,
        dedup=dedup,
        source_of=source_of,
        negative=negative,
        conflict=conflict,
        polarity=polarity,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    scene_started = time.perf_counter()
    cuts = {**result.contact_suppressed, **result.propagation.suppressed}
    scene = build_scene(
        activations=result.propagation.activations,
        seeds=result.seeds,
        votes=result.votes(),
        suppressed=cuts,
        graph=graph,
        layer_index=layer_index,
        weights=_layer_weights(request),
        view=ViewConfig(
            max_nodes=request.view.max_nodes,
            max_edges=request.view.max_edges,
            edge_mode=request.view.edge_mode,
            label_top_n=request.view.label_top_n,
        ),
        layout=LayoutConfig(),
        salt=label,
        texts=texts,
        disputed=result.disputed,
    )
    drawn = [node.id for node in scene.nodes]
    hops = {node.id: max(node.hop, 0) for node in scene.nodes}
    energies = {node.id: node.energy for node in scene.nodes}
    rings = hop_ring_layout(drawn, hops, energies, salt=label)
    scene_ms = (time.perf_counter() - scene_started) * 1000.0

    baseline_pairs = store.search(query, request.k)  # type: ignore[attr-defined]
    comparison = build_comparison(
        web_ranked=result.ranked(),
        baseline_ids=[node for node, _ in baseline_pairs],
        k=request.k,
        activations=result.propagation.activations,
        votes=result.votes(),
        graph=graph,
        seeds=set(result.seeds),
        disputed=result.disputed,
        contact_tau=result.contact_tau,
        dedup_enabled=dedup is not None,
        titles=titles,
        texts=texts,
        baseline_scores=dict(baseline_pairs),
    )

    book = build_ledger(result.propagation, graph, propagation)
    paths = activation_paths(result.propagation)
    labels = entity_edge_labels(
        [
            (contributor, node)
            for node, activation in result.propagation.activations.items()
            for contributor in activation.contributors
        ],
        entities,
    )
    clusters = theme_clusters(result.propagation, graph)
    output_config = OutputConfig()
    warnings = [
        WarningDto(kind="gap", message=warning.message, nodes=list(warning.nodes_a))
        for warning in gap_warnings(clusters, output_config)
    ] + [
        WarningDto(kind="dispute", message=warning.message, nodes=[warning.node])
        for warning in dispute_warnings(result.disputes)
    ]
    report = build_refusal_report(result.propagation, graph, config=output_config)
    seed_total = sum(result.seeds.values()) or 1.0
    confidence = result.confidence

    return InspectResponse(
        query_label=label,
        stats=InspectStats(
            total_energy=confidence.total_energy,
            node_count=confidence.node_count,
            hop_depth=confidence.hop_depth,
            stop_reason=result.propagation.stop_reason,
            threshold=result.propagation.threshold,
            elapsed_ms=elapsed_ms,
            scene_ms=scene_ms,
        ),
        scene=SceneDto(
            **scene_payload_of(
                scene,
                rings,
                titles=titles,
                legend=LAYER_COLORS,
                layer_order=EDGE_LAYER_ORDER,
            )
        ),
        ledger=LedgerDto(
            **ledger_payload(
                book,
                dedup_cuts=len(result.propagation.suppressed),
                contact_cuts=len(result.contact_suppressed),
                contact_tau=result.contact_tau,
            )
        ),
        comparison=ComparisonDto(
            web=[_row(row) for row in comparison.web],
            baseline=[_row(row) for row in comparison.baseline],
            only_in_web=list(comparison.only_in_web),
            only_in_baseline=list(comparison.only_in_baseline),
            overlap=list(comparison.overlap),
            contact_tau=comparison.contact_tau,
            dedup_enabled=comparison.dedup_enabled,
        ),
        paths=[ActivationPathDto(**row) for row in paths_payload(paths, labels=labels)],
        clusters=[ThemeClusterDto(**row) for row in clusters_payload(clusters)],
        warnings=warnings,
        refusal=RefusalDto(**refusal_payload(report)),
        seeds=[
            SeedDto(
                node=node,
                title=titles.get(node, node),
                similarity=score,
                energy=propagation.seed_energy * score / seed_total,
            )
            for node, score in result.seeds.items()
        ],
        texts_available=bool(texts),
    )


def _layer_weights(request: InspectRequest) -> object:
    from spiyweb.config import LayerWeights

    return LayerWeights(
        semantic=request.weights.semantic,
        entity=request.weights.entity,
        structural=request.weights.structural,
        derivation=request.weights.derivation,
        learned=request.weights.learned,
    )


def _row(row: object) -> ComparisonRowDto:
    return ComparisonRowDto(
        rank=row.rank,  # type: ignore[attr-defined]
        node_id=row.node_id,  # type: ignore[attr-defined]
        title=row.title,  # type: ignore[attr-defined]
        snippet=row.snippet,  # type: ignore[attr-defined]
        score=row.score,  # type: ignore[attr-defined]
        hop=row.hop,  # type: ignore[attr-defined]
        votes=row.votes,  # type: ignore[attr-defined]
        in_other=row.in_other,  # type: ignore[attr-defined]
        badges=list(row.badges),  # type: ignore[attr-defined]
    )
