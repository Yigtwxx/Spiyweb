"""Semantic edges: cosine kNN over precomputed embeddings.

This layer is seed contact and fallback only - hopping along paraphrases
returns repetition, not new information - which is why its merged weight is
deliberately weak. The builder just reports raw cosine.

Similarity is brute-force O(n^2 * d) in pure stdlib Python: the project ships
with zero runtime dependencies, and the FAISS-backed store arrives in a later
step. `Sequence[float]` also accepts numpy rows, so swapping the source of
neighbours later is non-breaking; only `_top_k` would need replacing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from spiyweb.config import SemanticEdgeConfig

if TYPE_CHECKING:
    from collections.abc import Sequence


def _cosine(
    u: Sequence[float], v: Sequence[float], norm_u: float, norm_v: float
) -> float:
    return math.fsum(a * b for a, b in zip(u, v, strict=True)) / (norm_u * norm_v)


def _top_k(similarities: dict[str, float], k: int) -> list[str]:
    """Ids of the `k` most similar candidates, ties broken by id for determinism."""
    ranked = sorted(similarities.items(), key=lambda item: (-item[1], item[0]))
    return [candidate for candidate, _ in ranked[:k]]


def build_semantic_edges(
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    config: SemanticEdgeConfig | None = None,
) -> list[tuple[str, str, float]]:
    """Emit raw cosine edges by union kNN over `ids` / `embeddings`.

    An unordered pair is emitted once when either endpoint ranks the other in
    its top-k (union, not mutual - mutual kNN prunes exactly the legitimate
    first-contact points this layer exists for) and its similarity is strictly
    above `config.min_similarity`. The strict comparison means a cosine of
    exactly `0.0` never leaks in as a fake suppressed edge, and the
    non-negative floor keeps negative cosine away from the graph, whose edge
    weights must never be negative.

    A zero-norm embedding raises instead of being skipped: real embedding
    models never emit one, and silence would hide upstream corruption.
    `k > n - 1` caps naturally. Output is canonically ordered (`u < v`) and
    sorted, deterministic across platforms.
    """
    cfg = config if config is not None else SemanticEdgeConfig()

    if len(ids) != len(embeddings):
        raise ValueError(
            f"got {len(ids)} ids but {len(embeddings)} embeddings; "
            "they must pair up one-to-one"
        )
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        for node_id in ids:
            if node_id in seen:
                raise ValueError(f"duplicate node id {node_id!r}")
            seen.add(node_id)
    if len(ids) < 2:
        return []

    dimension = len(embeddings[0])
    norms: list[float] = []
    for node_id, vector in zip(ids, embeddings, strict=True):
        if len(vector) != dimension:
            raise ValueError(
                f"embedding of node {node_id!r} has dimension {len(vector)}; "
                f"expected {dimension}"
            )
        norm = math.sqrt(math.fsum(component * component for component in vector))
        if norm == 0.0:
            raise ValueError(
                f"embedding of node {node_id!r} has zero norm; "
                "a real embedding model never emits one"
            )
        norms.append(norm)

    similarities: dict[tuple[str, str], float] = {}
    neighbours: dict[str, dict[str, float]] = {node_id: {} for node_id in ids}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            similarity = _cosine(embeddings[i], embeddings[j], norms[i], norms[j])
            u, v = ids[i], ids[j]
            pair = (u, v) if u < v else (v, u)
            similarities[pair] = similarity
            neighbours[u][v] = similarity
            neighbours[v][u] = similarity

    emitted: set[tuple[str, str]] = set()
    for node_id, candidates in neighbours.items():
        for candidate in _top_k(candidates, cfg.k):
            u, v = node_id, candidate
            emitted.add((u, v) if u < v else (v, u))

    return [
        (u, v, similarities[(u, v)])
        for (u, v) in sorted(emitted)
        if similarities[(u, v)] > cfg.min_similarity
    ]
