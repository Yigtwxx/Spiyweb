"""Entity edges: shared entity / concept links - the main hop fuel.

Cosine neighbours are paraphrases; entity edges are what let the web hop to a
*different* document that talks about the same thing. The builder consumes a
prepared ``chunk id -> entity strings`` mapping (extraction lives in
`spiyweb.entities`, outside this package) and treats entity strings as opaque
keys - normalisation is the extractor's job, a pure function holds no
linguistic opinion.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from spiyweb.config import EntityEdgeConfig

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


def build_entity_edges(
    entities: Mapping[str, Collection[str]],
    config: EntityEdgeConfig | None = None,
) -> list[tuple[str, str, float]]:
    """Emit raw entity edges: per shared entity a pair gains ``1 / df(entity)``.

    ``df(entity)`` is the number of chunks in `entities` mentioning it, so a
    rare shared entity binds strongly (df=2 contributes 0.5 per side of the
    pair) while a ubiquitous one barely counts. Contributions from several
    shared entities sum - converging evidence, same stance as the additive
    layer merge. A pair sharing nothing is omitted entirely - `0.0` is never
    emitted, that value belongs to the dedup-suppressed-edge contract.

    Entities mentioned by more than ``max_df_ratio * n_chunks`` chunks are
    dropped before pairing (strict `>`, so a ratio of `1.0` truly disables the
    guard): 1/df bounds their weight but not the near-clique of edges they
    would emit. Duplicate mentions of one entity within one chunk count once.

    Only chunks that share an entity are ever paired - the inverted index
    keeps the cost at O(mentions + sum(df^2)), never O(n^2) over the corpus.
    Iteration order is fixed (sorted entities, sorted chunk ids), so the
    accumulated float sums are bit-identical across platforms. Output is
    canonically ordered (`u < v` within each tuple) and sorted.
    """
    cfg = config if config is not None else EntityEdgeConfig()

    mentions: dict[str, set[str]] = {}
    for chunk_id, chunk_entities in entities.items():
        if not chunk_id:
            raise ValueError("chunk id must not be empty")
        for entity in chunk_entities:
            if not entity:
                raise ValueError(
                    f"chunk {chunk_id!r} carries an empty entity string; "
                    "extraction must never emit one"
                )
            mentions.setdefault(entity, set()).add(chunk_id)

    max_df = cfg.max_df_ratio * len(entities)
    weights: dict[tuple[str, str], float] = {}
    for entity in sorted(mentions):
        chunk_ids = mentions[entity]
        df = len(chunk_ids)
        if df < 2 or df > max_df:
            continue
        contribution = 1.0 / df
        for u, v in combinations(sorted(chunk_ids), 2):
            weights[(u, v)] = weights.get((u, v), 0.0) + contribution

    return [(u, v, weight) for (u, v), weight in sorted(weights.items())]
