"""Structural edges: same document, same section, adjacent chunk.

The builder takes lightweight `ChunkRef` records - the structural input
contract, deliberately separate from `core.graph.Node` so that adding this
layer never touches ``core/``. Whether `Node` should eventually absorb the
positional fields is an open question deferred to the chunking step.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import TYPE_CHECKING

from spiyweb.config import StructuralEdgeConfig

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ChunkRef:
    """Positional identity of one chunk, as the structural builder needs it.

    Not a parallel `Node`: it carries only what `Node` deliberately lacks
    (`position`, `section_id`) plus the two join keys. `section_id=None` means
    "no section" - two None sections never group together.

    Attributes:
        id: Node id the emitted edges will reference.
        source_id: Owning document; edges never cross documents.
        position: Reading-order index within the document. Gaps are fine:
            adjacency means consecutive in position-sorted order.
        section_id: Optional section within the document.
    """

    id: str
    source_id: str
    position: int
    section_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("chunk id must not be empty")
        if not self.source_id:
            raise ValueError(f"chunk {self.id!r} must carry a non-empty source_id")
        if self.position < 0:
            raise ValueError(
                f"chunk {self.id!r} has position {self.position!r}; "
                "must not be negative"
            )


def build_structural_edges(
    chunks: Sequence[ChunkRef],
    config: StructuralEdgeConfig | None = None,
) -> list[tuple[str, str, float]]:
    """Emit raw structural edges over `chunks`, one per unordered pair.

    Per pair the strongest enabled relation wins (relations are nested, so
    summing would double-count the same structural fact): adjacent, then
    same-section, then same-document. A pair with no enabled relation is
    omitted entirely - `0.0` is never emitted, that value belongs to the
    dedup-suppressed-edge contract. The same-document sweep is skipped
    entirely while its weight is `0.0`, so a disabled relation costs nothing.

    Output is canonically ordered (`u < v` within each tuple) and sorted, so
    the result is deterministic across platforms.
    """
    cfg = config if config is not None else StructuralEdgeConfig()

    seen_ids: set[str] = set()
    by_document: defaultdict[str, list[ChunkRef]] = defaultdict(list)
    for chunk in chunks:
        if chunk.id in seen_ids:
            raise ValueError(f"duplicate chunk id {chunk.id!r}")
        seen_ids.add(chunk.id)
        by_document[chunk.source_id].append(chunk)

    weights: dict[tuple[str, str], float] = {}

    def offer(u: str, v: str, weight: float) -> None:
        pair = (u, v) if u < v else (v, u)
        current = weights.get(pair)
        if current is None or weight > current:
            weights[pair] = weight

    for members in by_document.values():
        ordered = sorted(members, key=lambda chunk: chunk.position)
        for previous, current in pairwise(ordered):
            if previous.position == current.position:
                raise ValueError(
                    f"chunks {previous.id!r} and {current.id!r} share position "
                    f"{current.position!r} in source {current.source_id!r}"
                )
            if cfg.adjacent > 0.0:
                offer(previous.id, current.id, cfg.adjacent)

        if cfg.same_section > 0.0:
            by_section: defaultdict[str, list[ChunkRef]] = defaultdict(list)
            for chunk in ordered:
                if chunk.section_id is not None:
                    by_section[chunk.section_id].append(chunk)
            for section_members in by_section.values():
                for left, right in combinations(section_members, 2):
                    offer(left.id, right.id, cfg.same_section)

        if cfg.same_document > 0.0:
            for left, right in combinations(ordered, 2):
                offer(left.id, right.id, cfg.same_document)

    return [(u, v, weight) for (u, v), weight in sorted(weights.items())]
