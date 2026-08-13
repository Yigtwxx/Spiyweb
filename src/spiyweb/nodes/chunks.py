"""Chunk-level nodes: pre-split text units become Node + ChunkRef pairs.

The chunker does NOT split text: it consumes documents that already arrive as
units (MuSiQue provides paragraphs). The open chunk-size question (300-500
tokens, never decided) is explicitly NOT closed by this module - when a real
splitter lands, it will produce `TextUnit`s and feed this same builder.

Node stays lean on purpose: `ChunkRef` carries the positional fields the
structural builder needs, `Node` carries what propagation needs, and the
chunker is the single place that keeps the two in step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiyweb.core.graph import Node
from spiyweb.edges.structural import ChunkRef

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class TextUnit:
    """One pre-split unit of a document (a paragraph, a passage).

    Attributes:
        text: The unit's text; must contain at least one non-whitespace
            character - an empty unit is prepared-data corruption, not a
            chunk.
        section_id: Optional section the unit belongs to; `None` means "no
            section" and never groups with other `None`s.
    """

    text: str
    section_id: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text unit must contain non-whitespace text")


@dataclass(frozen=True)
class DocumentInput:
    """One source document, already split into ordered units.

    Attributes:
        source_id: Document identity; becomes `Node.source_id` (the vote
            granularity) and `ChunkRef.source_id` (the structural join key).
        units: Reading-ordered units; the tuple index becomes the position.
        timestamp: Optional UTC epoch, passed through to every chunk's node
            (freshness is a tie-breaker only).
    """

    source_id: str
    units: tuple[TextUnit, ...]
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("document source_id must not be empty")


@dataclass(frozen=True)
class Chunk:
    """One chunk with all three faces the pipeline needs.

    Attributes:
        node: Graph registry entry (`layer="chunk"`, `length` in characters -
            the raw input of the still-open mass formula).
        ref: Structural-builder input record; same `id` and `source_id`.
        text: The chunk's text, carried so the embedder and the entity
            extractor never re-read sources.
    """

    node: Node
    ref: ChunkRef
    text: str


def chunk_documents(documents: Sequence[DocumentInput]) -> list[Chunk]:
    """Build `Chunk`s for every unit of every document.

    Chunk ids follow the deterministic scheme ``{source_id}:{position}``.
    Duplicate `source_id`s across documents are rejected, and a defensive
    guard rejects any chunk-id collision the scheme could still produce.
    """
    seen_sources: set[str] = set()
    seen_ids: set[str] = set()
    chunks: list[Chunk] = []
    for document in documents:
        if document.source_id in seen_sources:
            raise ValueError(f"duplicate document source_id {document.source_id!r}")
        seen_sources.add(document.source_id)
        for position, unit in enumerate(document.units):
            chunk_id = f"{document.source_id}:{position}"
            if chunk_id in seen_ids:
                raise ValueError(f"duplicate chunk id {chunk_id!r}")
            seen_ids.add(chunk_id)
            chunks.append(
                Chunk(
                    node=Node(
                        id=chunk_id,
                        layer="chunk",
                        source_id=document.source_id,
                        length=len(unit.text),
                        timestamp=document.timestamp,
                    ),
                    ref=ChunkRef(
                        id=chunk_id,
                        source_id=document.source_id,
                        position=position,
                        section_id=unit.section_id,
                    ),
                    text=unit.text,
                )
            )
    return chunks
