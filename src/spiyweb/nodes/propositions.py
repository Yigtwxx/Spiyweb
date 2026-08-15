"""Proposition-level nodes: atomic facts extracted from chunks by an LLM.

The second node layer (D10): "says the same thing" is vague for a chunk and
precise for a proposition, which is exactly what redundancy voting and
contradiction detection need. Extraction runs at INDEX time - one LLM call
per chunk, the cost lives with the index, never with the query - and the
extraction-cost question (#2) stays open until measured on a real corpus.

The LLM is injected via the `LLMClient` Protocol, mirroring `entities.py`:
tests run on fakes, and the harness wraps the client in its deterministic
prompt cache. `length` is in CHARACTERS, the same unit the chunker uses -
the unit never matters across layers (mass normalises within a layer), but
it must be consistent inside one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiyweb.config import PropositionConfig
from spiyweb.core.graph import Node
from spiyweb.entities import parse_listing
from spiyweb.prompts import (
    PROPOSITION_EXTRACTION_POLARITY_PROMPT,
    PROPOSITION_EXTRACTION_PROMPT,
)

_NEGATION_PREFIX = "NEG:"

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from spiyweb.llm import LLMClient
    from spiyweb.nodes.chunks import Chunk


@dataclass(frozen=True)
class Proposition:
    """One extracted proposition with the two faces the pipeline needs.

    Attributes:
        node: Graph registry entry (`layer="proposition"`, `source_id` and
            `timestamp` inherited from the parent chunk, `length` in
            characters).
        chunk_id: The parent chunk - the derivation edge's other endpoint.
        text: The proposition sentence, carried so the embedder and the
            entity extractor never re-read sources (the chunker's rule).
    """

    node: Node
    chunk_id: str
    text: str


def extract_propositions(
    chunks: Sequence[Chunk],
    llm: LLMClient,
    config: PropositionConfig | None = None,
    *,
    texts: Mapping[str, str] | None = None,
) -> list[Proposition]:
    """Extract atomic propositions for every chunk, one LLM call per chunk.

    Ids follow the deterministic scheme ``{chunk_id}#p{n}`` over the KEPT
    propositions, so a filtered line never leaves a hole in the numbering.
    Within a chunk, propositions that normalise to the same text are kept
    once - an extractor echoing itself must not manufacture corpus support
    (the D6 vote discipline). An empty completion is a legitimate outcome
    (a chunk may state no extractable facts), never an error.

    `texts` optionally overrides the prompt text per chunk id (the harness
    passes its title-composed string, the same one the embedder and the
    entity extractor see); the chunk's own text is the default.

    With `tag_polarity` (the default), the same call also asks the model to
    prefix negated facts with ``NEG:`` - detection for the D34 atoms (open
    question #11, owner's LLM-piggyback choice). A tagged line becomes a
    `polarity=-1` node with the prefix stripped; the dedup key is the
    STRIPPED text, so a fact the model emits both tagged and untagged counts
    once (first occurrence wins, polarity included). `tag_polarity=False`
    selects the polarity-free prompt and reproduces pre-#11 output exactly.
    """
    cfg = config if config is not None else PropositionConfig()
    template = (
        PROPOSITION_EXTRACTION_POLARITY_PROMPT
        if cfg.tag_polarity
        else PROPOSITION_EXTRACTION_PROMPT
    )
    propositions: list[Proposition] = []
    for chunk in chunks:
        prompt_text = (
            chunk.text if texts is None else texts.get(chunk.node.id, chunk.text)
        )
        reply = llm.complete(template.format(text=prompt_text))
        kept: list[tuple[str, int]] = []
        seen: set[str] = set()
        for line in parse_listing(reply):
            if len(kept) == cfg.max_per_chunk:
                break
            polarity = 1
            if cfg.tag_polarity and line.startswith(_NEGATION_PREFIX):
                line = line[len(_NEGATION_PREFIX) :].strip()
                polarity = -1
            if len(line) < cfg.min_chars:
                continue
            normalized = " ".join(line.split()).casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            kept.append((line, polarity))
        for position, (text, polarity) in enumerate(kept):
            propositions.append(
                Proposition(
                    node=Node(
                        id=f"{chunk.node.id}#p{position}",
                        layer="proposition",
                        source_id=chunk.node.source_id,
                        length=len(text),
                        timestamp=chunk.node.timestamp,
                        polarity=polarity,
                    ),
                    chunk_id=chunk.node.id,
                    text=text,
                )
            )
    return propositions
