"""Hybrid entity extraction: spaCy for the bulk, an LLM for ambiguous chunks.

This module is the impure counterpart of `edges.entity`: it produces the
``chunk id -> entity strings`` mapping the pure builder consumes. Everything
heavy is injectable - the spaCy pipeline and the LLM client are parameters, so
tests run on fakes and the real dependencies stay optional extras.

A chunk is "ambiguous" when spaCy finds fewer than
`EntityExtractionConfig.min_entities` entities after label filtering; only
those chunks cost an LLM call. Passing `llm=None` (or `min_entities=0`)
disables the LLM path entirely - the ablation switch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from spiyweb.config import EntityExtractionConfig
from spiyweb.prompts import ENTITY_EXTRACTION_PROMPT

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from spiyweb.llm import LLMClient


class EntitySpan(Protocol):
    """The two attributes of a spaCy `Span` the pipeline reads."""

    @property
    def text(self) -> str: ...

    @property
    def label_(self) -> str: ...


class EntityDoc(Protocol):
    """The single attribute of a spaCy `Doc` the pipeline reads."""

    @property
    def ents(self) -> Sequence[EntitySpan]: ...


class EntityPipeline(Protocol):
    """Minimal NER interface; a real spaCy `Language` satisfies it as-is."""

    def pipe(self, texts: Iterable[str]) -> Iterator[EntityDoc]: ...


def normalize_entity(text: str) -> str:
    """Collapse internal whitespace, strip and casefold; `''` if nothing stays.

    Both extraction paths MUST route through this helper so "Marie  Curie" and
    "marie curie" land on the same inverted-index key in the edge builder.
    """
    return " ".join(text.split()).casefold()


def load_spacy_pipeline(
    config: EntityExtractionConfig | None = None,
) -> EntityPipeline:
    """Load the configured spaCy model, with actionable failure messages."""
    cfg = config if config is not None else EntityExtractionConfig()
    try:
        import spacy
    except ImportError as error:
        raise ImportError(
            "spaCy is required for entity extraction; "
            "install it with `pip install spiyweb[entity]`"
        ) from error
    try:
        return spacy.load(cfg.spacy_model)  # type: ignore[no-any-return]
    except OSError as error:
        raise OSError(
            f"spaCy model {cfg.spacy_model!r} is not installed; "
            f"run `python -m spacy download {cfg.spacy_model}` first"
        ) from error


def extract_entities(
    texts: Mapping[str, str],
    pipeline: EntityPipeline,
    config: EntityExtractionConfig | None = None,
    llm: LLMClient | None = None,
) -> dict[str, list[str]]:
    """Extract normalised, deduplicated entities per chunk.

    spaCy (or any injected `EntityPipeline`) runs over every chunk; chunks
    left with fewer than `min_entities` entities are retried once through the
    LLM when one is provided. The result feeds `build_entity_edges` directly.
    """
    cfg = config if config is not None else EntityExtractionConfig()

    chunk_ids = list(texts)
    results: dict[str, list[str]] = {}
    docs = pipeline.pipe(texts[chunk_id] for chunk_id in chunk_ids)
    for chunk_id, doc in zip(chunk_ids, docs, strict=True):
        found = [span.text for span in doc.ents if span.label_ in cfg.labels]
        results[chunk_id] = _normalize_and_dedupe(found)

    if llm is not None and cfg.min_entities > 0:
        for chunk_id in chunk_ids:
            if len(results[chunk_id]) < cfg.min_entities:
                reply = llm.complete(
                    ENTITY_EXTRACTION_PROMPT.format(text=texts[chunk_id])
                )
                results[chunk_id] = _normalize_and_dedupe(parse_listing(reply))

    return results


def _normalize_and_dedupe(raw: Iterable[str]) -> list[str]:
    """Normalise entity strings, drop empties, keep first-occurrence order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for text in raw:
        normalized = normalize_entity(text)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def parse_listing(reply: str) -> list[str]:
    """Parse a one-item-per-line completion, tolerating list decorations.

    Shared by every extractor that asks for plain-text listings (entities,
    propositions): bullets (`- `, `* `, `+ `) and `1. ` numbering are
    stripped, blank lines dropped, order preserved.
    """
    items: list[str] = []
    for raw_line in reply.splitlines():
        line = raw_line.strip()
        for prefix in ("- ", "* ", "+ "):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
                break
        else:
            head, dot, tail = line.partition(". ")
            if dot and head.isdigit():
                line = tail.strip()
        if line:
            items.append(line)
    return items
