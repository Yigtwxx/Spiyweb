"""Shared fixtures. The tiny corpus lives here so nothing imports a test file.

`test_trace.py` and `test_viewer.py` both need a real index - built by the
real `build_index`, with real entity edges and a real FAISS store - over a
corpus small enough to reason about by hand. Building it twice would double
the suite's slowest fixture, and importing one test module from another makes
collection order load-bearing, which is a bug waiting for a rainy day.

The corpus is four passages in two documents, and the numbers in it are
chosen rather than sampled: the passage directions are orthogonal enough that
a seed's cosine neighbourhood never crosses the document boundary on its own,
so `beta:0` can only be reached through the rare entity the two documents
share. That is the multi-hop shape the whole project is about, in miniature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spiyweb import open_index
from spiyweb.config import (
    EntityEdgeConfig,
    EntityExtractionConfig,
    SemanticEdgeConfig,
)
from spiyweb.indexing import DocumentInput, TextUnit, build_index

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spiyweb.config import TraceConfig
    from spiyweb.session import SpiywebIndex

DOCUMENTS = (
    DocumentInput(
        source_id="alpha",
        units=(
            TextUnit(text="The tower was raised on the shore."),
            TextUnit(text="Wardenclyffe drew power from the ground."),
        ),
    ),
    DocumentInput(
        source_id="beta",
        units=(
            TextUnit(text="What happened at Wardenclyffe afterwards."),
            TextUnit(text="An unrelated closing paragraph."),
        ),
    ),
)

PASSAGES = {
    "The tower was raised on the shore.": [1.0, 0.0, 0.0],
    "Wardenclyffe drew power from the ground.": [0.8, 0.6, 0.0],
    "What happened at Wardenclyffe afterwards.": [0.0, 1.0, 0.0],
    "An unrelated closing paragraph.": [0.0, 0.0, 1.0],
}

DIRECTIONS = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.8, 0.0])
"""Three query directions, picked by a stable property of the text, so a
fifty-question loop needs no fifty-entry lookup table."""


class FakeEmbedder:
    """Deterministic vectors: the tests are about the web, not the model."""

    model_name = "fake-e5"

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [PASSAGES[text] for text in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [DIRECTIONS[len(text) % len(DIRECTIONS)] for text in texts]


class KeywordPipeline:
    """A spaCy stand-in that finds exactly the two entities that matter."""

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


@pytest.fixture(scope="session")
def tiny_index_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real index over the tiny corpus, built once for the whole session."""
    root = tmp_path_factory.mktemp("tiny-index")
    build_index(
        DOCUMENTS,
        root,
        embedder=FakeEmbedder(),
        entity_pipeline=KeywordPipeline(),
        embedding_model="fake-e5",
        extraction_config=EntityExtractionConfig(min_entities=0),
        semantic_config=SemanticEdgeConfig(k=2, min_similarity=0.0),
        entity_config=EntityEdgeConfig(max_df_ratio=1.0),
        log=lambda message: None,
    )
    return root


@pytest.fixture
def open_tiny(
    tiny_index_root: Path,
) -> Callable[..., SpiywebIndex]:
    """Open the tiny index with the fake embedder already wired in."""

    def _open(trace: TraceConfig | None = None, **options: object) -> SpiywebIndex:
        return open_index(
            tiny_index_root, embedder=FakeEmbedder(), trace=trace, **options
        )

    return _open
