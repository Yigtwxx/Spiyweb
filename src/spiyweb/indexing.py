"""Index-time facade: everything needed to BUILD a web, in one namespace.

`spiyweb` itself is the QUERY-time contract - `retrieve`, `propagate` and the
configs they read. Building the graph is a different job with a different
dependency profile, so it gets its own front door instead of doubling the
top-level `__all__` and blurring the layering the package is built on.

The zero-dependency rule survives intact, and not by accident. Every name
below except the two FAISS-bound ones is pure Python at import time:
`embedding` imports torch inside `detect_device`, `entities` imports spaCy
inside `load_spacy_pipeline`, and `edges/` states the rule in its own
docstring. So they are imported eagerly, and type checkers and
autocomplete work on them with no magic at all.

`VectorStore` and `build_semantic_edges_fast` live in `spiyweb.store`, which
owns the faiss/numpy import at module level. They arrive through PEP 562
module `__getattr__`, so `import spiyweb.indexing` still works with nothing
installed and only TOUCHING those two names asks for `spiyweb[store]`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spiyweb.edges import (
    ChunkRef,
    build_entity_edges,
    build_semantic_edges,
    build_structural_edges,
    shared_subject_pairs,
)
from spiyweb.embedding import (
    Embedder,
    EncoderLike,
    SentenceTransformerEmbedder,
    detect_device,
    resolve_device,
)
from spiyweb.entities import EntityPipeline, extract_entities, load_spacy_pipeline
from spiyweb.llm import LLMClient, LLMError, NativeOllamaClient, OpenAICompatClient
from spiyweb.nodes import Chunk, DocumentInput, TextUnit, chunk_documents

if TYPE_CHECKING:
    # Real symbols for the type checkers. Never executed at runtime, so the
    # faiss import never happens on `import spiyweb.indexing`.
    from spiyweb.store import VectorStore, build_semantic_edges_fast

_LAZY: dict[str, str] = {
    "VectorStore": "spiyweb.store",
    "build_semantic_edges_fast": "spiyweb.store",
}

__all__ = [
    "Chunk",
    "ChunkRef",
    "DocumentInput",
    "Embedder",
    "EncoderLike",
    "EntityPipeline",
    "LLMClient",
    "LLMError",
    "NativeOllamaClient",
    "OpenAICompatClient",
    "SentenceTransformerEmbedder",
    "TextUnit",
    "VectorStore",
    "build_entity_edges",
    "build_semantic_edges",
    "build_semantic_edges_fast",
    "build_structural_edges",
    "chunk_documents",
    "detect_device",
    "extract_entities",
    "load_spacy_pipeline",
    "resolve_device",
    "shared_subject_pairs",
]


def __getattr__(name: str) -> object:
    """PEP 562: the two FAISS-bound names, resolved on first touch.

    `spiyweb.store` raises the install hint itself, so this neither wraps nor
    re-words it - the caller gets `pip install spiyweb[store]` verbatim.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    """List the declared surface, the two lazy names included.

    Without this they are missing from `dir()` until something
    touches them - they are not module attributes before that. The
    override exists for those two and nothing else, and the cost is
    that module internals stop showing up here.
    """
    return sorted(__all__)
