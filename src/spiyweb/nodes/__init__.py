"""Node builders - one module per node layer.

Builders turn prepared document inputs into the pairs the rest of the index
pipeline consumes: a `core.graph.Node` for the graph registry plus the
index-time records (`ChunkRef`, text) the edge builders and the embedder need.
`core/` is never touched from here.
"""

from spiyweb.nodes.chunks import Chunk, DocumentInput, TextUnit, chunk_documents

__all__ = [
    "Chunk",
    "DocumentInput",
    "TextUnit",
    "chunk_documents",
]
