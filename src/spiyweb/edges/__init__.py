"""Hybrid edge builders - one module per layer.

Builders consume prepared data (ids, metadata records, precomputed embeddings)
and emit raw per-layer ``(u, v, weight)`` lists for `Graph.from_layers`. They
never call a model and never touch I/O; whatever produced the inputs lives
outside this package. Layer scaling happens only in `Graph.from_layers` via
`LayerWeights` - builders report unscaled within-layer weights.
"""

from spiyweb.edges.entity import build_entity_edges
from spiyweb.edges.semantic import build_semantic_edges
from spiyweb.edges.structural import ChunkRef, build_structural_edges

__all__ = [
    "ChunkRef",
    "build_entity_edges",
    "build_semantic_edges",
    "build_structural_edges",
]
