"""Spiyweb - graph-based retrieval driven by spreading activation."""

from spiyweb.config import (
    EdgeLayer,
    EmbeddingConfig,
    EntityEdgeConfig,
    EntityExtractionConfig,
    LayerWeights,
    LLMConfig,
    PropagationConfig,
    SemanticEdgeConfig,
    StructuralEdgeConfig,
)
from spiyweb.core.graph import Graph, Node, NodeLayer, Polarity
from spiyweb.core.propagate import Activation, PropagationResult, propagate

__version__ = "0.0.1"

__all__ = [
    "Activation",
    "EdgeLayer",
    "EmbeddingConfig",
    "EntityEdgeConfig",
    "EntityExtractionConfig",
    "Graph",
    "LLMConfig",
    "LayerWeights",
    "Node",
    "NodeLayer",
    "Polarity",
    "PropagationConfig",
    "PropagationResult",
    "SemanticEdgeConfig",
    "StructuralEdgeConfig",
    "__version__",
    "propagate",
]
