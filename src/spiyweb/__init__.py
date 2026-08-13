"""Spiyweb - graph-based retrieval driven by spreading activation."""

from spiyweb.config import EdgeLayer, LayerWeights, PropagationConfig
from spiyweb.core.graph import Graph, Node, NodeLayer, Polarity
from spiyweb.core.propagate import Activation, PropagationResult, propagate

__version__ = "0.0.1"

__all__ = [
    "Activation",
    "EdgeLayer",
    "Graph",
    "LayerWeights",
    "Node",
    "NodeLayer",
    "Polarity",
    "PropagationConfig",
    "PropagationResult",
    "__version__",
    "propagate",
]
