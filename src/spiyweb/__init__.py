"""Spiyweb - graph-based retrieval driven by spreading activation."""

from spiyweb.config import PropagationConfig
from spiyweb.core.graph import Graph
from spiyweb.core.propagate import Activation, PropagationResult, propagate

__version__ = "0.0.1"

__all__ = [
    "Activation",
    "Graph",
    "PropagationConfig",
    "PropagationResult",
    "__version__",
    "propagate",
]
