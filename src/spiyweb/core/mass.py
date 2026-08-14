"""Node mass (D11): length-proportional inertia, normalised per layer.

A long atom activates late but, once lit, carries further; a short one is
the opposite. Raw length across layers would leave the proposition layer
permanently dark - propositions are short by definition - so mass is ALWAYS
computed against the mean length of the node's OWN layer. Mass never enters
edge weights or the proportional split: it gates a node's own activation
and scales its own forwarding, nothing else - which is the structural form
of the rule that mass is inert on cross-layer links.

Like everything in `core/`, this is pure computation: lengths and layers
come from `Graph.node_data`, numbers go out.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from spiyweb.config import MassConfig

if TYPE_CHECKING:
    from spiyweb.core.graph import Graph


def node_masses(graph: Graph, config: MassConfig) -> dict[str, float]:
    """Effective mass per node: `clamp((len / layer_mean) ** exp, floor, cap)`.

    Nodes absent from `node_data` are simply not listed - consumers treat a
    missing entry as mass `1.0`, the massless-world default. A disabled
    config returns an empty map for the same reason. Uniform lengths within
    a layer produce mass `1.0` everywhere, so a corpus of equal-sized atoms
    behaves exactly as if mass did not exist.
    """
    if not config.enabled or not graph.node_data:
        return {}
    totals: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for node in graph.node_data.values():
        totals[node.layer] += float(node.length)
        counts[node.layer] += 1
    means = {layer: totals[layer] / counts[layer] for layer in totals}
    masses: dict[str, float] = {}
    for node_id, node in graph.node_data.items():
        relative = float(node.length) / means[node.layer]
        mass = relative**config.exponent
        masses[node_id] = min(config.cap, max(config.floor, mass))
    return masses
