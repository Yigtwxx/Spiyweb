"""Sparse weighted graph over node identifiers, plus the node data model.

This is deliberately the smallest structure the propagation needs: an adjacency
mapping, an optional registry of node metadata, and nothing else. The edge
layers (semantic / entity / structural / learned) are merged into these weights
by `Graph.from_layers` *before* propagation ever sees the graph, so adding an
edge layer never requires touching the propagation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from spiyweb.config import EdgeLayer, LayerWeights

_EMPTY: Mapping[str, float] = MappingProxyType({})

NodeLayer = Literal["chunk", "proposition"]
_NODE_LAYERS: tuple[NodeLayer, ...] = ("chunk", "proposition")

Polarity = Literal[1, -1]
_POLARITIES: tuple[Polarity, ...] = (1, -1)


@dataclass(frozen=True)
class Node:
    """Metadata of one atom in the web - a chunk or a proposition.

    Node mass is *not* computed here and is not yet wired into the
    propagation: `length` and `layer` are the raw inputs the future mass
    formula will consume, normalised within each layer so the proposition
    layer never goes dark against the longer chunks.

    Attributes:
        id: Graph node identifier; matches the adjacency keys.
        layer: Which of the two node layers the atom lives in.
        source_id: Document / source the atom came from. Redundancy votes are
            counted per source, never per chunk.
        length: Token or character count; raw input to the future mass formula.
        timestamp: Seconds since the Unix epoch, UTC; `None` when unknown.
            Freshness is a ranking tie-breaker only, never a continuous
            multiplier, so total ordering is the only operation ever needed.
        cluster_id: Theme cluster the atom belongs to; `None` until clustering
            exists.
        polarity: `+1` for ordinary atoms, `-1` for negative-knowledge atoms
            (negated propositions that absorb the energy of queries asserting
            the opposite). The schema lands now; the absorbing mechanism is a
            post-measurement ablation.
    """

    id: str
    layer: NodeLayer
    source_id: str
    length: int
    timestamp: float | None = None
    cluster_id: str | None = None
    polarity: Polarity = 1

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("node id must not be empty")
        if self.layer not in _NODE_LAYERS:
            raise ValueError(
                f"unknown node layer {self.layer!r}; expected one of {_NODE_LAYERS}"
            )
        if not self.source_id:
            raise ValueError(f"node {self.id!r} must carry a non-empty source_id")
        if self.length < 1:
            raise ValueError(
                f"node {self.id!r} has length {self.length!r}; must be at least 1"
            )
        if self.polarity not in _POLARITIES:
            raise ValueError(
                f"node {self.id!r} has polarity {self.polarity!r}; must be +1 or -1"
            )


@dataclass(frozen=True)
class Graph:
    """Directed weighted adjacency; build it with `from_edges` or `from_layers`."""

    adjacency: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    node_data: Mapping[str, Node] = field(default_factory=dict)
    """Optional metadata registry. It may cover only part of the adjacency and
    may mention nodes the adjacency does not - it is carried, not enforced."""

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[str, str, float]],
        *,
        undirected: bool = True,
        nodes: Iterable[Node] = (),
    ) -> Graph:
        """Build a graph from `(source, target, weight)` triples.

        A weight of exactly `0.0` is meaningful: it marks a suppressed edge -
        the state a near-duplicate neighbour is left in once redundancy has been
        converted into a vote. The node stays in the graph, but no energy
        crosses the edge, and the propagation renormalises the remaining
        neighbours over the share it left behind.

        Negative weights are rejected for now. Contradiction is modelled as
        negative charge, but that mechanism lands in `core/conflict.py` with its
        own semantics rather than as a silently negative adjacency entry.
        """
        return cls(
            adjacency=_build_adjacency(edges, undirected=undirected),
            node_data=_build_registry(nodes),
        )

    @classmethod
    def from_layers(
        cls,
        layers: Mapping[EdgeLayer, Iterable[tuple[str, str, float]]],
        *,
        weights: LayerWeights | None = None,
        undirected: bool = True,
        nodes: Iterable[Node] = (),
    ) -> Graph:
        """Merge per-layer edge lists into one weighted adjacency.

        The merged weight of an edge is the weighted sum over the layers where
        the edge exists: `merged[u][v] = sum(weights.weight_of(L) * w_L(u, v))`.
        A sum, not an average - an edge confirmed by two layers must come out
        stronger than either alone, the same additive-evidence rule the
        propagation applies to converging energy.

        A *layer* weight of `0.0` disables the layer before anything is built:
        neither its edges nor nodes only it mentions enter the graph. An *edge*
        weight of `0.0` inside an enabled layer keeps the suppressed-edge
        contract of `from_edges`: the entry survives at `0.0` unless another
        layer lends the edge positive weight. Query-time dedup suppression
        operates on the merged graph; a per-layer `0.0` is just an inert edge,
        not a vote signal.
        """
        weights = weights or LayerWeights()
        merged: dict[str, dict[str, float]] = {}
        for layer, edges in layers.items():
            layer_weight = weights.weight_of(layer)
            if layer_weight == 0.0:
                continue
            for node, neighbors in _build_adjacency(
                edges, undirected=undirected
            ).items():
                merged_neighbors = merged.setdefault(node, {})
                for target, weight in neighbors.items():
                    merged_neighbors[target] = (
                        merged_neighbors.get(target, 0.0) + layer_weight * weight
                    )
        return cls(adjacency=merged, node_data=_build_registry(nodes))

    def neighbors(self, node: str) -> Mapping[str, float]:
        """Outgoing edges of `node`; empty when the node is unknown."""
        return self.adjacency.get(node, _EMPTY)

    def node(self, node_id: str) -> Node | None:
        """Metadata of `node_id`, or `None` when the registry does not know it."""
        return self.node_data.get(node_id)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self.adjacency)

    def __len__(self) -> int:
        return len(self.adjacency)


def _build_adjacency(
    edges: Iterable[tuple[str, str, float]],
    *,
    undirected: bool,
) -> dict[str, dict[str, float]]:
    """Adjacency of one edge list; duplicates are last-write-wins."""
    built: dict[str, dict[str, float]] = {}
    for source, target, weight in edges:
        if weight < 0.0:
            raise ValueError(
                f"negative edge weight {weight!r} between {source!r} and "
                f"{target!r}; negative charge is not an adjacency weight"
            )
        built.setdefault(source, {})[target] = weight
        built.setdefault(target, {})
        if undirected:
            built[target][source] = weight
    return built


def _build_registry(nodes: Iterable[Node]) -> dict[str, Node]:
    registry: dict[str, Node] = {}
    for node in nodes:
        if node.id in registry:
            raise ValueError(f"duplicate node id {node.id!r} in the registry")
        registry[node.id] = node
    return registry
