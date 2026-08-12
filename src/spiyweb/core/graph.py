"""Sparse weighted graph over node identifiers.

This is deliberately the smallest structure the propagation needs: an adjacency
mapping and nothing else. Layer merging (semantic / entity / structural /
learned) collapses into these weights *before* a `Graph` is built, so adding an
edge layer never requires touching this module or the propagation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

_EMPTY: Mapping[str, float] = MappingProxyType({})


@dataclass(frozen=True)
class Graph:
    """Directed weighted adjacency; build it with `Graph.from_edges`."""

    adjacency: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[str, str, float]],
        *,
        undirected: bool = True,
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
        return cls(adjacency={node: dict(edges_) for node, edges_ in built.items()})

    def neighbors(self, node: str) -> Mapping[str, float]:
        """Outgoing edges of `node`; empty when the node is unknown."""
        return self.adjacency.get(node, _EMPTY)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self.adjacency)

    def __len__(self) -> int:
        return len(self.adjacency)
