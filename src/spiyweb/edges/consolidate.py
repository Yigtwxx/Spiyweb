"""Offline consolidation pruning of never-used base-graph edges (D23).

The learned layer and consolidation are two faces of one coin: one thickens
the thread the spider uses, the other clears away the thread it never
touched. Pruning shrinks the index and makes the sparse matrix sparser -
but an edge nobody used may simply guard a question nobody has asked yet,
so Phase 1 pruning is deliberately cautious and REVERSIBLE: nothing is
deleted, `prune_layers` splits every layer into kept and removed lists and
feeding both back restores the original graph. Node merging (irreversible)
is deferred to Phase 2.

Usage evidence is the same as the learned layer's: the core deliberately
records no per-edge flows, so an edge "carried energy" when its endpoints
appear as a (contributor, node) pair in some run's activations. `EdgeUsage`
accumulates those pairs across runs; persistence is the caller's via
`to_dict` / `from_dict`. The `"learned"` layer is NOT pruned here - it has
its own `LearnedLayer.prune`, driven by strength instead of raw usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiyweb.config import ConsolidationConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from spiyweb.config import EdgeLayer
    from spiyweb.core.propagate import PropagationResult


class EdgeUsage:
    """Counts, per undirected edge, the runs in which it carried energy.

    `record` extracts the same evidence `LearnedLayer.reinforce` does - each
    activation's (contributor, node) pairs, self-loops skipped, keys
    canonicalised as the sorted pair - and increments a counter per pair
    plus a global run counter. Colored runs: record each `per_color` result
    separately - each colour is its own propagation and its own evidence.
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}
        self._runs = 0

    @property
    def runs(self) -> int:
        """Number of propagation runs recorded so far."""
        return self._runs

    def record(self, result: PropagationResult) -> int:
        """Count the edges `result` used; returns how many pairs were touched."""
        self._runs += 1
        touched = 0
        for node, activation in result.activations.items():
            for feeder in activation.contributors:
                if feeder == node:
                    continue
                key = (feeder, node) if feeder < node else (node, feeder)
                self._counts[key] = self._counts.get(key, 0) + 1
                touched += 1
        return touched

    def used(self, source: str, target: str) -> bool:
        """Whether the undirected edge ever carried energy, either direction."""
        key = (source, target) if source < target else (target, source)
        return self._counts.get(key, 0) > 0

    def to_dict(self) -> dict[str, int]:
        """Serialisable snapshot; keys are `"u\\tv"` with `u < v`, plus runs."""
        payload = {
            f"{source}\t{target}": count
            for (source, target), count in sorted(self._counts.items())
        }
        payload["\truns"] = self._runs
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, int]) -> EdgeUsage:
        """Rebuild a usage ledger from a `to_dict` snapshot."""
        usage = cls()
        for key, count in payload.items():
            if key == "\truns":
                usage._runs = int(count)
                continue
            source, separator, target = key.partition("\t")
            if not separator or not source or not target:
                raise ValueError(f"malformed edge-usage key: {key!r}")
            usage._counts[(source, target)] = int(count)
        return usage


@dataclass(frozen=True)
class ConsolidationReport:
    """Outcome of one pruning pass - kept and removed edges, per layer.

    Restoration is by construction: for every layer `kept + removed` is the
    input edge list (order preserved within each half), so persisting both
    keeps the pass reversible as Phase 1 requires.

    Attributes:
        kept: Surviving edges per layer, ready for `Graph.from_layers`.
        removed: Pruned edges per layer - the restorable archive.
        runs: How many recorded runs backed the decision.
    """

    kept: dict[EdgeLayer, list[tuple[str, str, float]]]
    removed: dict[EdgeLayer, list[tuple[str, str, float]]]
    runs: int


def prune_layers(
    layers: Mapping[EdgeLayer, Iterable[tuple[str, str, float]]],
    usage: EdgeUsage,
    config: ConsolidationConfig | None = None,
) -> ConsolidationReport:
    """Split each layer into edges that carried energy and edges that never did.

    An edge is removed exactly when its canonical pair never appears in the
    usage ledger AND at least `min_runs` runs were recorded - below that the
    evidence is too thin and everything is kept. Removed triples leave the
    list entirely; a weight of `0.0` is reserved for dedup-suppressed edges
    and must never mean "pruned".
    """
    config = config if config is not None else ConsolidationConfig()
    may_prune = usage.runs >= config.min_runs
    kept: dict[EdgeLayer, list[tuple[str, str, float]]] = {}
    removed: dict[EdgeLayer, list[tuple[str, str, float]]] = {}
    for layer, edges in layers.items():
        kept_edges: list[tuple[str, str, float]] = []
        removed_edges: list[tuple[str, str, float]] = []
        for edge in edges:
            source, target, _weight = edge
            if may_prune and not usage.used(source, target):
                removed_edges.append(edge)
            else:
                kept_edges.append(edge)
        kept[layer] = kept_edges
        removed[layer] = removed_edges
    return ConsolidationReport(kept=kept, removed=removed, runs=usage.runs)
