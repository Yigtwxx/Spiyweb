"""Spreading activation: multiplicative decay, proportional split, accumulation.

The query is not a filter over the graph, it is an energy seed injected into it.
Each activated node forwards `damping` of its energy, split among its still
inactive neighbours in proportion to edge weight, and whatever arrives at a node
from several paths is summed. Energy that falls below the relative threshold
dies there, which is what makes the web stop on its own instead of at `top-k`.

Formally this is a truncated Personalized PageRank; practically it is a handful
of dictionary passes over a sparse graph.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from spiyweb.config import PropagationConfig
from spiyweb.core.graph import Graph

StopReason = Literal["threshold", "max_hop", "max_nodes"]


@dataclass(frozen=True)
class Activation:
    """One node that the web reached, and how it got there."""

    energy: float
    hop: int
    contributors: tuple[str, ...]
    """Nodes that fed this one. More than one means converging evidence."""


@dataclass(frozen=True)
class PropagationResult:
    """Everything a single run produced, plus why it stopped."""

    activations: Mapping[str, Activation]
    injected_energy: float
    threshold: float
    hops_used: int
    stop_reason: StopReason

    def ranked(self) -> list[tuple[str, float]]:
        """Activated nodes by accumulated energy, strongest first.

        Ties break on node id so the ordering is stable across platforms; a
        freshness tie-breaker replaces that once nodes carry timestamps.
        """
        return sorted(
            ((node, act.energy) for node, act in self.activations.items()),
            key=lambda item: (-item[1], item[0]),
        )

    def energy_of(self, node: str) -> float:
        """Accumulated energy of `node`, or `0.0` if the web never reached it."""
        activation = self.activations.get(node)
        return activation.energy if activation is not None else 0.0


def propagate(
    graph: Graph,
    seeds: Mapping[str, float],
    config: PropagationConfig | None = None,
) -> PropagationResult:
    """Inject the query into `graph` at `seeds` and let it spread until it dies.

    Args:
        graph: The graph to spread over.
        seeds: Contact points of the query, mapped to their contact strength
            (cosine similarity, in practice). The injected energy is split
            among them in proportion to that strength, exactly the way a node
            later splits energy among its neighbours. Callers pass similarities;
            this module never computes them.
        config: Propagation settings; defaults to `PropagationConfig()`.

    Returns:
        The activated set with accumulated energies, and the reason the web
        stopped growing.

    Raises:
        ValueError: If `seeds` is empty or its weights do not sum above zero.
    """
    config = config or PropagationConfig()
    threshold = config.threshold

    frontier = _inject(seeds, config.seed_energy)
    contributors: defaultdict[str, list[str]] = defaultdict(list)
    activations: dict[str, Activation] = {}
    stop_reason: StopReason = "threshold"
    hop = 0
    hops_used = 0

    while frontier:
        if hop > config.max_hop:
            stop_reason = "max_hop"
            break

        room = config.max_nodes - len(activations)
        if len(frontier) > room:
            # Overflow guard, not a ranking mechanism: keep the strongest and
            # say so through `stop_reason` rather than truncating in silence.
            frontier = dict(
                sorted(frontier.items(), key=lambda item: (-item[1], item[0]))[:room]
            )
            stop_reason = "max_nodes"

        for node, energy in frontier.items():
            activations[node] = Activation(
                energy=energy,
                hop=hop,
                contributors=tuple(contributors[node]),
            )
        hops_used = hop

        if stop_reason == "max_nodes":
            break

        arrivals = _distribute(
            graph, frontier, activations, config.damping, contributors
        )
        # The threshold applies to a node's *accumulated* arrival, never to a
        # single contribution: two weak paths that meet must be allowed to add
        # up. That is where the multi-hop answer lives.
        frontier = {
            node: energy for node, energy in arrivals.items() if energy >= threshold
        }
        hop += 1

    return PropagationResult(
        activations=activations,
        injected_energy=config.seed_energy,
        threshold=threshold,
        hops_used=hops_used,
        stop_reason=stop_reason,
    )


def _inject(seeds: Mapping[str, float], seed_energy: float) -> dict[str, float]:
    """Split the injected energy across the seed contacts, weighted by strength."""
    if not seeds:
        raise ValueError("at least one seed node is required")
    total = sum(seeds.values())
    if total <= 0.0:
        raise ValueError("seed weights must sum to a positive value")
    return {node: seed_energy * weight / total for node, weight in seeds.items()}


def _distribute(
    graph: Graph,
    frontier: Mapping[str, float],
    activated: Mapping[str, Activation],
    damping: float,
    contributors: defaultdict[str, list[str]],
) -> dict[str, float]:
    """Forward `damping` of each frontier node's energy to its live neighbours.

    Suppressed edges (weight `0.0`) and already activated neighbours are left
    out of the denominator, so their share is redistributed over the remaining
    neighbours instead of leaking out of the run. Dedup redistributes energy;
    only negative seeds and contradictions are allowed to destroy it.
    """
    arrivals: dict[str, float] = defaultdict(float)
    for node, energy in frontier.items():
        outgoing = energy * damping
        live = {
            target: weight
            for target, weight in graph.neighbors(node).items()
            if weight > 0.0 and target not in activated and target not in frontier
        }
        total_weight = sum(live.values())
        if total_weight <= 0.0:
            continue
        for target, weight in live.items():
            arrivals[target] += outgoing * weight / total_weight
            contributors[target].append(node)
    return dict(arrivals)
