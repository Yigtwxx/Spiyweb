"""Negative seeds: an absorbing field over the excluded region.

"Excluding X" cannot be a post-filter: the filter removes X from the result,
but every chunk that arrived because it NEIGHBOURS X survives. Instead the
negative query spreads its own field with the ordinary propagation rules -
same damping, same proportional split - so the whole region around X becomes
absorbing, and positive energy entering it is destroyed on arrival.

The field itself is just a plain propagation run over the same graph; this
module only scales the budget and strips the result down to the node ->
absorbing-energy mapping the positive run consumes. Absorption events are
recorded per node, once, in the run's ledger - negative seeds, conflicts and
negative-polarity atoms (D34) are the only mechanisms allowed to destroy
energy (CLAUDE.md §2.1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spiyweb.config import PropagationConfig
    from spiyweb.core.graph import Graph


@dataclass(frozen=True)
class AbsorptionRecord:
    """One absorption event - the negative side of the destroyed-energy ledger.

    Attributes:
        node: The node whose arriving energy was (partly) destroyed.
        hop: The hop at which the node activated and the field fired.
        absorbed: Energy destroyed - `min(arriving, coefficient * field)`.
        field: The absorbing field held by the node when it fired.
        energy_before: The node's accumulated arrival before absorption.
        energy_after: What survived into the ranking.
    """

    node: str
    hop: int
    absorbed: float
    field: float
    energy_before: float
    energy_after: float


def negative_field(
    graph: Graph,
    seeds: Mapping[str, float],
    config: PropagationConfig,
    energy_ratio: float,
) -> dict[str, float]:
    """Spread the negative query's field and return node -> absorbing energy.

    The field runs under `energy_ratio * seed_energy` with the SAME rules as
    the positive web - its relative threshold scales with its own budget,
    exactly as a colour's share does. An empty seed set returns an empty
    field: an exclusion the index cannot even touch absorbs nothing, and that
    is a legitimate outcome, not an error.
    """
    # Imported here, not at module level: propagate() itself imports this
    # module for `AbsorptionRecord`, and a top-level import would be a cycle.
    from spiyweb.core.propagate import propagate

    if not seeds:
        return {}
    scaled = replace(config, seed_energy=config.seed_energy * energy_ratio)
    result = propagate(graph, seeds, scaled)
    return {node: activation.energy for node, activation in result.activations.items()}
