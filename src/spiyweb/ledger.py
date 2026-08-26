"""The energy ledger: where the injected energy actually went.

Promoted out of `server/` in Faz 2.5. It was written for the browser face,
but it is pure arithmetic over a propagation result - no HTTP, no numpy, no
I/O - and a trace record carries one now, so a reader with nothing installed
can still audit a run's energy. Keeping it behind a FastAPI extra would have
meant the zero-dependency trace reader could not open its own files.

CLAUDE.md §2.1 makes an auditable claim — dedup REDISTRIBUTES energy, while
contradictions, negative seeds and negative-polarity atoms DESTROY it, and
nothing else creates or destroys any. This module turns that claim into four
numbers a reader can check.

The core keeps no ledger, so this is a RECONSTRUCTION from the result:

    landed(n)    = activation energy + everything destroyed at n
                   (destruction is recorded, so this part is exact)
    forwarded(n) = damping**(1/mass) * activation energy, when n actually
                   distributed - the one PREDICTED quantity
    held(n)      = activation energy - forwarded(n)

Because `forwarded` is predicted, the reconstruction can be wrong, and the
honest design is to let it be caught rather than hidden. So the arrivals are
replayed through the same live-neighbour rule `_distribute` uses, and two
independent discrepancies are reported:

    dissipated = replayed arrivals that reached an atom which never activated
                 (it fell under the threshold, or the overflow guard trimmed it)
    mismatch   = |replayed arrival - actual landed| over atoms that DID activate

A correct reconstruction leaves `residual` and `mismatch` at zero. Anything
else is a finding, and the UI is required to say so rather than round it away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spiyweb.config import PropagationConfig
    from spiyweb.core.graph import Graph
    from spiyweb.core.propagate import PropagationResult

__all__ = ["Destroyed", "Ledger", "build_ledger", "destroyed_per_node"]


@dataclass(frozen=True)
class Destroyed:
    """The three mechanisms allowed to destroy energy, kept apart."""

    conflict: float
    conflict_events: int
    negative_seed: float
    negative_seed_events: int
    polarity: float
    polarity_events: int

    @property
    def total(self) -> float:
        return self.conflict + self.negative_seed + self.polarity


@dataclass(frozen=True)
class Ledger:
    """One run's energy accounting, plus how much of it failed to add up."""

    injected: float
    held: float
    dissipated: float
    destroyed: Destroyed
    residual: float
    mismatch: float
    tolerance: float
    balanced: bool
    exact: bool
    notes: tuple[str, ...]
    dedup_cuts: int
    dedup_taus: tuple[float, ...]


def destroyed_per_node(result: PropagationResult) -> dict[str, float]:
    """Energy destroyed at each atom, summed over the three mechanisms."""
    per_node: dict[str, float] = {}
    for record in result.conflicts:
        for node in (record.node_a, record.node_b):
            per_node[node] = per_node.get(node, 0.0) + record.absorbed_each
    for absorption in result.absorptions:
        per_node[absorption.node] = (
            per_node.get(absorption.node, 0.0) + absorption.absorbed
        )
    for dispute in result.disputes:
        per_node[dispute.node] = per_node.get(dispute.node, 0.0) + dispute.absorbed
    return per_node


def _destroyed_totals(result: PropagationResult) -> Destroyed:
    return Destroyed(
        conflict=sum(record.absorbed_total for record in result.conflicts),
        conflict_events=len(result.conflicts),
        negative_seed=sum(record.absorbed for record in result.absorptions),
        negative_seed_events=len(result.absorptions),
        polarity=sum(record.absorbed for record in result.disputes),
        polarity_events=len(result.disputes),
    )


def _live_neighbours(
    graph: Graph,
    node: str,
    hop: int,
    result: PropagationResult,
    split_alpha: float,
) -> dict[str, float]:
    """The targets `_distribute` would have fed, with their split weights.

    Mirrors the core's filter exactly: a positive edge, a target that had not
    already activated at or before this hop, and a target dedup did not
    suppress. Suppressed atoms never activate, so excluding them here is the
    same statement as the core's `is_duplicate` check.
    """
    activations = result.activations
    live: dict[str, float] = {}
    for target, weight in graph.neighbors(node).items():
        if weight <= 0.0 or target == node or target in result.suppressed:
            continue
        landed = activations.get(target)
        if landed is not None and landed.hop <= hop:
            continue
        live[target] = weight**split_alpha
    return live


def build_ledger(
    result: PropagationResult,
    graph: Graph,
    config: PropagationConfig,
) -> Ledger:
    """Reconstruct the ledger and report how well it adds up."""
    from spiyweb.core.mass import node_masses

    activations = result.activations
    destroyed_at = destroyed_per_node(result)
    masses = node_masses(graph, config.mass)
    notes: list[str] = []
    if config.mass.enabled:
        notes.append(
            "node mass is on, so carry is damping**(1/mass) and the "
            "reconstruction is more sensitive to per-atom rounding"
        )

    landed = {
        node: activation.energy + destroyed_at.get(node, 0.0)
        for node, activation in activations.items()
    }
    by_hop: dict[int, list[str]] = {}
    for node, activation in activations.items():
        by_hop.setdefault(activation.hop, []).append(node)

    # The last activated hop may not have distributed at all: the overflow
    # guard breaks the loop before `_distribute` runs.
    trimmed = result.stop_reason == "max_nodes"
    if trimmed:
        notes.append(
            "the max_nodes overflow guard fired: the final frontier was "
            "trimmed and its energy is genuinely lost, not held"
        )
    if result.stop_reason == "max_hop":
        notes.append("the max_hop guard fired before the last arrivals landed")

    held = 0.0
    dissipated = 0.0
    mismatch = 0.0
    for hop in sorted(by_hop):
        arrivals: dict[str, float] = {}
        for node in by_hop[hop]:
            energy = activations[node].energy
            mass = masses.get(node, 1.0)
            carry = config.damping if mass == 1.0 else config.damping ** (1.0 / mass)
            live = (
                {}
                if (trimmed and hop == result.hops_used)
                else _live_neighbours(graph, node, hop, result, config.split_alpha)
            )
            total_weight = sum(live.values())
            if total_weight <= 0.0:
                held += energy
                continue
            outgoing = energy * carry
            held += energy - outgoing
            for target, weight in live.items():
                arrivals[target] = (
                    arrivals.get(target, 0.0) + outgoing * weight / total_weight
                )
        for target, amount in arrivals.items():
            actual = activations.get(target)
            if actual is None or actual.hop != hop + 1:
                # Never landed: under the threshold, or trimmed away.
                dissipated += amount
            else:
                mismatch += abs(amount - landed[target])

    destroyed = _destroyed_totals(result)
    injected = result.injected_energy
    residual = injected - (held + dissipated + destroyed.total)
    tolerance = 1e-6 * max(1.0, injected)
    dead_ends = sum(
        1
        for node, activation in activations.items()
        if not _live_neighbours(graph, node, activation.hop, result, config.split_alpha)
    )
    if dead_ends:
        notes.append(f"{dead_ends} atom(s) had no live neighbour and held everything")
    return Ledger(
        injected=injected,
        held=held,
        dissipated=dissipated,
        destroyed=destroyed,
        residual=residual,
        mismatch=mismatch,
        tolerance=tolerance,
        balanced=abs(residual) <= tolerance and mismatch <= tolerance,
        exact=not config.mass.enabled,
        notes=tuple(notes),
        dedup_cuts=len(result.suppressed),
        dedup_taus=result.dedup_thresholds,
    )
