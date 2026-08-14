"""Contradiction as negative charge: opposing active atoms neutralise (D15).

Two chunks at cosine 0.9 can assert opposite things - embeddings do not encode
negation, so additive accumulation alone would *amplify* a false consensus.
Pre-marked negative edges (index-time NLI, `edges/nli.py`, D26) carry that
knowledge into the run; when both endpoints of one are active, they neutralise
like opposite charges: each side loses `coefficient * strength * min(E_a, E_b)`
- at full strength the weaker side dies entirely and the stronger keeps only
the difference. The owner picked this form (2026-08-14) over proportional
shaving because it is the physics metaphor verbatim and the ledger is a single
auditable number.

Conflict is the one mechanism besides negative seeds that is allowed to
DESTROY energy (CLAUDE.md §2.1); dedup only redistributes. Every absorption is
recorded, with the hop it happened at, so the ledger stays auditable.

Like the rest of `core/`, this module never detects anything: NLI runs at
index time outside the core, and this module only consumes pre-marked data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class NegativeEdge:
    """One pre-marked contradiction between two atoms.

    Attributes:
        source: One side of the contradiction.
        target: The other side.
        strength: NLI confidence in `(0, 1]`; scales the absorbed amount.
    """

    source: str
    target: str
    strength: float

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("negative edge endpoints must be non-empty ids")
        if self.source == self.target:
            raise ValueError(f"negative self-edge on {self.source!r}")
        if not 0.0 < self.strength <= 1.0:
            raise ValueError(
                f"negative edge strength {self.strength!r} must lie in (0, 1]"
            )


@dataclass(frozen=True)
class ConflictRecord:
    """One neutralisation event - the structured conflict datum of D16.

    `node_a < node_b` always (canonical pair order). Both sides lose exactly
    `absorbed_each`; the ledger's destroyed total is `absorbed_total`.
    """

    node_a: str
    node_b: str
    strength: float
    hop: int
    absorbed_each: float
    energy_a_before: float
    energy_a_after: float
    energy_b_before: float
    energy_b_after: float

    @property
    def absorbed_total(self) -> float:
        """Energy destroyed by this event - the auditable ledger entry."""
        return 2.0 * self.absorbed_each


def neutralize(
    energy_a: float, energy_b: float, strength: float, coefficient: float
) -> tuple[float, float, float]:
    """Apply charge neutralisation; return `(new_a, new_b, absorbed_each)`.

    Both sides lose the same amount, `coefficient * strength * min(a, b)` -
    equal quantities of opposite charge annihilate. The amount never exceeds
    the weaker side, so energies cannot go negative.
    """
    absorbed = coefficient * strength * min(energy_a, energy_b)
    return energy_a - absorbed, energy_b - absorbed, absorbed


def conflict_adjacency(
    edges: Iterable[NegativeEdge],
) -> dict[str, dict[str, float]]:
    """Symmetric node -> {opponent: strength} view of the negative edges.

    A pair marked more than once keeps its STRONGEST evidence - two NLI hits
    on the same pair are corroboration, not accumulation (accumulating
    confidences above 1.0 would be meaningless).
    """
    adjacency: dict[str, dict[str, float]] = {}
    for edge in edges:
        for a, b in ((edge.source, edge.target), (edge.target, edge.source)):
            per_node = adjacency.setdefault(a, {})
            per_node[b] = max(per_node.get(b, 0.0), edge.strength)
    return adjacency
