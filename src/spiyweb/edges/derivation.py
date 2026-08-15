"""Derivation edges: chunk -> proposition containment links (D10).

The bridge between the two node layers, in its own edge layer (owner's
2026-08-14 choice): a proposition is linked to exactly the chunk it was
extracted from. Within-layer weight is uniform `1.0` - containment is not
graded - and the layer's strength against the others lives in
`LayerWeights.derivation`, where `0.0` cuts the layers apart (the ablation
switch). Like every builder here: no model, no I/O, raw `(u, v, weight)`
triples for `Graph.from_layers`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spiyweb.nodes.propositions import Proposition


def build_derivation_edges(
    propositions: Sequence[Proposition],
) -> list[tuple[str, str, float]]:
    """One `(chunk_id, proposition_id, 1.0)` edge per proposition."""
    return [
        (proposition.chunk_id, proposition.node.id, 1.0) for proposition in propositions
    ]
