"""Negative-knowledge atoms (D34): the dispute ledger's record type.

A negated proposition ("X does not ...") enters the box as a PERMANENT
negative-polarity atom (`Node.polarity == -1`). Polarity detection happens at
index time, outside `core/` - this package only consumes the pre-marked
label, exactly as it consumes pre-marked NLI edges. When query energy reaches
such an atom, the atom absorbs it (see `PolarityConfig`): the opposing
claim's evidence dies instead of reinforcing it, and the event recorded here
is what the caller's "corpus disputes this" warning is built from
(`spiyweb.output.dispute_warnings`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisputeRecord:
    """One negative-polarity absorption event - proof of a corpus dispute.

    Attributes:
        node: The negative-polarity atom that fired.
        hop: Hop at which the query's energy reached the atom.
        absorbed: Energy destroyed - `coefficient * energy_before`.
        energy_before: The atom's accumulated energy at activation.
        energy_after: What remained after the absorption.
    """

    node: str
    hop: int
    absorbed: float
    energy_before: float
    energy_after: float
