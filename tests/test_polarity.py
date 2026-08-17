"""Negative-knowledge atoms (D34): the corpus's "no" absorbs the claim.

The claim under test: a `polarity == -1` atom destroys the energy that
reaches it - the opposing query's evidence dies there instead of flowing on
- the event lands in the dispute ledger, the warning is template-built, and
`enabled=False` makes the atom an ordinary node (the ablation switch).
"""

from __future__ import annotations

import pytest

from spiyweb import (
    Graph,
    Node,
    PolarityConfig,
    PropagationConfig,
    dispute_warnings,
    propagate,
    retrieve,
)

# A -> B -> C chain; B is the negative atom ("B says the opposite").
# Default run without polarity: A 10.0, B 6.0, C 3.6.
NEGATIVE_B = Node(id="B", layer="chunk", source_id="doc-b", length=10, polarity=-1)
CHAIN = Graph.from_edges(
    [("A", "B", 0.5), ("B", "C", 0.5)],
    nodes=[NEGATIVE_B],
)


def test_negative_atom_absorbs_fully_and_stops_the_spread() -> None:
    result = propagate(
        CHAIN, {"A": 1.0}, PropagationConfig(), polarity=PolarityConfig()
    )
    assert result.energy_of("B") == pytest.approx(0.0)
    assert "C" not in result.activations, (
        "the opposing claim's evidence dies AT the atom - it must not flow on"
    )
    record = result.disputes[0]
    assert record.node == "B"
    assert record.absorbed == pytest.approx(6.0)
    assert record.energy_before == pytest.approx(6.0)
    assert record.energy_after == pytest.approx(0.0)


def test_partial_coefficient_leaves_a_weakened_atom_spreading() -> None:
    result = propagate(
        CHAIN,
        {"A": 1.0},
        PropagationConfig(),
        polarity=PolarityConfig(coefficient=0.5),
    )
    # B keeps 3.0 >= threshold 1.5 and forwards 3.0 * .6 = 1.8 to C.
    assert result.energy_of("B") == pytest.approx(3.0)
    assert result.energy_of("C") == pytest.approx(1.8)
    assert result.disputes[0].absorbed == pytest.approx(3.0)


def test_disabled_polarity_makes_the_atom_an_ordinary_node() -> None:
    result = propagate(
        CHAIN,
        {"A": 1.0},
        PropagationConfig(),
        polarity=PolarityConfig(enabled=False),
    )
    assert result.energy_of("B") == pytest.approx(6.0), "the ablation switch"
    assert result.energy_of("C") == pytest.approx(3.6)
    assert result.disputes == ()


def test_no_polarity_config_changes_nothing() -> None:
    result = propagate(CHAIN, {"A": 1.0}, PropagationConfig())
    assert result.energy_of("C") == pytest.approx(3.6)
    assert result.disputes == ()


def test_a_seeded_negative_atom_fires_at_hop_zero() -> None:
    # The query lands DIRECTLY on the atom that denies it - embeddings do not
    # carry negation, so this is the D34 headline case.
    result = propagate(
        CHAIN, {"B": 1.0}, PropagationConfig(), polarity=PolarityConfig()
    )
    assert result.energy_of("B") == pytest.approx(0.0)
    assert result.disputes[0].hop == 0
    assert result.disputes[0].absorbed == pytest.approx(10.0)


def test_dispute_warning_is_template_built() -> None:
    result = propagate(
        CHAIN, {"A": 1.0}, PropagationConfig(), polarity=PolarityConfig()
    )
    warning = dispute_warnings(result.disputes)[0]
    assert warning.node == "B"
    assert warning.absorbed == pytest.approx(6.0)
    assert "corpus disputes" in warning.message
    assert "B" in warning.message


def test_retrieve_surfaces_the_dispute_ledger() -> None:
    class FakeIndex:
        def search(self, query: object, k: int) -> list[tuple[str, float]]:
            return [("A", 0.9)]

    result = retrieve([1.0], FakeIndex(), CHAIN, polarity=PolarityConfig())
    assert result.disputes[0].node == "B"


def test_config_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="coefficient"):
        PolarityConfig(coefficient=0.0)
    with pytest.raises(ValueError, match="coefficient"):
        PolarityConfig(coefficient=1.5)
