"""Negative seeds: the absorbing field, per-hop firing, retrieve glue.

The claim under test: "excluding X" is physics, not a post-filter - the
excluded region's field spreads with the ordinary rules, positive energy
activating inside it is destroyed (and recorded), and a node damped below
the threshold stops carrying energy onward, so the PATHS through the region
die, not just the node.
"""

from __future__ import annotations

import pytest

from spiyweb import (
    Graph,
    NegativeSeedConfig,
    PropagationConfig,
    negative_field,
    propagate,
    retrieve,
)

# ---------------------------------------------------------------- the field


def test_negative_field_spreads_with_the_ordinary_rules_and_scaled_budget() -> None:
    graph = Graph.from_edges([("X", "Y", 1.0)])
    field = negative_field(graph, {"X": 1.0}, PropagationConfig(), 0.5)
    # Budget 10 * 0.5 = 5; X keeps 5.0, Y receives 5 * .6 = 3.0.
    assert field == {"X": pytest.approx(5.0), "Y": pytest.approx(3.0)}


def test_negative_field_of_no_contacts_is_empty_not_an_error() -> None:
    graph = Graph.from_edges([("X", "Y", 1.0)])
    assert negative_field(graph, {}, PropagationConfig(), 1.0) == {}


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("seed_width", 0),
        ("energy_ratio", 0.0),
        ("coefficient", 0.0),
        ("coefficient", 1.5),
    ],
)
def test_negative_seed_config_rejects_out_of_range_values(
    field_name: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field_name.split("_")[0]):
        NegativeSeedConfig(**{field_name: value})


# ---------------------------------------------------------------- absorption

CHAIN = Graph.from_edges([("P", "M", 1.0), ("M", "T", 1.0)])


def test_energy_activating_inside_the_field_is_destroyed_and_recorded() -> None:
    result = propagate(
        CHAIN,
        {"P": 1.0},
        PropagationConfig(),
        absorb={"P": 10.0},
        negative_seed=NegativeSeedConfig(),
    )
    assert result.energy_of("P") == pytest.approx(0.0)
    assert "M" not in result.activations, "a fully absorbed seed spreads nothing"
    (record,) = result.absorptions
    assert record.node == "P"
    assert record.hop == 0
    assert record.absorbed == pytest.approx(10.0)
    assert record.energy_after == pytest.approx(0.0)


def test_the_paths_through_the_excluded_region_die_not_just_the_node() -> None:
    # Without the field: P=10 -> M=6 -> T=3.6. With M absorbing, T must
    # never light - that is the difference from a post-filter, which would
    # only have removed M from the result while T kept its energy.
    plain = propagate(CHAIN, {"P": 1.0}, PropagationConfig())
    assert plain.energy_of("T") == pytest.approx(3.6)

    absorbed = propagate(
        CHAIN,
        {"P": 1.0},
        PropagationConfig(),
        absorb={"M": 8.0},
        negative_seed=NegativeSeedConfig(),
    )
    assert absorbed.energy_of("M") == pytest.approx(0.0)
    assert "T" not in absorbed.activations
    (record,) = absorbed.absorptions
    assert record.hop == 1
    assert record.absorbed == pytest.approx(6.0), "capped by the arriving energy"


def test_partial_absorption_damps_but_does_not_kill_the_transit() -> None:
    result = propagate(
        CHAIN,
        {"P": 1.0},
        PropagationConfig(),
        absorb={"M": 2.0},
        negative_seed=NegativeSeedConfig(),
    )
    assert result.energy_of("M") == pytest.approx(4.0)  # 6 - 1.0 * 2.0
    assert result.energy_of("T") == pytest.approx(2.4), (
        "the survivor forwards its REMAINING energy"
    )
    (record,) = result.absorptions
    assert record.absorbed == pytest.approx(2.0), "capped by the field"


def test_disabled_config_behaves_exactly_as_no_field() -> None:
    plain = propagate(CHAIN, {"P": 1.0}, PropagationConfig())
    ablated = propagate(
        CHAIN,
        {"P": 1.0},
        PropagationConfig(),
        absorb={"M": 8.0},
        negative_seed=NegativeSeedConfig(enabled=False),
    )
    assert ablated.activations == plain.activations
    assert ablated.absorptions == ()


# ---------------------------------------------------------------- retrieve


class FakeIndex:
    """Contacts keyed by the exact query embedding."""

    def __init__(
        self, contacts: dict[tuple[float, ...], list[tuple[str, float]]]
    ) -> None:
        self._contacts = contacts

    def search(self, query: list[float], k: int) -> list[tuple[str, float]]:
        return self._contacts[tuple(query)][:k]


def test_retrieve_spreads_the_exclusion_field_and_reports_the_ledger() -> None:
    index = FakeIndex(
        {
            (1.0, 0.0): [("P", 0.9)],  # the question
            (0.0, 1.0): [("M", 0.8)],  # "excluding M"
        }
    )
    result = retrieve(
        [1.0, 0.0],
        index,
        CHAIN,
        negative_queries=[[0.0, 1.0]],
    )
    # M's field (budget 10) also covers its neighbours P and T with 3.0 each:
    # P activates with 10, loses 3; its remaining 4.2 arrives at M, which
    # absorbs all of it; T never receives positive energy at all.
    assert result.ranked()[0][0] == "P"
    assert result.propagation.energy_of("P") == pytest.approx(7.0)
    assert result.propagation.energy_of("M") == pytest.approx(0.0)
    assert "T" not in result.propagation.activations
    assert [record.node for record in result.absorptions] == ["P", "M"]
    assert sum(record.absorbed for record in result.absorptions) == pytest.approx(
        3.0 + 4.2
    )


def test_an_exclusion_touching_nothing_absorbs_nothing_silently() -> None:
    index = FakeIndex(
        {
            (1.0, 0.0): [("P", 0.9)],
            (0.0, 1.0): [("M", -0.2)],  # non-positive: no contact
        }
    )
    with_dead_exclusion = retrieve(
        [1.0, 0.0], index, CHAIN, negative_queries=[[0.0, 1.0]]
    )
    plain = retrieve([1.0, 0.0], index, CHAIN)
    assert with_dead_exclusion.absorptions == ()
    assert with_dead_exclusion.propagation.activations == plain.propagation.activations
