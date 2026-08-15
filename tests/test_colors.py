"""Coloured multi-seed activation (D12): parts of a decomposed query spread as
separately coloured webs, and a node where two colours meet is a bridge.

The load-bearing test is the chain: two colours injected at opposite ends of
`A - X - B` must meet exactly at `X`, with neither colour's web reaching the
other's seed. That meeting point is where the multi-hop answer lives.
"""

from __future__ import annotations

import pytest

from spiyweb import Graph, PropagationConfig, propagate
from spiyweb.core.colors import ColoredResult, propagate_colored

# damping 0.3: each colour crosses one edge and dies, so the webs overlap
# only at the middle node instead of flooding the whole chain.
CHAIN_EDGES = [("A", "X", 0.9), ("X", "B", 0.9)]
CHAIN_CONFIG = PropagationConfig(damping=0.3)
CHAIN_SEEDS = {"c1": {"A": 1.0}, "c2": {"B": 1.0}}


@pytest.fixture
def chain_result() -> ColoredResult:
    graph = Graph.from_edges(CHAIN_EDGES)
    return propagate_colored(graph, CHAIN_SEEDS, CHAIN_CONFIG)


def test_colors_meet_at_the_bridge_node(chain_result: ColoredResult) -> None:
    """`X` is reached by both colours; the seeds stay single-coloured."""
    assert chain_result.bridges == {"X": ("c1", "c2")}


def test_each_color_gets_an_equal_share_of_the_injection(
    chain_result: ColoredResult,
) -> None:
    """Two colours split the 10.0 injection into 5.0 webs - energy conserved."""
    assert chain_result.per_color["c1"].injected_energy == pytest.approx(5.0)
    assert chain_result.per_color["c2"].injected_energy == pytest.approx(5.0)
    assert chain_result.per_color["c1"].energy_of("A") == pytest.approx(5.0)


def test_combined_ranking_sums_energy_across_colors(
    chain_result: ColoredResult,
) -> None:
    """`X` accumulates from both colours: 2 * (5.0 * 0.3) = 3.0."""
    assert chain_result.energy_of("X") == pytest.approx(3.0)
    ranking = [node for node, _ in chain_result.ranked()]
    assert ranking == ["A", "B", "X"], f"unexpected ranking {ranking}"


def test_per_color_webs_die_at_their_own_relative_threshold(
    chain_result: ColoredResult,
) -> None:
    """Each colour's threshold scales with its own 5.0 budget (0.75 here)."""
    assert chain_result.per_color["c1"].threshold == pytest.approx(0.75)
    assert "B" not in chain_result.per_color["c1"].activations
    assert "A" not in chain_result.per_color["c2"].activations


def test_single_color_matches_plain_propagation() -> None:
    graph = Graph.from_edges(CHAIN_EDGES)
    colored = propagate_colored(graph, {"only": {"A": 1.0}}, CHAIN_CONFIG)
    plain = propagate(graph, {"A": 1.0}, CHAIN_CONFIG)
    assert colored.ranked() == plain.ranked()
    assert colored.bridges == {}


def test_empty_color_mapping_is_rejected() -> None:
    graph = Graph.from_edges(CHAIN_EDGES)
    with pytest.raises(ValueError):
        propagate_colored(graph, {}, CHAIN_CONFIG)


def test_color_without_seeds_is_rejected_by_name() -> None:
    graph = Graph.from_edges(CHAIN_EDGES)
    with pytest.raises(ValueError, match="c2"):
        propagate_colored(graph, {"c1": {"A": 1.0}, "c2": {}}, CHAIN_CONFIG)
