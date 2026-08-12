"""Graph construction rules that the propagation depends on."""

from __future__ import annotations

import pytest

from spiyweb import Graph
from spiyweb.config import PropagationConfig


def test_edges_are_undirected_by_default() -> None:
    graph = Graph.from_edges([("A", "B", 0.7)])
    assert graph.neighbors("A")["B"] == pytest.approx(0.7)
    assert graph.neighbors("B")["A"] == pytest.approx(0.7)


def test_directed_edges_stay_one_way() -> None:
    graph = Graph.from_edges([("A", "B", 0.7)], undirected=False)
    assert "B" in graph.neighbors("A")
    assert graph.neighbors("B") == {}
    assert graph.nodes == {"A", "B"}


def test_suppressed_edge_keeps_the_node_but_carries_no_energy() -> None:
    graph = Graph.from_edges([("A", "A_dup", 0.0)])
    assert "A_dup" in graph.nodes
    assert graph.neighbors("A")["A_dup"] == 0.0


def test_negative_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative charge"):
        Graph.from_edges([("A", "B", -0.5)])


def test_unknown_node_has_no_neighbors() -> None:
    graph = Graph.from_edges([("A", "B", 1.0)])
    assert graph.neighbors("ghost") == {}
    assert len(graph) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed_energy": 0.0},
        {"damping": 0.0},
        {"damping": 1.0},
        {"threshold_ratio": 1.0},
        {"threshold_ratio": -0.1},
        {"max_hop": -1},
        {"max_nodes": 0},
    ],
)
def test_config_rejects_values_outside_its_range(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        PropagationConfig(**kwargs)


def test_threshold_is_relative_to_the_injected_energy() -> None:
    assert PropagationConfig().threshold == pytest.approx(1.5)
    assert PropagationConfig(seed_energy=4.0).threshold == pytest.approx(0.6)
