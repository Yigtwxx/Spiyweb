"""Node mass (D11): per-layer inertia - late to light, far to carry.

The claim under test: mass is length normalised against the node's OWN
layer (cross-layer raw length would leave propositions dark), it gates a
node's activation (`threshold * mu`) and scales its forwarding
(`damping ** (1 / mu)`), and every neutral path - disabled config, uniform
lengths, exponent 0 - reproduces today's massless behaviour exactly.
"""

from __future__ import annotations

import pytest

from spiyweb import (
    Graph,
    MassConfig,
    Node,
    PropagationConfig,
    node_masses,
    propagate,
)


def make_node(node_id: str, length: int, layer: str = "chunk") -> Node:
    return Node(
        id=node_id,
        layer=layer,  # type: ignore[arg-type]
        source_id=f"doc-{node_id}",
        length=length,
    )


def test_mass_is_normalised_within_each_layer() -> None:
    # Chunk layer mean 200; proposition layer mean 40. The proposition of
    # length 60 is HEAVY in its own layer despite being short in characters.
    graph = Graph.from_edges(
        [("A", "B", 1.0)],
        nodes=[
            make_node("A", 300),
            make_node("B", 100),
            make_node("P1", 60, layer="proposition"),
            make_node("P2", 20, layer="proposition"),
        ],
    )
    masses = node_masses(graph, MassConfig(enabled=True))
    assert masses["A"] == pytest.approx(1.5)
    assert masses["B"] == pytest.approx(0.5)
    assert masses["P1"] == pytest.approx(1.5), (
        "a proposition weighs against other propositions, never against chunks"
    )
    assert masses["P2"] == pytest.approx(0.5)


def test_mass_is_clamped_to_floor_and_cap() -> None:
    graph = Graph.from_edges(
        [("A", "B", 1.0)],
        nodes=[make_node("A", 1000), make_node("B", 10), make_node("C", 10)],
    )
    masses = node_masses(graph, MassConfig(enabled=True))
    assert masses["A"] == pytest.approx(2.0), "cap"
    assert masses["B"] == pytest.approx(0.5), "floor"


def test_disabled_or_flat_configs_are_massless() -> None:
    graph = Graph.from_edges(
        [("A", "B", 1.0)], nodes=[make_node("A", 300), make_node("B", 100)]
    )
    assert node_masses(graph, MassConfig()) == {}, "disabled by default"
    flat = node_masses(graph, MassConfig(enabled=True, exponent=0.0))
    assert set(flat.values()) == {1.0}, "exponent 0 is the second neutral path"


def test_uniform_lengths_reproduce_the_massless_run_exactly() -> None:
    edges = [("A", "B", 0.5), ("B", "C", 0.5)]
    nodes = [make_node(node_id, 100) for node_id in "ABC"]
    massless = propagate(Graph.from_edges(edges), {"A": 1.0}, PropagationConfig())
    massed = propagate(
        Graph.from_edges(edges, nodes=nodes),
        {"A": 1.0},
        PropagationConfig(mass=MassConfig(enabled=True)),
    )
    assert {n: a.energy for n, a in massed.activations.items()} == {
        n: a.energy for n, a in massless.activations.items()
    }, "equal-length atoms must behave as if mass did not exist"


def test_a_heavy_node_demands_more_evidence_to_light_up() -> None:
    # Only B (and two inactive fillers) carry node data - unlisted nodes
    # default to mass 1.0, so A forwards its plain 6.0. Lengths 600/100/100
    # give a layer mean of ~266.7 and B's mass caps at 2.0: the gate at
    # threshold_ratio .45 is 4.5 * 2.0 = 9.0 > 6.0 and heavy-B dies, while
    # the massless graph keeps it.
    edges = [("A", "B", 0.5)]
    heavy = Graph.from_edges(
        edges,
        nodes=[make_node("B", 600), make_node("X", 100), make_node("Y", 100)],
    )
    config = PropagationConfig(threshold_ratio=0.45, mass=MassConfig(enabled=True))
    result = propagate(heavy, {"A": 1.0}, config)
    assert "B" not in result.activations, (
        "a heavy atom activates late: 6.0 < gate 4.5 * mass 2.0"
    )
    light = Graph.from_edges(edges)
    assert "B" in propagate(light, {"A": 1.0}, config).activations


def test_a_heavy_node_carries_further_once_lit() -> None:
    # Same mass-2.0 construction for B; A and C stay at the default 1.0.
    # B receives 6.0 (gate 1.5 * 2.0 = 3.0, alive) and forwards
    # damping ** (1/2) ~ .7746 instead of .6: C receives ~4.65, not 3.6.
    graph = Graph.from_edges(
        [("A", "B", 0.5), ("B", "C", 0.5)],
        nodes=[make_node("B", 600), make_node("X", 100), make_node("Y", 100)],
    )
    result = propagate(
        graph, {"A": 1.0}, PropagationConfig(mass=MassConfig(enabled=True))
    )
    assert result.energy_of("C") == pytest.approx(6.0 * 0.6**0.5), (
        "heavy atoms forward damping ** (1 / mass) - the ball rolls further"
    )


def test_seeds_are_never_mass_gated_at_injection() -> None:
    # A heavy seed holds its full injected energy; the gate is for arrivals.
    graph = Graph.from_edges(
        [("A", "B", 0.5)], nodes=[make_node("A", 500), make_node("B", 50)]
    )
    result = propagate(
        graph, {"A": 1.0}, PropagationConfig(mass=MassConfig(enabled=True))
    )
    assert result.energy_of("A") == pytest.approx(10.0)


def test_config_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="exponent"):
        MassConfig(exponent=-0.1)
    with pytest.raises(ValueError, match="floor"):
        MassConfig(floor=0.0)
    with pytest.raises(ValueError, match="cap"):
        MassConfig(cap=0.9)
