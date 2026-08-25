"""Hop rings: the decay made spatial, under the same determinism contract."""

from __future__ import annotations

import math

from spiyweb.scene import hop_ring_layout

_HOPS = {"a": 0, "b": 1, "c": 1, "d": 2, "e": 2, "f": 2}
_ENERGY = {"a": 5.0, "b": 3.0, "c": 2.0, "d": 1.5, "e": 1.2, "f": 1.0}
_NODES = list(_HOPS)


def test_layout_is_deterministic() -> None:
    assert hop_ring_layout(_NODES, _HOPS, _ENERGY) == hop_ring_layout(
        _NODES, _HOPS, _ENERGY
    )


def test_input_order_does_not_change_the_picture() -> None:
    assert hop_ring_layout(_NODES, _HOPS, _ENERGY) == hop_ring_layout(
        list(reversed(_NODES)), _HOPS, _ENERGY
    )


def test_a_lone_seed_sits_at_the_centre() -> None:
    placed = hop_ring_layout(_NODES, _HOPS, _ENERGY)
    assert placed["a"] == (0.5, 0.5)


def test_radius_grows_with_hop() -> None:
    placed = hop_ring_layout(_NODES, _HOPS, _ENERGY)

    def radius(node: str) -> float:
        x, y = placed[node]
        return math.hypot(x - 0.5, y - 0.5)

    assert radius("b") < radius("d")
    assert (
        radius("b") == round(radius("c"), 12) or abs(radius("b") - radius("c")) < 1e-9
    )


def test_every_point_stays_inside_the_unit_square() -> None:
    for x, y in hop_ring_layout(_NODES, _HOPS, _ENERGY).values():
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_different_query_rotates_the_rings() -> None:
    assert hop_ring_layout(_NODES, _HOPS, _ENERGY, salt="q1") != hop_ring_layout(
        _NODES, _HOPS, _ENERGY, salt="q2"
    )


def test_degenerate_inputs_are_not_errors() -> None:
    assert hop_ring_layout([], {}, {}) == {}
    assert hop_ring_layout(["only"], {"only": 0}, {"only": 1.0}) == {"only": (0.5, 0.5)}


def test_unknown_hops_default_to_the_centre_ring() -> None:
    placed = hop_ring_layout(["x", "y"], {}, {})
    assert len(placed) == 2
    assert all(math.isfinite(value) for point in placed.values() for value in point)
