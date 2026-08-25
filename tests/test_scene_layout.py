"""The layout contract: same query, same picture - on any machine, any run."""

from __future__ import annotations

import math

import pytest

from spiyweb.scene import (
    LayoutConfig,
    _shelf_boxes,
    layout_seed,
    spring_layout,
)

_EDGES: list[tuple[str, str, float]] = [
    ("a", "b", 1.0),
    ("b", "c", 0.5),
    ("c", "d", 0.8),
    ("d", "a", 0.2),
]
_NODES = ["a", "b", "c", "d"]


def test_layout_is_bit_identical_across_calls() -> None:
    first = spring_layout(_NODES, _EDGES)
    second = spring_layout(_NODES, _EDGES)
    assert first == second


def test_layout_ignores_input_order() -> None:
    """The picture depends on the node SET, never on how it was listed."""
    forward = spring_layout(_NODES, _EDGES)
    backward = spring_layout(list(reversed(_NODES)), list(reversed(_EDGES)))
    assert forward == backward


def test_layout_ignores_edge_endpoint_order() -> None:
    flipped = [(v, u, w) for u, v, w in _EDGES]
    assert spring_layout(_NODES, _EDGES) == spring_layout(_NODES, flipped)


def test_different_seed_moves_the_cloud() -> None:
    default = spring_layout(_NODES, _EDGES)
    shifted = spring_layout(_NODES, _EDGES, LayoutConfig(seed=17))
    assert default != shifted


def test_different_salt_moves_the_cloud() -> None:
    """The query is part of the seed, so two queries do not share a frame."""
    assert spring_layout(_NODES, _EDGES, salt="q1") != spring_layout(
        _NODES, _EDGES, salt="q2"
    )


def test_positions_stay_inside_the_unit_square() -> None:
    for x, y in spring_layout(_NODES, _EDGES).values():
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert math.isfinite(x) and math.isfinite(y)


def test_empty_and_single_node_are_not_errors() -> None:
    assert spring_layout([], []) == {}
    assert spring_layout(["only"], []) == {"only": (0.5, 0.5)}


def test_nodes_without_edges_survive() -> None:
    placed = spring_layout(["a", "b", "c"], [])
    assert set(placed) == {"a", "b", "c"}
    assert all(math.isfinite(value) for point in placed.values() for value in point)


def test_two_disconnected_components_stay_finite() -> None:
    """Gravity is what keeps components from drifting apart without bound."""
    edges = [("a", "b", 1.0), ("c", "d", 1.0)]
    placed = spring_layout(["a", "b", "c", "d"], edges)
    assert len(placed) == 4
    for x, y in placed.values():
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_components_are_packed_into_the_frame_not_flung_to_the_corners() -> None:
    """Repulsion alone parks disconnected pieces in opposite corners.

    A real 36-atom web (one cluster of 32, one of 4) was using about a quarter
    of the canvas, with an empty band down the middle. Packing each component
    into its own box is what fills the frame.
    """
    edges = [("a", "b", 1.0), ("b", "c", 1.0), ("a", "c", 1.0), ("y", "z", 1.0)]
    placed = spring_layout(["a", "b", "c", "y", "z"], edges)
    xs = [x for x, _ in placed.values()]
    ys = [y for _, y in placed.values()]
    assert max(xs) - min(xs) > 0.8, "the packing should span the width"
    assert max(ys) - min(ys) > 0.8, "the packing should span the height"

    big = [placed[node] for node in ("a", "b", "c")]
    small = [placed[node] for node in ("y", "z")]
    # Boxes are disjoint, so the two clouds must not interleave vertically.
    assert max(y for _, y in big) < min(y for _, y in small), (
        "each component keeps its own box - separation stays legible"
    )


def test_a_bigger_component_gets_a_bigger_box() -> None:
    """Room is shared by size, not equally: a stray pair must not crowd out
    the cluster the reader is actually studying."""
    boxes = _shelf_boxes([9, 1])
    assert boxes[0][2] * boxes[0][3] > boxes[1][2] * boxes[1][3]


def test_one_component_is_left_on_the_original_path() -> None:
    """The packing must not perturb any picture that was already correct."""
    assert _shelf_boxes([4]) == [(0.0, 0.0, 1.0, 1.0)]
    connected = spring_layout(_NODES, _EDGES)
    assert set(connected) == set(_NODES)
    for x, y in connected.values():
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    # A connected graph touches every edge of the unit square after `_rescale`.
    assert min(x for x, _ in connected.values()) == pytest.approx(0.0)
    assert max(x for x, _ in connected.values()) == pytest.approx(1.0)


def test_coincident_start_points_do_not_divide_by_zero() -> None:
    initial = {node: (0.5, 0.5) for node in _NODES}
    placed = spring_layout(_NODES, _EDGES, initial=initial)
    assert all(math.isfinite(value) for point in placed.values() for value in point)


def test_edges_to_undrawn_nodes_are_ignored() -> None:
    with_stranger = [*_EDGES, ("a", "elsewhere", 1.0)]
    assert spring_layout(_NODES, with_stranger) == spring_layout(_NODES, _EDGES)


def test_duplicate_ids_collapse() -> None:
    assert spring_layout(["a", "a", "b"], [("a", "b", 1.0)]).keys() == {"a", "b"}


def test_layout_seed_is_stable_and_set_based() -> None:
    assert layout_seed(["b", "a"], "q", 0) == layout_seed(["a", "b"], "q", 0)
    assert layout_seed(["a", "b"], "q", 0) != layout_seed(["a", "c"], "q", 0)
    assert layout_seed(["a"], "q", 0) >= 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"gravity": -0.1},
        {"gravity": 1.5},
        {"initial_temperature": 0.0},
        {"min_distance": 0.0},
    ],
)
def test_layout_config_rejects_impossible_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LayoutConfig(**kwargs)
