"""The edge-layer memory the merged adjacency deliberately throws away."""

from __future__ import annotations

import pytest

from spiyweb.config import LayerWeights
from spiyweb.scene import build_layer_index

_LAYERS: dict[str, list[tuple[str, str, float]]] = {
    "semantic": [("a", "b", 0.9), ("b", "c", 0.4)],
    "entity": [("b", "a", 0.5), ("c", "d", 1.0)],
    "structural": [],
    "derivation": [("a", "a#p0", 1.0)],
}
_NODES = ["a", "b", "c", "d", "a#p0"]


def _index() -> object:
    return build_layer_index(_NODES, _LAYERS)


def test_single_layer_edge_reports_that_layer() -> None:
    primary, layers = _index().layer_of("c", "d", LayerWeights())
    assert primary == "entity"
    assert layers == ("entity",)


def test_multi_layer_edge_lists_every_layer_and_picks_the_dominant_one() -> None:
    """`a-b` lives in two layers; the winner is the largest weighted share."""
    index = _index()
    # semantic .5 * .9 = .45 against entity 1.0 * .5 = .50 -> entity leads.
    primary, layers = index.layer_of("a", "b", LayerWeights())
    assert layers == ("semantic", "entity")
    assert primary == "entity"


def test_raising_a_layer_weight_recolours_the_edge() -> None:
    """The colour tracks which layer actually drags the edge into the graph."""
    index = _index()
    heavy_semantic = LayerWeights(semantic=2.0, entity=1.0)
    primary, _ = index.layer_of("a", "b", heavy_semantic)
    assert primary == "semantic"


def test_zero_weighted_layer_is_not_counted() -> None:
    """`Graph.from_layers` drops a 0.0 layer entirely; so does the colouring."""
    index = _index()
    primary, layers = index.layer_of("a", "b", LayerWeights(entity=0.0))
    assert primary == "semantic"
    assert layers == ("semantic",)


def test_edge_only_in_a_disabled_layer_disappears() -> None:
    index = _index()
    assert index.layer_of("c", "d", LayerWeights(entity=0.0)) == (None, ())


def test_lookup_is_undirected() -> None:
    index = _index()
    assert index.layer_of("a", "b", LayerWeights()) == index.layer_of(
        "b", "a", LayerWeights()
    )
    assert index.raw_weight("entity", "c", "d") == index.raw_weight("entity", "d", "c")


def test_unknown_pair_is_not_an_error() -> None:
    index = _index()
    assert index.layer_of("a", "nowhere", LayerWeights()) == (None, ())
    assert index.layer_of("a", "c", LayerWeights()) == (None, ())
    assert index.raw_weight("entity", "a", "nowhere") is None


def test_self_pair_is_never_an_edge() -> None:
    assert _index().raw_weight("semantic", "a", "a") is None


def test_contributions_are_the_weighted_shares() -> None:
    shares = _index().contributions("a", "b", LayerWeights())
    assert shares == pytest.approx({"semantic": 0.45, "entity": 0.5})


def test_counts_expose_empty_layers() -> None:
    """An empty layer must be visible so the UI can grey out its slider."""
    counts = _index().counts
    assert counts["semantic"] == 2
    assert counts["entity"] == 2
    assert counts["structural"] == 0
    assert counts["derivation"] == 1
    assert counts["learned"] == 0


def test_edges_naming_unknown_nodes_are_dropped() -> None:
    index = build_layer_index(["a", "b"], {"entity": [("a", "ghost", 1.0)]})
    assert index.counts["entity"] == 0


def test_tie_breaks_on_the_fixed_layer_order() -> None:
    """Equal contributions resolve by `EDGE_LAYER_ORDER`, never dict order."""
    layers = {"entity": [("x", "y", 1.0)], "semantic": [("x", "y", 2.0)]}
    index = build_layer_index(["x", "y"], layers)
    primary, held = index.layer_of("x", "y", LayerWeights(semantic=0.5, entity=1.0))
    assert primary == "semantic"
    assert held == ("semantic", "entity")


def test_suppressed_zero_weight_edge_still_belongs_to_its_layer() -> None:
    index = build_layer_index(["x", "y"], {"entity": [("x", "y", 0.0)]})
    primary, held = index.layer_of("x", "y", LayerWeights())
    assert primary == "entity"
    assert held == ("entity",)
