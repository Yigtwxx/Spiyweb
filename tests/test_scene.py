"""Scene assembly: what the picture contains, and what it may never drop."""

from __future__ import annotations

from spiyweb.config import LayerWeights
from spiyweb.core.graph import Graph, Node
from spiyweb.core.propagate import Activation
from spiyweb.scene import (
    LayoutConfig,
    ViewConfig,
    build_layer_index,
    build_scene,
    contributor_edges,
    induced_edges,
    select_subgraph,
)

_FAST = LayoutConfig(iterations=5)


def _graph() -> Graph:
    nodes = [
        Node(id="a", layer="chunk", source_id="d0", length=100),
        Node(id="b", layer="chunk", source_id="d1", length=100),
        Node(id="c", layer="chunk", source_id="d2", length=100),
        Node(id="a_dup", layer="chunk", source_id="d3", length=100),
    ]
    edges = [("a", "b", 0.8), ("b", "c", 0.4), ("a", "c", 0.2), ("a", "a_dup", 0.0)]
    return Graph.from_edges(edges, nodes=nodes)


def _activations() -> dict[str, Activation]:
    return {
        "a": Activation(energy=5.0, hop=0, contributors=()),
        "b": Activation(energy=3.0, hop=1, contributors=("a",)),
        "c": Activation(energy=1.5, hop=2, contributors=("a", "b")),
    }


def _index() -> object:
    layers = {
        "semantic": [("a", "b", 0.9), ("a", "a_dup", 0.99)],
        "entity": [("b", "c", 0.5), ("a", "c", 0.3)],
    }
    return build_layer_index(["a", "b", "c", "a_dup"], layers)


def _scene(**overrides: object) -> object:
    defaults: dict[str, object] = {
        "activations": _activations(),
        "seeds": {"a": 0.9},
        "votes": {},
        "suppressed": {},
        "graph": _graph(),
        "layer_index": _index(),
        "weights": LayerWeights(),
        "layout": _FAST,
    }
    defaults.update(overrides)
    return build_scene(**defaults)  # type: ignore[arg-type]


def test_select_subgraph_keeps_the_strongest_and_counts_the_rest() -> None:
    energies = {"a": 5.0, "b": 3.0, "c": 1.5}
    kept, dropped = select_subgraph(energies, energies, limit=2)
    assert set(kept) == {"a", "b"}
    assert dropped == 1


def test_select_subgraph_breaks_ties_on_id() -> None:
    energies = {"z": 1.0, "y": 1.0}
    kept, _ = select_subgraph(energies, energies, limit=1)
    assert kept == ("y",)


def test_contributor_edges_are_the_causal_skeleton() -> None:
    edges = contributor_edges(_activations(), {"a", "b", "c"})
    assert edges == [("a", "b"), ("a", "c"), ("b", "c")]


def test_contributor_edges_ignore_undrawn_endpoints() -> None:
    assert contributor_edges(_activations(), {"b", "c"}) == [("b", "c")]


def test_induced_edges_skip_suppressed_zero_weight_links() -> None:
    edges, dropped = induced_edges(_graph(), {"a", "b", "c", "a_dup"}, limit=10)
    assert ("a", "a_dup", 0.0) not in edges
    assert dropped == 0


def test_induced_edges_cap_reports_what_it_dropped() -> None:
    edges, dropped = induced_edges(_graph(), {"a", "b", "c"}, limit=1)
    assert len(edges) == 1
    assert dropped == 2


def test_seed_and_bridge_kinds_are_marked() -> None:
    scene = _scene(bridges=("c",))
    kinds = {node.id: node.kind for node in scene.nodes}
    assert kinds["a"] == "seed"
    assert kinds["c"] == "bridge"
    assert kinds["b"] == "activated"


def test_suppressed_pair_becomes_a_dashed_edge_with_a_ghost_node() -> None:
    """The scope requires the cut link; a dashed line needs both endpoints."""
    scene = _scene(suppressed={"a_dup": "a"})
    ghosts = [node for node in scene.nodes if node.kind == "suppressed"]
    assert [node.id for node in ghosts] == ["a_dup"]
    dashed = [edge for edge in scene.edges if edge.kind == "suppressed"]
    assert len(dashed) == 1
    assert (dashed[0].source, dashed[0].target) == ("a_dup", "a")


def test_dashed_edges_survive_the_edge_cap() -> None:
    """Dedup links are a required element - the cap may never trim them."""
    scene = _scene(
        suppressed={"a_dup": "a"},
        view=ViewConfig(max_edges=0, edge_mode="induced"),
    )
    assert any(edge.kind == "suppressed" for edge in scene.edges)


def test_every_edge_endpoint_is_a_drawn_node() -> None:
    scene = _scene(suppressed={"a_dup": "a"}, view=ViewConfig(edge_mode="induced"))
    drawn = {node.id for node in scene.nodes}
    for edge in scene.edges:
        assert edge.source in drawn
        assert edge.target in drawn


def test_node_cap_reports_the_dropped_count() -> None:
    scene = _scene(view=ViewConfig(max_nodes=2))
    assert scene.dropped_nodes == 1
    assert len(scene.nodes) == 2


def test_edges_carry_their_dominant_layer() -> None:
    scene = _scene(view=ViewConfig(edge_mode="induced"))
    by_pair = {(edge.source, edge.target): edge for edge in scene.edges}
    assert by_pair[("a", "b")].layer == "semantic"
    assert by_pair[("b", "c")].layer == "entity"


def test_votes_fall_back_to_the_source_id() -> None:
    """Votes are counted per document, so the lookup must resolve per source."""
    scene = _scene(votes={"d1": 3})
    votes = {node.id: node.votes for node in scene.nodes}
    assert votes["b"] == 3
    assert votes["a"] == 1


def test_labels_are_limited_to_the_strongest_nodes() -> None:
    scene = _scene(view=ViewConfig(label_top_n=1))
    labelled = [node.id for node in scene.nodes if node.label]
    assert labelled == ["a"]


def test_scene_is_deterministic() -> None:
    assert _scene() == _scene()


def test_caption_states_what_was_left_out() -> None:
    scene = _scene(view=ViewConfig(max_nodes=2))
    assert "drawing 2 of 3 activated atoms" in scene.caption
    assert "dedup cut" in scene.caption


def _spec(**overrides: object) -> dict:
    from spiyweb.scene import build_vega_spec

    return build_vega_spec(_scene(**overrides))  # type: ignore[arg-type]


def test_spec_separates_active_and_dashed_edge_layers() -> None:
    """Two rule layers, not one dash encoding - readable and testable."""
    layers = _spec(suppressed={"a_dup": "a"}, view=ViewConfig(edge_mode="induced"))[
        "layer"
    ]
    rules = [layer for layer in layers if layer["mark"]["type"] == "rule"]
    assert len(rules) == 2
    assert "strokeDash" not in rules[0]["mark"]
    assert rules[1]["mark"]["strokeDash"] == [4, 3]
    assert len(rules[1]["data"]["values"]) == 1


def test_spec_colour_domain_covers_every_layer() -> None:
    """A legend that reshuffles per query cannot be read."""
    from spiyweb.scene import EDGE_LAYER_ORDER

    rule = _spec()["layer"][0]
    scale = rule["encoding"]["color"]["scale"]
    assert scale["domain"] == list(EDGE_LAYER_ORDER)
    assert len(scale["range"]) == len(EDGE_LAYER_ORDER)


def test_spec_pins_the_unit_square() -> None:
    nodes = _spec()["layer"][2]
    assert nodes["encoding"]["x"]["scale"]["domain"] == [0, 1]
    assert nodes["encoding"]["y"]["scale"]["domain"] == [0, 1]


def test_spec_carries_every_node_and_only_labelled_text() -> None:
    spec = _spec(view=ViewConfig(label_top_n=1))
    assert len(spec["layer"][2]["data"]["values"]) == 3
    assert len(spec["layer"][3]["data"]["values"]) == 1


def test_spec_is_deterministic() -> None:
    assert _spec() == _spec()


def test_zoom_param_is_declared_on_exactly_one_layer() -> None:
    """A layered param duplicates the `x1` signal and kills the whole chart."""
    spec = _spec(suppressed={"a_dup": "a"})
    assert "params" not in spec
    carriers = [layer for layer in spec["layer"] if "params" in layer]
    assert len(carriers) == 1
    assert carriers[0]["encoding"]["x"]["field"] == "x"


def test_edge_and_node_colour_scales_stay_independent() -> None:
    """Merged colour scales hand the atoms the edge palette - a silent lie."""
    spec = _spec()
    assert spec["resolve"]["scale"]["color"] == "independent"


def test_only_one_edge_layer_legend_is_printed() -> None:
    """Independent scales give each layer a legend; the duplicate is muted."""
    layers = _spec(suppressed={"a_dup": "a"})["layer"]
    rules = [layer for layer in layers if layer["mark"]["type"] == "rule"]
    assert rules[0]["encoding"]["color"]["legend"] == {"title": "edge layer"}
    assert rules[1]["encoding"]["color"]["legend"] is None
    assert (
        rules[1]["encoding"]["color"]["scale"] == rules[0]["encoding"]["color"]["scale"]
    )


def test_the_dashed_cut_is_drawn_at_a_visible_length() -> None:
    """A ghost glued onto its survivor hides the link the scope requires."""
    scene = _scene(suppressed={"a_dup": "a"}, layout=LayoutConfig(iterations=200))
    dashed = next(edge for edge in scene.edges if edge.kind == "suppressed")
    length = ((dashed.x2 - dashed.x1) ** 2 + (dashed.y2 - dashed.y1) ** 2) ** 0.5
    assert length > 0.05, f"dashed cut is only {length:.4f} long"
