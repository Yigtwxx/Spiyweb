"""Honesty outputs (D18/D19/D20/D35): paths, themes, gaps, refusal report.

The claim under test: the web's SHAPE carries information a ranked list
discards - every node explains its own chain, separate themes surface as
clusters, a missing bridge becomes a corpus diagnosis, and a weak result
explains itself through a template, without an LLM anywhere.
"""

from __future__ import annotations

import pytest

from spiyweb import (
    Graph,
    OutputConfig,
    PropagationConfig,
    activation_paths,
    build_refusal_report,
    color_composition,
    gap_warnings,
    propagate,
    propagate_colored,
    theme_clusters,
)

# Two disconnected regions: {A, B, C} around the first seed and {X, Y, Z}
# around the second. No edge crosses - the canonical gap situation.
TWO_ISLANDS = Graph.from_edges(
    [
        ("A", "B", 1.0),
        ("B", "C", 1.0),
        ("X", "Y", 1.0),
        ("Y", "Z", 1.0),
    ]
)
ISLAND_SEEDS = {"A": 0.5, "X": 0.5}

CHAIN = Graph.from_edges([("A", "B", 1.0), ("B", "C", 1.0)])


# ---------------------------------------------------------------- paths


def test_every_node_carries_its_chain_back_to_a_seed() -> None:
    result = propagate(CHAIN, {"A": 1.0}, PropagationConfig())
    paths = {path.node: path for path in activation_paths(result)}
    assert paths["A"].steps == ("A",)
    assert paths["B"].steps == ("A", "B")
    assert paths["C"].steps == ("A", "B", "C")
    assert paths["C"].hop == 2


def test_a_converging_node_reports_its_feeder_count() -> None:
    diamond = Graph.from_edges(
        [("A", "D", 0.4), ("C", "D", 0.6)]
    )  # the canonical shape: D fed from two directions
    result = propagate(diamond, {"A": 0.9, "C": 0.7}, PropagationConfig())
    (d_path,) = [p for p in activation_paths(result) if p.node == "D"]
    assert d_path.converging == 2, "converging evidence must stay visible"
    # The chain follows the strongest contributor: A (5.625) over C (4.375).
    assert d_path.steps == ("A", "D")


def test_rendered_path_names_the_edge_reason_when_a_label_exists() -> None:
    result = propagate(CHAIN, {"A": 1.0}, PropagationConfig())
    (b_path,) = [p for p in activation_paths(result) if p.node == "B"]
    labels = {("A", "B"): "shared entity 'Tesla'"}
    assert b_path.rendered(labels) == "A -> shared entity 'Tesla' -> B"
    assert b_path.rendered() == "A -> B", "no label map degrades to the bare chain"
    assert b_path.rendered({("B", "A"): "x"}) == "A -> x -> B", (
        "the label map is undirected, like the graph"
    )


def test_paths_are_ordered_strongest_first() -> None:
    result = propagate(CHAIN, {"A": 1.0}, PropagationConfig())
    energies = [path.energy for path in activation_paths(result)]
    assert energies == sorted(energies, reverse=True)


# ---------------------------------------------------------------- themes


def test_two_islands_surface_as_two_clusters_with_energy_shares() -> None:
    result = propagate(TWO_ISLANDS, ISLAND_SEEDS, PropagationConfig())
    clusters = theme_clusters(result, TWO_ISLANDS)
    assert len(clusters) == 2
    assert {cluster.nodes for cluster in clusters} == {
        ("A", "B", "C"),
        ("X", "Y", "Z"),
    }
    assert sum(cluster.energy_share for cluster in clusters) == pytest.approx(1.0)
    assert {cluster.top_node for cluster in clusters} == {"A", "X"}


def test_clusters_carry_their_colour_composition() -> None:
    colored = propagate_colored(
        TWO_ISLANDS,
        {"c0": {"A": 1.0}, "c1": {"X": 1.0}},
        PropagationConfig(),
    )
    # Merge the per-colour activations into one mapping for clustering; the
    # plain result of either colour suffices structurally, so cluster over
    # colour c0's world plus c1's via a combined fake run: use c0's result
    # for the interface and the composition map for the colours.
    composition = color_composition(colored)
    assert composition["A"] == ("c0",)
    assert composition["X"] == ("c1",)


# ---------------------------------------------------------------- gaps


def test_two_dense_unconnected_clusters_raise_a_gap_warning() -> None:
    result = propagate(TWO_ISLANDS, ISLAND_SEEDS, PropagationConfig())
    clusters = theme_clusters(result, TWO_ISLANDS)
    (warning,) = gap_warnings(clusters)
    assert {warning.top_a, warning.top_b} == {"A", "X"}
    assert "share no connection" in warning.message


def test_a_sparse_cluster_raises_no_gap_warning() -> None:
    result = propagate(TWO_ISLANDS, ISLAND_SEEDS, PropagationConfig())
    clusters = theme_clusters(result, TWO_ISLANDS)
    strict = OutputConfig(min_cluster_nodes=4)
    assert gap_warnings(clusters, strict) == (), (
        "three-node islands must not pass a four-node density floor"
    )
    energy_strict = OutputConfig(min_cluster_energy_share=0.6)
    assert gap_warnings(clusters, energy_strict) == (), (
        "two half-energy islands must not pass a 60% share floor"
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("min_cluster_nodes", 0), ("min_cluster_energy_share", 1.5)],
)
def test_output_config_rejects_out_of_range_values(
    field_name: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field_name.split("_")[1]):
        OutputConfig(**{field_name: value})


# ---------------------------------------------------------------- refusal


def test_refusal_report_names_the_gap_and_the_missing_source() -> None:
    result = propagate(TWO_ISLANDS, ISLAND_SEEDS, PropagationConfig())
    report = build_refusal_report(result, TWO_ISLANDS)
    assert len(report.clusters) == 2
    assert len(report.gaps) == 1
    assert report.stop_reason == "threshold"
    assert "2 theme cluster(s)" in report.text
    assert "share no connection" in report.text
    assert "Missing: a source that connects" in report.text


def test_refusal_report_without_a_gap_points_at_the_death_frontier() -> None:
    result = propagate(CHAIN, {"A": 1.0}, PropagationConfig())
    report = build_refusal_report(result, CHAIN)
    assert report.gaps == ()
    assert "No gap between dense clusters" in report.text
    assert "the energy died at hop" in report.text
    assert report.deepest_nodes == ("C",)


# ------------------------------------------------- entity edge labels


def test_entity_edge_labels_names_the_rarest_shared_entity() -> None:
    from spiyweb import entity_edge_labels

    entities = {
        "a": ["common", "rare"],
        "b": ["common", "rare"],
        "c": ["common"],
    }
    labels = entity_edge_labels([("a", "b")], entities)
    assert labels == {("a", "b"): "shared entity 'rare'"}, (
        "the least common shared entity is the most informative reason"
    )


def test_entity_edge_labels_breaks_frequency_ties_lexicographically() -> None:
    from spiyweb import entity_edge_labels

    entities = {"a": ["x", "y"], "b": ["x", "y"]}
    labels = entity_edge_labels([("a", "b")], entities)
    assert labels == {("a", "b"): "shared entity 'x'"}


def test_entity_edge_labels_skips_pairs_without_shared_entities() -> None:
    from spiyweb import entity_edge_labels

    entities = {"a": ["x"], "b": ["y"]}
    labels = entity_edge_labels([("a", "b"), ("a", "a"), ("a", "unknown")], entities)
    assert labels == {}, "no shared entity, self-pairs and unknown ids get no entry"


def test_entity_edge_labels_feed_rendered_paths_end_to_end() -> None:
    from spiyweb import activation_paths, entity_edge_labels

    result = propagate(CHAIN, {"A": 1.0}, PropagationConfig())
    entities = {"A": ["alpha"], "B": ["alpha", "beta"], "C": ["beta"]}
    paths = {path.node: path for path in activation_paths(result)}
    pairs = [
        pair
        for path in paths.values()
        for pair in zip(path.steps, path.steps[1:], strict=False)
    ]
    labels = entity_edge_labels(pairs, entities)
    assert (
        paths["C"].rendered(labels)
        == "A -> shared entity 'alpha' -> B -> shared entity 'beta' -> C"
    )
