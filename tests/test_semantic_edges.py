"""Semantic edge builder: cosine, union kNN, non-negative floor, determinism."""

from __future__ import annotations

import pytest

from spiyweb import Graph, SemanticEdgeConfig
from spiyweb.edges import build_semantic_edges


def test_config_k_below_one_raises_value_error() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        SemanticEdgeConfig(k=0)


@pytest.mark.parametrize("floor", [-0.1, 1.0])
def test_config_min_similarity_outside_unit_interval_raises(floor: float) -> None:
    with pytest.raises(ValueError, match="min_similarity"):
        SemanticEdgeConfig(min_similarity=floor)


def test_hand_computable_vectors_yield_exact_cosine() -> None:
    edges = build_semantic_edges(["a", "b"], [[1.0, 0.0], [1.0, 1.0]])
    assert edges == [("a", "b", pytest.approx(2**-0.5))]


def test_negative_cosine_is_filtered_and_the_graph_accepts_the_rest() -> None:
    edges = build_semantic_edges(
        ["a", "b", "c"],
        [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.1]],
    )
    pairs = {(u, v) for u, v, _ in edges}
    assert ("a", "b") not in pairs, "anti-parallel vectors must not survive"
    graph = Graph.from_layers({"semantic": edges})
    assert "a" in graph.nodes, "the floor is what protects the graph invariant"


def test_orthogonal_vectors_are_not_emitted_at_the_default_floor() -> None:
    edges = build_semantic_edges(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    assert edges == [], (
        "strict > at floor 0.0: an exact-zero cosine is not a suppressed edge"
    )


def test_union_rule_keeps_one_way_top_k_membership() -> None:
    # With k=1: "far" ranks "hub" first, but "hub" prefers "twin" (its near
    # duplicate, tilted away from "far"); union kNN must still keep the
    # (far, hub) contact edge.
    edges = build_semantic_edges(
        ["hub", "twin", "far"],
        [[1.0, 0.0], [1.0, -0.05], [0.6, 0.8]],
        config=SemanticEdgeConfig(k=1),
    )
    pairs = {(u, v) for u, v, _ in edges}
    assert ("far", "hub") in pairs, (
        "mutual kNN would prune this legitimate first-contact point"
    )
    assert ("far", "twin") not in pairs, (
        "a pair in nobody's top-k must stay absent - union, not all-pairs"
    )
    assert len(pairs) == len(edges), "each pair is emitted at most once"


def test_top_k_tie_at_the_boundary_breaks_by_id() -> None:
    # All three vectors are colinear, so every similarity is exactly 1.0 and
    # each top-1 slot is decided purely by the id tie-break: "a" picks "b",
    # while "b" and "c" both pick "a" - so (b, c) must stay absent.
    edges = build_semantic_edges(
        ["a", "b", "c"],
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        config=SemanticEdgeConfig(k=1),
    )
    pairs = {(u, v) for u, v, _ in edges}
    assert pairs == {("a", "b"), ("a", "c")}, (
        "boundary ties must resolve by id, identically on every platform"
    )


def test_k_larger_than_the_corpus_caps_naturally() -> None:
    edges = build_semantic_edges(
        ["a", "b"],
        [[1.0, 0.0], [1.0, 1.0]],
        config=SemanticEdgeConfig(k=50),
    )
    assert len(edges) == 1


def test_min_similarity_floor_filters_weak_pairs() -> None:
    edges = build_semantic_edges(
        ["a", "b"],
        [[1.0, 0.0], [1.0, 1.0]],
        config=SemanticEdgeConfig(min_similarity=0.9),
    )
    assert edges == [], "cosine ~0.707 must not pass a 0.9 floor"


def test_zero_norm_embedding_raises_with_the_node_id() -> None:
    with pytest.raises(ValueError, match=r"'b'.*zero norm"):
        build_semantic_edges(["a", "b"], [[1.0, 0.0], [0.0, 0.0]])


def test_length_mismatch_between_ids_and_embeddings_raises() -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        build_semantic_edges(["a", "b"], [[1.0, 0.0]])


def test_inconsistent_embedding_dimension_raises_with_the_node_id() -> None:
    with pytest.raises(ValueError, match=r"'b'.*dimension"):
        build_semantic_edges(["a", "b"], [[1.0, 0.0], [1.0]])


def test_duplicate_node_id_raises_value_error() -> None:
    with pytest.raises(ValueError, match="duplicate node id"):
        build_semantic_edges(["a", "a"], [[1.0, 0.0], [0.0, 1.0]])


def test_empty_and_single_node_inputs_yield_no_edges() -> None:
    assert build_semantic_edges([], []) == []
    assert build_semantic_edges(["a"], [[1.0, 0.0]]) == []


def test_output_is_canonically_ordered_and_sorted() -> None:
    edges = build_semantic_edges(
        ["z", "a", "m"],
        [[1.0, 0.0], [1.0, 0.1], [1.0, 0.2]],
    )
    assert edges == sorted(edges)
    assert all(u < v for u, v, _ in edges), "each pair is emitted as (min, max)"
