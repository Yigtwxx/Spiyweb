"""The FAISS-backed semantic builder is a twin, not a fork.

The claim under test: `build_semantic_edges_fast` emits exactly the edges the
pure-stdlib builder emits - same union-kNN pair set, same weights up to float
noise - on any input where similarities are distinct. The pure builder stays
the semantics oracle; the fast one only makes corpus scale feasible. If these
two ever disagree beyond float noise, the fast path has drifted and the graph
built for the measurement no longer matches the documented semantics.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("numpy")
pytest.importorskip("faiss")

from spiyweb.config import SemanticEdgeConfig
from spiyweb.edges import build_semantic_edges
from spiyweb.store import build_semantic_edges_fast


def make_corpus(
    count: int, dimension: int, seed: int = 7
) -> tuple[list[str], list[list[float]]]:
    """Seeded random vectors: distinct similarities with probability one."""
    rng = random.Random(seed)
    ids = [f"n{i:03d}" for i in range(count)]
    vectors = [[rng.uniform(-1.0, 1.0) for _ in range(dimension)] for _ in range(count)]
    return ids, vectors


def assert_same_edges(
    fast: list[tuple[str, str, float]], pure: list[tuple[str, str, float]]
) -> None:
    assert [(u, v) for u, v, _ in fast] == [(u, v) for u, v, _ in pure], (
        "the fast builder must select exactly the pure builder's union-kNN "
        "pair set - a drifted selection silently changes the graph"
    )
    for (_, _, fast_weight), (_, _, pure_weight) in zip(fast, pure, strict=True):
        assert fast_weight == pytest.approx(pure_weight), (
            "edge weights are the same cosine, differing only by float noise"
        )


def test_fast_builder_matches_the_pure_oracle_on_random_vectors() -> None:
    ids, vectors = make_corpus(40, 8)
    assert_same_edges(
        build_semantic_edges_fast(ids, vectors),
        build_semantic_edges(ids, vectors),
    )


def test_unnormalised_input_is_normalised_internally_like_the_oracle() -> None:
    ids, vectors = make_corpus(25, 6, seed=11)
    rng = random.Random(12)
    scaled = [
        [component * scale for component in vector]
        for vector, scale in zip(
            vectors, [rng.uniform(0.1, 9.0) for _ in vectors], strict=True
        )
    ]
    assert_same_edges(
        build_semantic_edges_fast(ids, scaled),
        build_semantic_edges(ids, scaled),
    )
    # Cosine is scale-invariant, so scaling must not change the pair set.
    unscaled_pairs = [(u, v) for u, v, _ in build_semantic_edges_fast(ids, vectors)]
    scaled_pairs = [(u, v) for u, v, _ in build_semantic_edges_fast(ids, scaled)]
    assert scaled_pairs == unscaled_pairs


def test_k_larger_than_the_corpus_caps_naturally() -> None:
    ids, vectors = make_corpus(4, 3, seed=3)
    config = SemanticEdgeConfig(k=99)
    assert_same_edges(
        build_semantic_edges_fast(ids, vectors, config),
        build_semantic_edges(ids, vectors, config),
    )


def test_exactly_zero_similarity_never_leaks_in() -> None:
    # Orthogonal pairs have cosine exactly 0.0; the strict floor drops them in
    # both builders - 0.0 stays reserved for dedup-suppressed edges.
    ids = ["a", "b", "c"]
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    fast = build_semantic_edges_fast(ids, vectors)
    assert all(weight > 0.0 for _, _, weight in fast)
    assert {(u, v) for u, v, _ in fast} == {("a", "c"), ("b", "c")}, (
        "a-b is orthogonal and must be absent, not present at weight 0.0"
    )


def test_output_is_canonically_ordered() -> None:
    ids, vectors = make_corpus(15, 4, seed=5)
    fast = build_semantic_edges_fast(ids, vectors)
    assert fast == sorted(fast)
    assert all(u < v for u, v, _ in fast)


def test_fewer_than_two_vectors_yield_no_edges() -> None:
    assert build_semantic_edges_fast([], []) == []
    assert build_semantic_edges_fast(["a"], [[1.0, 0.0]]) == []


def test_id_and_embedding_counts_must_pair_up() -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        build_semantic_edges_fast(["a", "b"], [[1.0, 0.0]])


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate node id"):
        build_semantic_edges_fast(["a", "a"], [[1.0, 0.0], [0.0, 1.0]])


def test_mismatched_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="dimension"):
        build_semantic_edges_fast(["a", "b"], [[1.0, 0.0], [1.0]])


def test_zero_norm_embedding_raises_instead_of_being_skipped() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        build_semantic_edges_fast(["a", "b"], [[0.0, 0.0], [1.0, 0.0]])
