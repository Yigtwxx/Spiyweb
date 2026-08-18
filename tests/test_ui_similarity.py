"""The similarity backend: the half of dedup the harness never supplies."""

from __future__ import annotations

import math

import numpy as np
import pytest

from graph_view import make_similarity, vector_matrix
from spiyweb.config import DedupConfig
from spiyweb.core.dedup import adaptive_threshold, find_survivor

_ROWS = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]


def _similarity() -> object:
    return make_similarity(vector_matrix(["a", "b", "c"], _ROWS))


def test_cosine_matches_the_hand_computation() -> None:
    scores = _similarity()("a", ["a", "b", "c"])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(1 / math.sqrt(2))


def test_batch_shape_is_preserved() -> None:
    assert len(_similarity()("a", ["a", "b", "c"])) == 3
    assert _similarity()("a", []) == []


def test_unknown_ids_score_zero_instead_of_raising() -> None:
    scores = _similarity()("a", ["ghost", "b"])
    assert scores == pytest.approx([0.0, 0.0])
    assert _similarity()("ghost", ["a"]) == pytest.approx([0.0])


def test_unnormalised_rows_are_rescued() -> None:
    """A metric-agnostic artifact must not turn cosine into a dot product."""
    scaled = make_similarity(vector_matrix(["a", "b"], [[3.0, 0.0], [0.0, 5.0]]))
    assert scaled("a", ["a", "b"]) == pytest.approx([1.0, 0.0])


def test_zero_vector_does_not_produce_nan() -> None:
    zeroed = make_similarity(vector_matrix(["a", "b"], [[0.0, 0.0], [1.0, 0.0]]))
    assert all(math.isfinite(value) for value in zeroed("a", ["a", "b"]))


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        vector_matrix(["a", "b"], [[1.0, 0.0]])


def test_it_satisfies_the_core_dedup_protocol() -> None:
    """The whole point: `core.dedup` must accept what the inspector builds."""
    ids = [f"n{index}" for index in range(12)]
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(12, 8))
    rows[1] = rows[0]  # an exact duplicate pair
    similarity = make_similarity(vector_matrix(ids, rows))
    config = DedupConfig()
    tau = adaptive_threshold(ids, similarity, config)
    assert 0.0 < tau <= 1.0
    assert find_survivor("n1", ["n0"], similarity, tau) == "n0"
    assert find_survivor("n2", ["n0"], similarity, tau) is None


def test_below_min_pairs_the_threshold_falls_back_to_the_floor() -> None:
    similarity = _similarity()
    config = DedupConfig(min_pairs=100)
    assert adaptive_threshold(["a", "b", "c"], similarity, config) == config.floor
