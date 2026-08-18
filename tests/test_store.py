"""Vector store: exact IP search, single-file roundtrip, validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("numpy")
pytest.importorskip("faiss")

from spiyweb.store import VectorStore

if TYPE_CHECKING:
    from pathlib import Path


def make_store() -> VectorStore:
    store = VectorStore(dimension=2)
    store.add(["a", "b", "c"], [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    return store


def test_search_returns_exact_inner_products_best_first() -> None:
    results = make_store().search([1.0, 0.0], k=3)
    assert [chunk_id for chunk_id, _ in results] == ["a", "c", "b"]
    scores = dict(results)
    assert scores["a"] == pytest.approx(1.0)
    assert scores["c"] == pytest.approx(0.8)
    assert scores["b"] == pytest.approx(0.0)


def test_save_load_roundtrip_returns_identical_results(tmp_path: Path) -> None:
    store = make_store()
    target = tmp_path / "web.npz"
    store.save(target)
    loaded = VectorStore.load(target)
    assert len(loaded) == len(store)
    assert loaded.search([0.5, 0.5], k=3) == store.search([0.5, 0.5], k=3), (
        "the rebuilt flat index must reproduce the saved one exactly"
    )


def test_save_writes_exactly_the_given_path(tmp_path: Path) -> None:
    target = tmp_path / "no-suffix"
    make_store().save(target)
    assert target.exists(), "numpy's silent .npz suffix appending is bypassed"
    assert VectorStore.load(target).search([1.0, 0.0], k=1)[0][0] == "a"


def test_duplicate_id_within_one_call_raises() -> None:
    store = VectorStore(dimension=2)
    with pytest.raises(ValueError, match="duplicate vector id 'x'"):
        store.add(["x", "x"], [[1.0, 0.0], [0.0, 1.0]])


def test_duplicate_id_across_calls_raises() -> None:
    store = make_store()
    with pytest.raises(ValueError, match="duplicate vector id 'a'"):
        store.add(["a"], [[1.0, 0.0]])


def test_length_mismatch_and_empty_id_raise() -> None:
    store = VectorStore(dimension=2)
    with pytest.raises(ValueError, match="one-to-one"):
        store.add(["a"], [])
    with pytest.raises(ValueError, match="must not be empty"):
        store.add([""], [[1.0, 0.0]])


def test_dimension_mismatch_on_add_and_on_query_raises() -> None:
    store = VectorStore(dimension=2)
    with pytest.raises(ValueError, match=r"expected \(n, 2\)"):
        store.add(["a"], [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="expected 2"):
        make_store().search([1.0, 0.0, 0.0], k=1)


def test_k_below_one_raises_and_k_above_size_caps() -> None:
    store = make_store()
    with pytest.raises(ValueError, match="k must be at least 1"):
        store.search([1.0, 0.0], k=0)
    assert len(store.search([1.0, 0.0], k=50)) == 3


def test_empty_store_returns_no_results() -> None:
    assert VectorStore(dimension=2).search([1.0, 0.0], k=5) == []


def test_dimension_below_one_raises() -> None:
    with pytest.raises(ValueError, match="dimension"):
        VectorStore(dimension=0)


def test_load_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        VectorStore.load(tmp_path / "absent.npz")


def test_empty_store_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "empty.npz"
    VectorStore(dimension=3).save(target)
    loaded = VectorStore.load(target)
    assert len(loaded) == 0
    assert loaded.dimension == 3
