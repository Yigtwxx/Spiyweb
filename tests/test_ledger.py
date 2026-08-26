"""The energy ledger must add up on the canonical trace, and say so when it does not."""

from __future__ import annotations

import pytest

from spiyweb import ledger
from spiyweb.config import (
    ConflictConfig,
    MassConfig,
    PolarityConfig,
    PropagationConfig,
)
from spiyweb.core.conflict import NegativeEdge, conflict_adjacency
from spiyweb.core.graph import Graph, Node
from spiyweb.core.propagate import propagate

# CLAUDE.md §2.6: the documented worked example, the same edges the README
# and `tests/test_propagate.py` pin.
_EDGES = [
    ("A", "A_dup", 0.0),
    ("A", "B", 0.8),
    ("A", "D", 0.4),
    ("C", "D", 0.6),
    ("C", "E", 0.3),
    ("D", "F", 0.5),
]
_SEEDS = {"A": 0.9, "C": 0.7}


def _graph(nodes: list[Node] | None = None) -> Graph:
    return Graph.from_edges(_EDGES, nodes=nodes or [])


def test_canonical_trace_balances() -> None:
    config = PropagationConfig()
    result = propagate(_graph(), _SEEDS, config)
    book = ledger.build_ledger(result, _graph(), config)
    assert book.injected == pytest.approx(10.0)
    assert book.destroyed.total == pytest.approx(0.0)
    assert book.residual == pytest.approx(0.0, abs=book.tolerance)
    assert book.mismatch == pytest.approx(0.0, abs=book.tolerance)
    assert book.balanced is True


def test_the_three_slices_sum_to_the_injection() -> None:
    config = PropagationConfig()
    result = propagate(_graph(), _SEEDS, config)
    book = ledger.build_ledger(result, _graph(), config)
    total = book.held + book.dissipated + book.destroyed.total
    assert total == pytest.approx(book.injected, abs=book.tolerance)


def test_energy_that_died_under_the_threshold_shows_as_dissipated() -> None:
    """`E` receives 0.875 against a floor of 1.5 - that energy is gone."""
    config = PropagationConfig()
    result = propagate(_graph(), _SEEDS, config)
    book = ledger.build_ledger(result, _graph(), config)
    assert book.dissipated > 0.5
    assert "E" not in result.activations


def test_polarity_absorption_lands_in_the_destroyed_column() -> None:
    nodes = [
        Node(id="A", layer="chunk", source_id="d0", length=100),
        Node(id="B", layer="chunk", source_id="d1", length=100, polarity=-1),
        Node(id="C", layer="chunk", source_id="d2", length=100),
        Node(id="D", layer="chunk", source_id="d3", length=100),
        Node(id="E", layer="chunk", source_id="d4", length=100),
        Node(id="F", layer="chunk", source_id="d5", length=100),
        Node(id="A_dup", layer="chunk", source_id="d6", length=100),
    ]
    config = PropagationConfig()
    graph = _graph(nodes)
    result = propagate(graph, _SEEDS, config, polarity=PolarityConfig())
    book = ledger.build_ledger(result, graph, config)
    assert book.destroyed.polarity > 0.0
    assert book.destroyed.polarity_events >= 1
    assert book.residual == pytest.approx(0.0, abs=book.tolerance)


def test_conflict_neutralisation_lands_in_the_destroyed_column() -> None:
    config = PropagationConfig()
    graph = _graph()
    negative = conflict_adjacency([NegativeEdge(source="B", target="D", strength=1.0)])
    result = propagate(
        graph, _SEEDS, config, negative=negative, conflict=ConflictConfig()
    )
    book = ledger.build_ledger(result, graph, config)
    assert book.destroyed.conflict > 0.0
    assert book.destroyed.conflict_events == 1
    assert book.residual == pytest.approx(0.0, abs=book.tolerance)


def test_overflow_guard_is_named_rather_than_absorbed_silently() -> None:
    config = PropagationConfig(max_nodes=2)
    graph = _graph()
    result = propagate(graph, _SEEDS, config)
    book = ledger.build_ledger(result, graph, config)
    assert result.stop_reason == "max_nodes"
    assert any("max_nodes" in note for note in book.notes)
    assert book.residual == pytest.approx(0.0, abs=book.tolerance)


def test_dedup_is_reported_but_never_counted_as_destruction() -> None:
    """Dedup redistributes; calling it destruction would invert the invariant."""

    # `A` is active from hop 0, so a candidate that duplicates it is suppressed
    # the moment the web tries to distribute into it.
    def similarity(node: str, others: list[str]) -> list[float]:
        return [1.0 if {node, other} == {"A", "D"} else 0.0 for other in others]

    from spiyweb.config import DedupConfig

    config = PropagationConfig()
    graph = _graph()
    result = propagate(
        graph, _SEEDS, config, similarity=similarity, dedup=DedupConfig(min_pairs=1)
    )
    book = ledger.build_ledger(result, graph, config)
    assert book.dedup_cuts >= 1
    assert book.destroyed.total == pytest.approx(0.0)
    assert book.residual == pytest.approx(0.0, abs=book.tolerance)


def test_mass_marks_the_reconstruction_as_inexact() -> None:
    nodes = [
        Node(id=name, layer="chunk", source_id=f"d{index}", length=100 + index * 40)
        for index, name in enumerate(("A", "B", "C", "D", "E", "F", "A_dup"))
    ]
    config = PropagationConfig(mass=MassConfig(enabled=True))
    graph = _graph(nodes)
    result = propagate(graph, _SEEDS, config)
    book = ledger.build_ledger(result, graph, config)
    assert book.exact is False
    assert any("mass" in note for note in book.notes)
