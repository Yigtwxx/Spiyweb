"""Consolidation pruning (D23): never-used edges leave, reversibly.

The claim under test: an edge is pruned only when it never carried energy
across ENOUGH recorded runs - below `min_runs` nothing moves - and pruning
never destroys information: kept + removed always reassemble the input
layer, so the pass can be undone.
"""

from __future__ import annotations

import pytest

from spiyweb import (
    ConsolidationConfig,
    ConsolidationReport,
    EdgeUsage,
    Graph,
    PropagationConfig,
    PropagationResult,
    propagate,
    prune_layers,
)

# One seed, one neighbour: A holds 10.0, forwards 6.0 to B - the smallest run
# that leaves a (contributor, node) usage pair behind.
SINGLE_EDGE = Graph.from_edges([("A", "B", 0.5)])

LAYERS = {
    "semantic": [("A", "B", 0.5), ("C", "D", 0.4)],
    "entity": [("B", "A", 0.7)],
}


def single_edge_result() -> PropagationResult:
    return propagate(SINGLE_EDGE, {"A": 1.0}, PropagationConfig())


def recorded_usage() -> EdgeUsage:
    usage = EdgeUsage()
    usage.record(single_edge_result())
    return usage


def test_prune_removes_the_never_used_edge_and_keeps_the_used_one() -> None:
    report = prune_layers(LAYERS, recorded_usage(), ConsolidationConfig(min_runs=1))
    assert report.kept["semantic"] == [("A", "B", 0.5)]
    assert report.removed["semantic"] == [("C", "D", 0.4)], (
        "an edge that never carried energy is the one consolidation clears"
    )


def test_usage_matches_either_direction_of_an_undirected_edge() -> None:
    # The entity layer writes the same edge as (B, A); the usage pair is
    # canonical (A, B) - direction must not defeat the match.
    report = prune_layers(LAYERS, recorded_usage(), ConsolidationConfig(min_runs=1))
    assert report.kept["entity"] == [("B", "A", 0.7)]
    assert report.removed["entity"] == []


def test_below_min_runs_nothing_is_pruned() -> None:
    report = prune_layers(LAYERS, recorded_usage(), ConsolidationConfig(min_runs=2))
    assert report.removed == {"semantic": [], "entity": []}, (
        "an unused edge may guard a question nobody asked yet - evidence first"
    )
    assert report.kept == LAYERS
    assert report.runs == 1


def test_kept_and_removed_reassemble_the_input() -> None:
    report = prune_layers(LAYERS, recorded_usage(), ConsolidationConfig(min_runs=1))
    for layer, edges in LAYERS.items():
        restored = report.kept[layer] + report.removed[layer]
        assert sorted(restored) == sorted(edges), (
            "Phase 1 pruning must stay reversible - nothing is deleted"
        )


def test_each_record_counts_one_run() -> None:
    usage = EdgeUsage()
    result = single_edge_result()
    # A colored run records each per-colour result separately - two colours,
    # two runs of evidence.
    assert usage.record(result) == 1
    assert usage.record(result) == 1
    assert usage.runs == 2


def test_snapshot_roundtrip_preserves_counts_and_runs() -> None:
    usage = recorded_usage()
    clone = EdgeUsage.from_dict(usage.to_dict())
    assert clone.runs == usage.runs
    assert clone.used("A", "B")
    assert not clone.used("C", "D")


def test_from_dict_rejects_a_malformed_key() -> None:
    with pytest.raises(ValueError, match="malformed"):
        EdgeUsage.from_dict({"no-separator": 1})


def test_report_is_a_plain_record() -> None:
    report = ConsolidationReport(kept={}, removed={}, runs=0)
    assert report.runs == 0


def test_config_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="min_runs"):
        ConsolidationConfig(min_runs=0)
