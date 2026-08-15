"""The canonical trace from CLAUDE.md section 2.6, as executable regression.

The point of this file is not coverage. It is that node `D` - never the most
similar node to the query - has to come out ranked third, because two weak paths
converge on it. Any change that destroys that ordering is a regression whatever
a benchmark says, and this test is where it fails first.
"""

from __future__ import annotations

import pytest

from spiyweb import Graph, PropagationConfig, PropagationResult, propagate

# Seed Q = 10.0, damping 0.60, threshold 15% of injected energy = 1.5.
#
#   Q --.9--> A --0.0--> A'      (near duplicate: edge suppressed, idea A voted twice)
#             A --.8--> B
#             A --.4--> D
#   Q --.7--> C --.6--> D        (D is reached from two directions)
#             C --.3--> E        (arrives at 0.875, below threshold, dies)
#                       D --.5--> F
CANONICAL_EDGES = [
    ("A", "A_dup", 0.0),
    ("A", "B", 0.8),
    ("A", "D", 0.4),
    ("C", "D", 0.6),
    ("C", "E", 0.3),
    ("D", "F", 0.5),
]
CANONICAL_SEEDS = {"A": 0.9, "C": 0.7}


@pytest.fixture
def canonical_result() -> PropagationResult:
    graph = Graph.from_edges(CANONICAL_EDGES)
    return propagate(graph, CANONICAL_SEEDS, PropagationConfig())


def test_canonical_energies_match_the_worked_example(
    canonical_result: PropagationResult,
) -> None:
    """Exact arithmetic of the documented ledger.

    CLAUDE.md rounds hop 0 to two decimals (5.60 / 4.40) and carries that
    rounding down the trace; the unrounded values are asserted here.
    """
    expected = {
        "A": 10.0 * 0.9 / 1.6,  # 5.625
        "C": 10.0 * 0.7 / 1.6,  # 4.375
        "D": 2.875,  # 1.125 from A + 1.750 from C
        "B": 2.25,
        "F": 1.725,
    }
    assert set(canonical_result.activations) == set(expected)
    for node, energy in expected.items():
        assert canonical_result.energy_of(node) == pytest.approx(energy)


def test_converging_evidence_lifts_the_bridge_node(
    canonical_result: PropagationResult,
) -> None:
    """`D` outranks `B` although `B` sits on the stronger single edge."""
    ranking = [node for node, _ in canonical_result.ranked()]
    assert ranking == ["A", "C", "D", "B", "F"]
    assert canonical_result.energy_of("D") > canonical_result.energy_of("B")


def test_energy_arriving_by_two_paths_is_summed(
    canonical_result: PropagationResult,
) -> None:
    activation = canonical_result.activations["D"]
    assert sorted(activation.contributors) == ["A", "C"]
    assert activation.hop == 1


def test_weak_branch_dies_below_the_threshold(
    canonical_result: PropagationResult,
) -> None:
    """`E` would arrive with 0.875 against a floor of 1.5."""
    assert "E" not in canonical_result.activations
    assert canonical_result.threshold == pytest.approx(1.5)
    assert canonical_result.stop_reason == "threshold"


def test_suppressed_edge_is_excluded_and_its_share_redistributed(
    canonical_result: PropagationResult,
) -> None:
    """The duplicate never activates, and `B` gets the renormalised share."""
    assert "A_dup" not in canonical_result.activations
    forwarded = canonical_result.energy_of("A") * 0.60
    assert canonical_result.energy_of("B") == pytest.approx(forwarded * 0.8 / 1.2)


def test_web_stops_on_its_own_without_a_result_count(
    canonical_result: PropagationResult,
) -> None:
    """`F` forwards 1.035, below the floor, so hop 3 produces nothing."""
    assert canonical_result.hops_used == 2
    assert canonical_result.activations["F"].hop == 2


def test_single_seed_receives_the_whole_injection() -> None:
    graph = Graph.from_edges([("A", "B", 1.0)])
    result = propagate(graph, {"A": 0.42})
    assert result.energy_of("A") == pytest.approx(10.0)


def test_max_hop_is_a_hard_guard() -> None:
    chain = [(f"n{i}", f"n{i + 1}", 1.0) for i in range(10)]
    config = PropagationConfig(damping=0.99, threshold_ratio=0.0, max_hop=3)
    result = propagate(Graph.from_edges(chain), {"n0": 1.0}, config)
    assert result.stop_reason == "max_hop"
    assert result.hops_used == 3
    assert set(result.activations) == {"n0", "n1", "n2", "n3"}


def test_max_nodes_is_a_hard_guard() -> None:
    star = [("hub", f"leaf{i}", 1.0) for i in range(20)]
    config = PropagationConfig(threshold_ratio=0.0, max_nodes=5)
    result = propagate(Graph.from_edges(star), {"hub": 1.0}, config)
    assert result.stop_reason == "max_nodes"
    assert len(result.activations) == 5


def test_isolated_seed_activates_and_stops() -> None:
    graph = Graph.from_edges([("A", "B", 1.0)])
    result = propagate(graph, {"lonely": 1.0})
    assert set(result.activations) == {"lonely"}
    assert result.hops_used == 0


@pytest.mark.parametrize("seeds", [{}, {"A": 0.0}])
def test_unusable_seeds_are_rejected(seeds: dict[str, float]) -> None:
    graph = Graph.from_edges([("A", "B", 1.0)])
    with pytest.raises(ValueError):
        propagate(graph, seeds)


def test_split_alpha_one_matches_the_canonical_trace(
    canonical_result: PropagationResult,
) -> None:
    """`split_alpha=1.0` written out explicitly is the documented behaviour."""
    graph = Graph.from_edges(CANONICAL_EDGES)
    explicit = propagate(graph, CANONICAL_SEEDS, PropagationConfig(split_alpha=1.0))
    assert explicit.ranked() == canonical_result.ranked()


def test_split_alpha_sharpens_the_neighbour_split() -> None:
    """With alpha=2 a 2:1 weight ratio becomes a 4:1 energy ratio."""
    graph = Graph.from_edges([("Q", "strong", 0.8), ("Q", "weak", 0.4)])
    config = PropagationConfig(threshold_ratio=0.0, split_alpha=2.0)
    result = propagate(graph, {"Q": 1.0}, config)
    forwarded = 10.0 * 0.60
    strong = result.energy_of("strong")
    weak = result.energy_of("weak")
    assert strong == pytest.approx(forwarded * 0.64 / 0.80)
    assert weak == pytest.approx(forwarded * 0.16 / 0.80)
    assert strong / weak == pytest.approx(4.0)


def test_split_alpha_does_not_change_the_forwarded_total() -> None:
    """Sharpening reshapes shares; it must never create or destroy energy."""
    graph = Graph.from_edges([("Q", "a", 0.9), ("Q", "b", 0.5), ("Q", "c", 0.2)])
    config = PropagationConfig(threshold_ratio=0.0, split_alpha=3.0)
    result = propagate(graph, {"Q": 1.0}, config)
    arrived = sum(result.energy_of(node) for node in ("a", "b", "c"))
    assert arrived == pytest.approx(10.0 * 0.60)


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_non_positive_split_alpha_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError):
        PropagationConfig(split_alpha=alpha)
