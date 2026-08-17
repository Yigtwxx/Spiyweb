"""Hebbian learned layer: usage thickens threads, forgetting thins them.

The claim under test: learning lives in its OWN layer - the base graph is
never touched, `LayerWeights.learned = 0.0` kills the layer's whole effect
(the read-side ablation), and `LearnedLayerConfig.enabled = False` freezes
the write side without discarding what was learned.
"""

from __future__ import annotations

import pytest

from spiyweb import (
    Graph,
    LayerWeights,
    LearnedLayer,
    LearnedLayerConfig,
    PropagationConfig,
    PropagationResult,
    propagate,
)

# One seed, one neighbour: A holds 10.0, forwards 6.0 to B - the smallest run
# that exercises a (contributor, node) pair.
SINGLE_EDGE = Graph.from_edges([("A", "B", 0.5)])


def single_edge_result() -> PropagationResult:
    return propagate(SINGLE_EDGE, {"A": 1.0}, PropagationConfig())


def test_reinforce_strengthens_the_carrying_edge_proportionally() -> None:
    layer = LearnedLayer()
    touched = layer.reinforce(single_edge_result())
    # carried = 6.0 / 1 contributor, injected = 10.0 -> 0.1 * 0.6 = 0.06.
    assert touched == 1
    assert layer.edges() == [("A", "B", pytest.approx(0.06))]


def test_second_round_ages_before_adding() -> None:
    layer = LearnedLayer()
    result = single_edge_result()
    layer.reinforce(result)
    layer.reinforce(result)
    # 0.06 * 0.99 + 0.06 = 0.1194 - ageing fires on every round.
    strength = layer.edges()[0][2]
    assert strength == pytest.approx(0.06 * 0.99 + 0.06)


def test_strength_saturates_at_max_strength() -> None:
    layer = LearnedLayer(LearnedLayerConfig(max_strength=0.1))
    result = single_edge_result()
    layer.reinforce(result)
    layer.reinforce(result)
    assert layer.edges()[0][2] == pytest.approx(0.1), (
        "the owner's saturation cap: proportional growth, never unbounded"
    )


def test_unused_edges_thin_out_while_others_are_reinforced() -> None:
    layer = LearnedLayer.from_dict({"X\tY": 0.5})
    layer.reinforce(single_edge_result())
    strengths = dict((f"{u}-{v}", w) for u, v, w in layer.edges())
    assert strengths["X-Y"] == pytest.approx(0.5 * 0.99), (
        "forgetting is mandatory - an edge no run uses must keep thinning"
    )
    assert strengths["A-B"] == pytest.approx(0.06)


def test_disabled_layer_neither_learns_nor_ages() -> None:
    layer = LearnedLayer.from_dict({"X\tY": 0.5}, LearnedLayerConfig(enabled=False))
    assert layer.reinforce(single_edge_result()) == 0
    assert layer.edges() == [("X", "Y", 0.5)], "frozen, not discarded"


def test_accepted_filter_narrows_the_evidence() -> None:
    layer = LearnedLayer()
    assert layer.reinforce(single_edge_result(), accepted={"C"}) == 0
    assert layer.reinforce(single_edge_result(), accepted={"B"}) == 1


def test_seeds_alone_produce_no_edges() -> None:
    isolated = Graph.from_edges([("X", "Y", 0.5)])
    result = propagate(isolated, {"A": 1.0}, PropagationConfig())
    layer = LearnedLayer()
    assert layer.reinforce(result) == 0, "a seed has no contributor, no edge"


def test_learned_layer_revives_a_dead_neighbour_only_when_weighted_in() -> None:
    """The end-to-end ablation proof: same base graph, learned weight decides.

    In the base graph B's share of A's 6.0 is 6.0 * .1/1.0 = 0.6 < 1.5 and B
    dies. A learned edge of strength 0.5 lifts the merged weight to 0.6, B
    receives 2.4 and lives - but ONLY when `LayerWeights.learned` is nonzero.
    """
    layers = {"semantic": [("A", "B", 0.1), ("A", "C", 0.9)]}
    learned = LearnedLayer.from_dict({"A\tB": 0.5})
    seeds = {"A": 1.0}
    weights_off = LayerWeights(semantic=1.0, learned=0.0)
    weights_on = LayerWeights(semantic=1.0, learned=1.0)

    dead = propagate(
        Graph.from_layers({**layers, "learned": learned.edges()}, weights=weights_off),
        seeds,
        PropagationConfig(),
    )
    alive = propagate(
        Graph.from_layers({**layers, "learned": learned.edges()}, weights=weights_on),
        seeds,
        PropagationConfig(),
    )
    assert "B" not in dead.activations
    assert alive.energy_of("B") == pytest.approx(6.0 * 0.6 / 1.5)


def test_prune_discards_below_threshold() -> None:
    layer = LearnedLayer.from_dict({"A\tB": 0.05, "C\tD": 0.5})
    assert layer.prune(0.1) == 1
    assert layer.edges() == [("C", "D", 0.5)]


def test_snapshot_roundtrip_preserves_the_layer() -> None:
    layer = LearnedLayer.from_dict({"A\tB": 0.25, "C\tD": 0.75})
    clone = LearnedLayer.from_dict(layer.to_dict())
    assert clone.edges() == layer.edges()


def test_from_dict_rejects_a_malformed_key() -> None:
    with pytest.raises(ValueError, match="malformed"):
        LearnedLayer.from_dict({"no-separator": 0.5})


def test_config_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        LearnedLayerConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="forgetting"):
        LearnedLayerConfig(forgetting=1.5)
    with pytest.raises(ValueError, match="max_strength"):
        LearnedLayerConfig(max_strength=0.0)
