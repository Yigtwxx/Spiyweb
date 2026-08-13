"""Node schema and layer-weight configuration rules."""

from __future__ import annotations

import pytest

from spiyweb import LayerWeights, Node


def make_node(**overrides: object) -> Node:
    """A valid chunk node; tests override single fields to probe validation."""
    defaults: dict[str, object] = {
        "id": "A",
        "layer": "chunk",
        "source_id": "doc-1",
        "length": 120,
    }
    defaults.update(overrides)
    return Node(**defaults)  # type: ignore[arg-type]


def test_node_minimal_construction_uses_documented_defaults() -> None:
    node = make_node()
    assert node.timestamp is None
    assert node.cluster_id is None
    assert node.polarity == 1


@pytest.mark.parametrize("field", ["id", "source_id"])
def test_node_empty_identity_field_raises_value_error(field: str) -> None:
    with pytest.raises(ValueError):
        make_node(**{field: ""})


def test_node_unknown_layer_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown node layer"):
        make_node(layer="paragraph")


@pytest.mark.parametrize("length", [0, -3])
def test_node_non_positive_length_raises_value_error(length: int) -> None:
    with pytest.raises(ValueError, match="length"):
        make_node(length=length)


@pytest.mark.parametrize("polarity", [0, 2, -2])
def test_node_polarity_outside_plus_minus_one_raises_value_error(
    polarity: int,
) -> None:
    with pytest.raises(ValueError, match="polarity"):
        make_node(polarity=polarity)


def test_layer_weights_defaults_match_phase1_hand_weights() -> None:
    weights = LayerWeights()
    assert weights.semantic == pytest.approx(0.5)
    assert weights.entity == pytest.approx(1.0)
    assert weights.structural == pytest.approx(0.3)
    assert weights.learned == 0.0, "learned layer must ship disabled in Phase 1"


@pytest.mark.parametrize("field", ["semantic", "entity", "structural", "learned"])
def test_layer_weights_negative_value_raises_value_error(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        LayerWeights(**{field: -0.1})


def test_layer_weights_all_zero_is_a_legal_ablation() -> None:
    weights = LayerWeights(semantic=0.0, entity=0.0, structural=0.0, learned=0.0)
    assert weights.weight_of("entity") == 0.0


def test_layer_weights_weight_of_matches_fields() -> None:
    weights = LayerWeights(semantic=0.7)
    assert weights.weight_of("semantic") == pytest.approx(0.7)
    assert weights.weight_of("entity") == pytest.approx(1.0)
