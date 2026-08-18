"""Structural edge builder: nested relations, max-wins weights, omission rule."""

from __future__ import annotations

import pytest

from spiyweb import StructuralEdgeConfig
from spiyweb.edges import ChunkRef, build_structural_edges


def make_chunk(**overrides: object) -> ChunkRef:
    defaults: dict[str, object] = {
        "id": "c1",
        "source_id": "doc-1",
        "position": 0,
        "section_id": None,
    }
    defaults.update(overrides)
    return ChunkRef(**defaults)  # type: ignore[arg-type]


def test_config_defaults_match_the_provisional_hand_weights() -> None:
    config = StructuralEdgeConfig()
    assert config.adjacent == pytest.approx(1.0)
    assert config.same_section == pytest.approx(0.6)
    assert config.same_document == pytest.approx(0.0), (
        "same_document defaults off: per-document cliques feed the hub penalty"
    )


@pytest.mark.parametrize("field", ["adjacent", "same_section", "same_document"])
def test_config_negative_relation_weight_raises_value_error(field: str) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        StructuralEdgeConfig(**{field: -0.1})


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": ""},
        {"source_id": ""},
        {"position": -1},
    ],
)
def test_chunk_ref_invalid_fields_raise_value_error(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        make_chunk(**overrides)


def test_consecutive_positions_produce_an_adjacent_edge() -> None:
    edges = build_structural_edges(
        [make_chunk(id="a", position=0), make_chunk(id="b", position=1)]
    )
    assert edges == [("a", "b", pytest.approx(1.0))]


def test_position_gap_still_counts_as_adjacent() -> None:
    edges = build_structural_edges(
        [make_chunk(id="a", position=0), make_chunk(id="b", position=7)]
    )
    assert edges == [("a", "b", pytest.approx(1.0))], (
        "adjacency means consecutive in sorted order, robust to position gaps"
    )


def test_same_section_non_adjacent_pair_gets_the_section_weight() -> None:
    edges = build_structural_edges(
        [
            make_chunk(id="a", position=0, section_id="s1"),
            make_chunk(id="b", position=1, section_id="s2"),
            make_chunk(id="c", position=2, section_id="s1"),
        ]
    )
    weights = {(u, v): w for u, v, w in edges}
    assert weights[("a", "c")] == pytest.approx(0.6)


def test_adjacent_same_section_pair_takes_max_not_sum() -> None:
    edges = build_structural_edges(
        [
            make_chunk(id="a", position=0, section_id="s1"),
            make_chunk(id="b", position=1, section_id="s1"),
        ]
    )
    assert edges == [("a", "b", pytest.approx(1.0))], (
        "nested relations are not independent evidence - max wins, never sum"
    )


def test_disabled_adjacent_demotes_the_pair_to_its_section_weight() -> None:
    edges = build_structural_edges(
        [
            make_chunk(id="a", position=0, section_id="s1"),
            make_chunk(id="b", position=1, section_id="s1"),
        ],
        config=StructuralEdgeConfig(adjacent=0.0),
    )
    assert edges == [("a", "b", pytest.approx(0.6))]


def test_all_relations_disabled_emit_nothing_and_never_a_zero_weight() -> None:
    edges = build_structural_edges(
        [
            make_chunk(id="a", position=0, section_id="s1"),
            make_chunk(id="b", position=1, section_id="s1"),
        ],
        config=StructuralEdgeConfig(adjacent=0.0, same_section=0.0),
    )
    assert edges == [], (
        "0.0 is the dedup-suppressed-edge contract; a disabled relation omits"
    )


def test_same_document_off_by_default_leaves_distant_pairs_unconnected() -> None:
    edges = build_structural_edges(
        [
            make_chunk(id="a", position=0, section_id="s1"),
            make_chunk(id="b", position=1, section_id="s1"),
            make_chunk(id="c", position=2, section_id="s2"),
        ]
    )
    pairs = {(u, v) for u, v, _ in edges}
    assert ("a", "c") not in pairs, (
        "cross-section non-adjacent same-doc pair needs same_document enabled"
    )


def test_same_document_enabled_builds_the_full_clique() -> None:
    edges = build_structural_edges(
        [
            make_chunk(id="a", position=0),
            make_chunk(id="b", position=1),
            make_chunk(id="c", position=2),
        ],
        config=StructuralEdgeConfig(same_document=0.3),
    )
    weights = {(u, v): w for u, v, w in edges}
    assert weights[("a", "c")] == pytest.approx(0.3)
    assert weights[("a", "b")] == pytest.approx(1.0), "adjacent still wins by max"


def test_none_sections_never_group_together() -> None:
    edges = build_structural_edges(
        [
            make_chunk(id="a", position=0, section_id=None),
            make_chunk(id="b", position=1, section_id="s1"),
            make_chunk(id="c", position=2, section_id=None),
        ]
    )
    pairs = {(u, v) for u, v, _ in edges}
    assert ("a", "c") not in pairs, "None means no section, not a shared section"


def test_edges_never_cross_documents() -> None:
    edges = build_structural_edges(
        [
            make_chunk(id="a", source_id="doc-1", position=0, section_id="s1"),
            make_chunk(id="b", source_id="doc-2", position=1, section_id="s1"),
        ],
        config=StructuralEdgeConfig(same_document=0.3),
    )
    assert edges == []


def test_duplicate_position_in_one_document_raises_value_error() -> None:
    with pytest.raises(ValueError, match="share position"):
        build_structural_edges(
            [make_chunk(id="a", position=3), make_chunk(id="b", position=3)]
        )


def test_duplicate_chunk_id_raises_value_error() -> None:
    with pytest.raises(ValueError, match="duplicate chunk id"):
        build_structural_edges(
            [
                make_chunk(id="a", position=0),
                make_chunk(id="a", source_id="doc-2", position=0),
            ]
        )


def test_empty_and_single_chunk_inputs_yield_no_edges() -> None:
    assert build_structural_edges([]) == []
    assert build_structural_edges([make_chunk()]) == []


def test_output_is_canonically_ordered_and_sorted() -> None:
    edges = build_structural_edges(
        [
            make_chunk(id="z", position=0),
            make_chunk(id="a", position=1),
            make_chunk(id="m", position=2),
        ]
    )
    assert edges == sorted(edges)
    assert all(u < v for u, v, _ in edges), "each pair is emitted as (min, max)"
