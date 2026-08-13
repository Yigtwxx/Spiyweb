"""Chunker: Node + ChunkRef pairing, deterministic ids, validation."""

from __future__ import annotations

import pytest

from spiyweb.edges import build_structural_edges
from spiyweb.nodes import Chunk, DocumentInput, TextUnit, chunk_documents


def make_document(**overrides: object) -> DocumentInput:
    defaults: dict[str, object] = {
        "source_id": "doc-1",
        "units": (TextUnit("first paragraph"), TextUnit("second paragraph")),
    }
    defaults.update(overrides)
    return DocumentInput(**defaults)  # type: ignore[arg-type]


def test_chunk_ids_follow_the_source_position_scheme() -> None:
    chunks = chunk_documents([make_document()])
    assert [chunk.node.id for chunk in chunks] == ["doc-1:0", "doc-1:1"]


def test_node_and_ref_agree_on_id_and_source() -> None:
    for chunk in chunk_documents([make_document()]):
        assert chunk.node.id == chunk.ref.id, (
            "the chunker is the single place keeping the two faces in step"
        )
        assert chunk.node.source_id == chunk.ref.source_id


def test_node_length_is_the_character_count_of_the_text() -> None:
    chunks = chunk_documents([make_document(units=(TextUnit("abcde"),))])
    assert chunks[0].node.length == 5
    assert chunks[0].text == "abcde"


def test_node_layer_is_chunk_and_position_follows_unit_order() -> None:
    chunks = chunk_documents([make_document()])
    assert all(chunk.node.layer == "chunk" for chunk in chunks)
    assert [chunk.ref.position for chunk in chunks] == [0, 1]


def test_section_and_timestamp_pass_through() -> None:
    document = make_document(
        units=(TextUnit("intro text", section_id="intro"),),
        timestamp=1700000000.0,
    )
    (chunk,) = chunk_documents([document])
    assert chunk.ref.section_id == "intro"
    assert chunk.node.timestamp == pytest.approx(1700000000.0)


def test_whitespace_only_unit_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        TextUnit("   \n\t")


def test_empty_source_id_raises_value_error() -> None:
    with pytest.raises(ValueError, match="source_id"):
        make_document(source_id="")


def test_duplicate_source_id_across_documents_raises() -> None:
    with pytest.raises(ValueError, match="duplicate document source_id"):
        chunk_documents([make_document(), make_document()])


def test_empty_input_and_unitless_document_yield_no_chunks() -> None:
    assert chunk_documents([]) == []
    assert chunk_documents([make_document(units=())]) == []


def test_refs_feed_the_structural_builder_directly() -> None:
    # The contract seam: the chunker's output is the structural builder's
    # input, with no adaptation layer in between.
    chunks = chunk_documents([make_document()])
    edges = build_structural_edges([chunk.ref for chunk in chunks])
    assert edges == [("doc-1:0", "doc-1:1", pytest.approx(1.0))]


def test_chunk_is_frozen() -> None:
    (chunk, _) = chunk_documents([make_document()])
    assert isinstance(chunk, Chunk)
    with pytest.raises(AttributeError):
        chunk.text = "mutated"  # type: ignore[misc]
