"""Corpus lint: the four ways a knowledge base can be shaped wrong.

Phase 2.7, and the project's plan B. Every other diagnostic here answers "why
did THIS query go badly"; this one answers whether the corpus is shaped so
that retrieval can work at all - a question that needs no query to ask.

The fixtures are hand-built graphs rather than an indexed corpus on purpose.
A finding has to be checkable by reading the test: eleven atoms around one
hub, two islands, one pair at 0.99 cosine. If the input needs a paragraph to
describe, the assertion is not pinning what it claims to pin.
"""

from __future__ import annotations

import pytest

from spiyweb.config import CorpusLintConfig, LayerWeights
from spiyweb.core.conflict import NegativeEdge
from spiyweb.core.graph import Graph, Node
from spiyweb.lint import lint_corpus, source_summary


def _nodes(*pairs: tuple[str, str]) -> tuple[Node, ...]:
    return tuple(
        Node(id=node, layer="chunk", source_id=source, length=100)
        for node, source in pairs
    )


# --- orphans ---------------------------------------------------------------


def test_an_island_is_reported_and_the_main_mass_is_not() -> None:
    """Something has to be the main mass; calling it an orphan says nothing."""
    graph = Graph.from_edges(
        [("a", "b", 0.9), ("b", "c", 0.8), ("x", "y", 0.7)],
    )
    report = lint_corpus(graph)
    orphans = report.by_kind("orphan")
    assert len(orphans) == 1
    assert set(orphans[0].nodes) == {"x", "y"}
    assert report.components == 2
    assert report.largest_component == 3


def test_a_lone_atom_is_isolated_rather_than_an_island() -> None:
    """One unconnected atom is a different problem with a different fix."""
    graph = Graph.from_edges([("a", "b", 0.9)], nodes=_nodes(("lonely", "doc")))
    report = lint_corpus(graph)
    assert report.by_kind("orphan") == ()
    assert [f.subject for f in report.by_kind("isolated")] == ["lonely"]
    assert report.isolated == 1


def test_a_suppressed_edge_does_not_connect_anything() -> None:
    """A zero-weight edge carries no energy, so it bridges no island."""
    graph = Graph.from_edges([("a", "b", 0.9), ("b", "x", 0.0), ("x", "y", 0.7)])
    report = lint_corpus(graph)
    assert report.components == 2, "a cut edge was counted as connective"


def test_a_connected_corpus_reports_no_island() -> None:
    graph = Graph.from_edges([("a", "b", 0.9), ("b", "c", 0.8)])
    report = lint_corpus(graph)
    assert report.by_kind("orphan") == ()
    assert report.components == 1


# --- hubs ------------------------------------------------------------------


def test_a_hub_is_measured_by_the_share_it_forwards_not_its_degree() -> None:
    """Known-risk #2 of CLAUDE.md §8, as a number rather than an assertion."""
    graph = Graph.from_edges(
        [("hub", f"n{i}", 0.5) for i in range(40)],
        nodes=_nodes(("hub", "doc")),
    )
    report = lint_corpus(graph)
    hubs = report.by_kind("hub")
    assert [f.subject for f in hubs] == ["hub"]
    # Forty equal neighbours: each takes exactly 1/40 of what is forwarded.
    assert hubs[0].value == pytest.approx(1 / 40)
    assert "40 neighbours" in hubs[0].message


def test_a_dominant_edge_makes_a_high_degree_node_harmless() -> None:
    """Eight hundred neighbours behind one strong edge waste nothing."""
    graph = Graph.from_edges(
        [("hub", "strong", 100.0), *[("hub", f"n{i}", 0.01) for i in range(40)]],
    )
    assert lint_corpus(graph).by_kind("hub") == ()


def test_the_hub_share_uses_the_configured_split_exponent() -> None:
    """The lint must describe the propagation actually being run."""
    edges = [("hub", "strong", 0.9), *[("hub", f"n{i}", 0.3) for i in range(30)]]
    graph = Graph.from_edges(edges)
    # A floor of 12% straddles the two: the plain split gives the strongest
    # edge 9.1%, the cubed one gives it 50%.
    plain = lint_corpus(
        graph, config=CorpusLintConfig(split_alpha=1.0, hub_share_floor=0.12)
    )
    sharp = lint_corpus(
        graph, config=CorpusLintConfig(split_alpha=3.0, hub_share_floor=0.12)
    )
    # A higher exponent concentrates energy on the strongest edge, so the
    # same graph is less of a hub - and the report has to say so.
    assert plain.by_kind("hub")
    assert not sharp.by_kind("hub")


def test_a_small_node_is_never_a_hub() -> None:
    graph = Graph.from_edges([("a", f"n{i}", 0.5) for i in range(5)])
    assert lint_corpus(graph).by_kind("hub") == ()


# --- duplicates ------------------------------------------------------------


def test_near_identical_passages_are_found_in_the_raw_cosine_layer() -> None:
    """The merged adjacency sums layers, so a merged weight is not a cosine."""
    graph = Graph.from_edges(
        [("a", "b", 1.4)], nodes=_nodes(("a", "doc1"), ("b", "doc2"))
    )
    report = lint_corpus(graph, semantic_edges=[("a", "b", 0.99)])
    duplicates = report.by_kind("duplicate")
    assert len(duplicates) == 1
    assert duplicates[0].nodes == ("a", "b")
    assert duplicates[0].value == pytest.approx(0.99)


def test_a_merged_weight_is_never_mistaken_for_a_cosine() -> None:
    """Without the raw layer there are no duplicate findings, and that is right."""
    graph = Graph.from_edges([("a", "b", 1.4)])
    assert lint_corpus(graph).by_kind("duplicate") == ()


def test_repetition_inside_one_document_is_reported_separately() -> None:
    """Within-source repetition never becomes a vote (D7), so it is pure cost."""
    graph = Graph.from_edges(
        [("a", "b", 0.9), ("b", "c", 0.9)],
        nodes=_nodes(("a", "same"), ("b", "same"), ("c", "other")),
    )
    report = lint_corpus(graph, semantic_edges=[("a", "b", 0.99), ("b", "c", 0.97)])
    inside = report.by_kind("duplicate_source")
    assert [f.subject for f in inside] == ["same"]
    assert inside[0].value == 1.0, "only the within-source pair counts"


def test_the_duplicate_threshold_is_a_reporting_knob() -> None:
    graph = Graph.from_edges([("a", "b", 0.9)])
    assert (
        lint_corpus(graph, semantic_edges=[("a", "b", 0.9)]).by_kind("duplicate") == ()
    )
    loose = lint_corpus(
        graph,
        semantic_edges=[("a", "b", 0.9)],
        config=CorpusLintConfig(duplicate_weight=0.85),
    )
    assert loose.by_kind("duplicate")


# --- contradictions --------------------------------------------------------


def test_the_contradiction_map_aggregates_by_source() -> None:
    """A document that disagrees twenty times is one problem, not twenty."""
    graph = Graph.from_edges(
        [("a", "b", 0.5), ("a", "c", 0.5)],
        nodes=_nodes(("a", "loud"), ("b", "quiet"), ("c", "other")),
    )
    report = lint_corpus(
        graph,
        negative_edges=[
            NegativeEdge(source="a", target="b", strength=0.9),
            NegativeEdge(source="a", target="c", strength=0.8),
        ],
    )
    contradictions = report.by_kind("contradiction")
    assert contradictions[0].subject == "loud"
    assert contradictions[0].value == 2.0
    assert "2 other source(s)" in contradictions[0].message


def test_a_contradiction_inside_one_source_is_not_a_corpus_finding() -> None:
    """Two propositions of one passage arguing is an extraction problem."""
    graph = Graph.from_edges(
        [("a", "b", 0.5)], nodes=_nodes(("a", "same"), ("b", "same"))
    )
    report = lint_corpus(
        graph, negative_edges=[NegativeEdge(source="a", target="b", strength=0.9)]
    )
    assert report.by_kind("contradiction") == ()


# --- the report ------------------------------------------------------------


def test_the_report_counts_what_it_found() -> None:
    graph = Graph.from_edges([("a", "b", 0.9), ("x", "y", 0.7)])
    report = lint_corpus(graph, semantic_edges=[("a", "b", 0.99)])
    assert report.counts == {"orphan": 1, "duplicate": 1}
    assert "2 connected component(s)" in report.text
    assert "1 duplicate, 1 orphan" in report.text


def test_a_healthy_corpus_says_so_in_words() -> None:
    graph = Graph.from_edges([("a", "b", 0.9), ("b", "c", 0.8)])
    report = lint_corpus(graph)
    assert report.findings == ()
    assert "No structural problem" in report.text


def test_findings_are_capped_per_kind() -> None:
    """Ten thousand orphans need the worst twenty and a count, not ten thousand."""
    edges = [(f"a{i}", f"b{i}", 0.9) for i in range(30)]
    report = lint_corpus(
        Graph.from_edges(edges), config=CorpusLintConfig(max_per_kind=5)
    )
    assert len(report.by_kind("orphan")) == 5


def test_the_roll_up_ranks_the_worst_subject_first() -> None:
    graph = Graph.from_edges(
        [("a", f"n{i}", 0.5) for i in range(40)], nodes=_nodes(("a", "doc"))
    )
    report = lint_corpus(graph, semantic_edges=[("a", "n0", 0.99)])
    worst = source_summary(report)
    assert next(iter(worst)) == "a"


def test_an_empty_corpus_does_not_divide_by_zero() -> None:
    report = lint_corpus(Graph())
    assert report.nodes == 0
    assert report.components == 0
    assert "0 atom(s)" in report.text


# --- the config is a reporting knob, never a mechanism ---------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_orphan_nodes", 1, "min_orphan_nodes"),
        ("min_hub_degree", 1, "min_hub_degree"),
        ("hub_share_floor", 0.0, "hub_share_floor"),
        ("hub_share_floor", 1.0, "hub_share_floor"),
        ("split_alpha", 0.0, "split_alpha"),
        ("duplicate_weight", 0.0, "duplicate_weight"),
        ("duplicate_weight", 1.5, "duplicate_weight"),
        ("max_per_kind", 0, "max_per_kind"),
        ("max_nodes_per_finding", 0, "max_nodes_per_finding"),
    ],
)
def test_a_nonsense_threshold_is_refused(
    field: str, value: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CorpusLintConfig(**{field: value})


# --- empty layers ----------------------------------------------------------


def test_a_layer_merged_at_a_real_weight_that_carries_nothing_is_reported() -> None:
    """Phase 2.8's signal: both known instances went unnoticed for months.

    The structural layer was empty in all four sealed Phase 1 indexes and the
    entity layer was empty on every corpus under a hundred chunks. The number
    came out, the layer was configured, and nothing connected the two facts.
    """
    graph = Graph.from_edges([("a", "b", 0.9)], nodes=_nodes(("a", "d1"), ("b", "d2")))
    report = lint_corpus(
        graph,
        layer_edges={"semantic": 12, "entity": 0, "structural": 0},
        weights=LayerWeights(),
    )
    empty = report.by_kind("empty_layer")
    assert [f.subject for f in empty] == ["entity", "structural"]
    assert empty[0].value == 1.0, "the entity layer's merge weight"
    assert "does nothing on this corpus" in empty[1].message


def test_a_layer_switched_off_on_purpose_is_not_a_finding() -> None:
    """`learned` is disabled by default; an empty disabled layer is expected."""
    graph = Graph.from_edges([("a", "b", 0.9)])
    report = lint_corpus(
        graph, layer_edges={"learned": 0}, weights=LayerWeights(learned=0.0)
    )
    assert report.by_kind("empty_layer") == ()


def test_the_single_chunk_cause_is_named_when_it_is_knowable() -> None:
    """Phase 1's side finding, explained rather than merely restated.

    Every sealed index holds exactly one chunk per source, so the structural
    layer's within-document relations cannot exist. That is not a bug, and a
    report that said only "empty" would have kept it looking like one.
    """
    graph = Graph.from_edges(
        [("a:0", "b:0", 0.9)], nodes=_nodes(("a:0", "a"), ("b:0", "b"))
    )
    report = lint_corpus(graph, layer_edges={"structural": 0}, weights=LayerWeights())
    assert "single chunk" in report.by_kind("empty_layer")[0].message


def test_a_multi_chunk_corpus_gets_no_invented_cause() -> None:
    graph = Graph.from_edges(
        [("a:0", "a:1", 0.9)], nodes=_nodes(("a:0", "a"), ("a:1", "a"))
    )
    report = lint_corpus(graph, layer_edges={"structural": 0}, weights=LayerWeights())
    assert "single chunk" not in report.by_kind("empty_layer")[0].message


def test_without_layer_counts_there_is_nothing_to_say() -> None:
    """The merged adjacency cannot tell an empty layer from an absent one."""
    graph = Graph.from_edges([("a", "b", 0.9)])
    assert lint_corpus(graph).by_kind("empty_layer") == ()
