"""Corpus lint: what is wrong with the knowledge base, not with the query.

Phase 2's primary product (D37), and the project's plan B. Every retrieval
diagnostic this library ships answers "why did THIS query go badly". This one
answers a question nobody has to ask a question to ask: **is this corpus
shaped so that retrieval can work at all?**

That question has an answer in the graph's topology alone, before any query
exists, and the four findings are the four ways the shape can be wrong:

- **Orphans** - connected components with no path to the rest. An island of
  knowledge is unreachable unless the seed lands directly on it, which means
  multi-hop retrieval cannot help it and neither can anything else.
- **Hubs** - the known-risk #2 of CLAUDE.md §8 made measurable. Proportional
  splitting divides a node's energy among its neighbours, so a node with
  eight hundred of them forwards dust. The metric is not degree: it is the
  share the STRONGEST neighbour actually receives, computed with the same
  `split_alpha` the propagation would use.
- **Duplicates** - near-identical passages, found statically in the semantic
  layer's cosines. A static proxy for a dynamic mechanism, and it says so.
- **Contradictions** - the index-time NLI edges, aggregated by source, so a
  document that argues with the rest of the corpus is visible as one row.
- **Empty layers** - a layer merged at a non-zero weight that carries no
  edges at all. Added 2026-08-26 after two of these turned up in one day:
  the structural layer was empty in all four sealed Phase 1 indexes, and the
  entity layer was empty on every corpus under a hundred chunks. Neither
  said so. A weight that does nothing is not a neutral setting - it is a
  belief about the graph that the graph does not share.

Pure and dependency-free, like `output.py`: a graph and some edge lists in,
findings out. Reading the artifacts is `spiyweb.indexing`'s job, and the
report's prose is template-built - there is no LLM anywhere near this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiyweb.config import CorpusLintConfig, LayerWeights

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from spiyweb.core.conflict import NegativeEdge
    from spiyweb.core.graph import Graph

__all__ = ["Finding", "LintReport", "lint_corpus", "source_summary"]

ORPHAN_TEMPLATE = (
    "{count} atom(s) around {label} form an island: nothing connects them to "
    "the rest of the corpus, so only a query that lands directly on them can "
    "reach them."
)

ISOLATED_TEMPLATE = (
    "{label} has no edges at all - it can only ever be a seed, never a hop."
)

HUB_TEMPLATE = (
    "{label} has {degree} neighbours; its strongest one receives {share:.1%} "
    "of what it forwards. Energy arriving here is ground into dust before it "
    "travels."
)

DUPLICATE_TEMPLATE = (
    "{label} and {other} are {weight:.3f} cosine apart in the semantic layer "
    "- near-identical passages that will compete for the same seed slot."
)

DUPLICATE_SOURCE_TEMPLATE = (
    "{label} carries {count} near-duplicate pair(s) inside itself; repetition "
    "within one document is not corroboration and its votes will not rise."
)

EMPTY_LAYER_TEMPLATE = (
    "the {layer} layer carries no edges at all, yet it is merged at weight "
    "{weight}: that weight does nothing on this corpus.{because}"
)

SINGLE_CHUNK_REASON = (
    " Every document here is a single chunk, so the layer's within-document "
    "relations (adjacent, same section) cannot exist."
)

SMALL_CORPUS_REASON = (
    " The corpus may be too small for the entity layer's document-frequency "
    "guard - an entity has to appear in at least two chunks to pair."
)

CONTRADICTION_TEMPLATE = (
    "{label} contradicts {others} other source(s) across {count} marked "
    "pair(s) - the densest disagreement in this corpus."
)

REPORT_TEMPLATE = (
    "{nodes} atom(s), {edges} edge(s), {components} connected component(s); "
    "the largest holds {largest} atom(s) ({largest_share:.1%} of the corpus).\n"
    "{summary}"
)

NOTHING_FOUND = "No structural problem passed the configured thresholds."

_SMALL_CORPUS = 100
"""Below this many atoms the entity layer's `max_df_ratio` guard is the
likeliest reason it came out empty - `0.02 * n` falls under 2 there, and an
entity needs two chunks to pair. Used only to word the explanation."""


@dataclass(frozen=True)
class Finding:
    """One thing wrong with the corpus's shape.

    Attributes:
        kind: `"orphan"`, `"isolated"`, `"hub"`, `"duplicate"`,
            `"duplicate_source"`, `"contradiction"` or `"empty_layer"`.
        subject: What the finding is about - a node id, a source id, or a
            component's strongest member.
        value: The number that made it a finding, in the kind's own unit
            (atoms for an orphan, a share for a hub, a cosine for a
            duplicate, a pair count for a contradiction). Sorting within a
            kind is on this.
        nodes: The atoms involved, sorted; capped by config for the big kinds.
        message: Template-built, LLM-free explanation.
    """

    kind: str
    subject: str
    value: float
    nodes: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class LintReport:
    """The corpus's structural health, as numbers and as prose.

    `text` is a summary a person reads; `findings` is what a tool consumes.
    Neither is a verdict - which of these matters is the corpus owner's call,
    the same way `Confidence` leaves "I don't know" to the caller (D17).
    """

    nodes: int
    edges: int
    components: int
    largest_component: int
    isolated: int
    findings: tuple[Finding, ...]
    text: str

    def by_kind(self, kind: str) -> tuple[Finding, ...]:
        """The findings of one kind, worst first."""
        return tuple(finding for finding in self.findings if finding.kind == kind)

    @property
    def counts(self) -> dict[str, int]:
        """How many findings of each kind, for a summary line."""
        totals: dict[str, int] = {}
        for finding in self.findings:
            totals[finding.kind] = totals.get(finding.kind, 0) + 1
        return totals


def lint_corpus(
    graph: Graph,
    *,
    semantic_edges: Iterable[tuple[str, str, float]] = (),
    negative_edges: Iterable[NegativeEdge] = (),
    layer_edges: Mapping[str, int] | None = None,
    weights: LayerWeights | None = None,
    config: CorpusLintConfig | None = None,
) -> LintReport:
    """Inspect a corpus's shape. No query, no model, no I/O.

    `semantic_edges` are the RAW cosine layer, not the merged adjacency: the
    merge sums layers, so a merged weight is not a cosine and cannot answer
    "are these two passages near-identical". Callers who do not have the
    layer simply get no duplicate findings, which is honest - the mechanism
    that finds duplicates at query time is dynamic and adaptive anyway, and
    this static pass never claims to be it.
    """
    cfg = config if config is not None else CorpusLintConfig()
    components = _components(graph)
    largest = max((len(part) for part in components), default=0)
    findings: list[Finding] = []
    findings.extend(_orphans(graph, components, largest, cfg))
    findings.extend(_hubs(graph, cfg))
    findings.extend(_duplicates(graph, semantic_edges, cfg))
    findings.extend(_contradictions(negative_edges, graph, cfg))
    findings.extend(_empty_layers(graph, layer_edges, weights))

    node_count = len(set(graph.adjacency) | set(graph.node_data))
    edge_count = sum(
        1
        for source, targets in graph.adjacency.items()
        for target, weight in targets.items()
        if weight > 0.0 and source < target
    )
    isolated = sum(1 for part in components if len(part) == 1)
    counts = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    summary = (
        ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
        if counts
        else NOTHING_FOUND
    )
    return LintReport(
        nodes=node_count,
        edges=edge_count,
        components=len(components),
        largest_component=largest,
        isolated=isolated,
        findings=tuple(findings),
        text=REPORT_TEMPLATE.format(
            nodes=node_count,
            edges=edge_count,
            components=len(components),
            largest=largest,
            largest_share=largest / node_count if node_count else 0.0,
            summary=summary,
        ),
    )


# --- internals -------------------------------------------------------------


def _components(graph: Graph) -> list[tuple[str, ...]]:
    """Connected components over POSITIVE edges, largest first.

    A zero-weight edge is a suppressed one, and a suppressed edge carries no
    energy - treating it as connective would report an island as connected
    on the strength of a link nothing can cross.
    """
    # The union, not the adjacency: `node_data` may carry an atom the
    # adjacency never mentions, and an atom with no edges at all is exactly
    # the thing `isolated` exists to report. Walking the adjacency alone made
    # that finding structurally impossible to produce.
    unvisited = set(graph.adjacency) | set(graph.node_data)
    parts: list[tuple[str, ...]] = []
    while unvisited:
        start = min(unvisited)
        seen = {start}
        frontier = [start]
        while frontier:
            node = frontier.pop()
            for neighbor, weight in graph.neighbors(node).items():
                if weight > 0.0 and neighbor in unvisited and neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        unvisited -= seen
        parts.append(tuple(sorted(seen)))
    return sorted(parts, key=len, reverse=True)


def _orphans(
    graph: Graph,
    components: Sequence[tuple[str, ...]],
    largest: int,
    cfg: CorpusLintConfig,
) -> list[Finding]:
    """Every component but the biggest one is, by definition, cut off.

    The largest is excluded rather than judged: something has to be the main
    mass, and calling it an orphan would make the report vacuous on a corpus
    that is simply small. `components` arrives largest-first, so the main
    mass is exactly the first entry - and only when it holds more than one
    atom, because a corpus of nothing but isolated atoms has no main mass to
    excuse.
    """
    findings: list[Finding] = []
    rest = components[1:] if largest > 1 else components
    for part in rest:
        if len(part) == 1:
            node = part[0]
            findings.append(
                Finding(
                    kind="isolated",
                    subject=node,
                    value=0.0,
                    nodes=part,
                    message=ISOLATED_TEMPLATE.format(label=node),
                )
            )
            continue
        if len(part) < cfg.min_orphan_nodes:
            continue
        label = _label(graph, part[0])
        findings.append(
            Finding(
                kind="orphan",
                subject=part[0],
                value=float(len(part)),
                nodes=part[: cfg.max_nodes_per_finding],
                message=ORPHAN_TEMPLATE.format(count=len(part), label=label),
            )
        )
    findings.sort(key=lambda f: (-f.value, f.subject))
    orphans = [f for f in findings if f.kind == "orphan"][: cfg.max_per_kind]
    isolated = [f for f in findings if f.kind == "isolated"][: cfg.max_per_kind]
    return orphans + isolated


def _hubs(graph: Graph, cfg: CorpusLintConfig) -> list[Finding]:
    """Known-risk #2, measured rather than asserted.

    The share is computed with the propagation's own split rule -
    `w**alpha / sum(w**alpha)` - so the number reported is the share energy
    would ACTUALLY take, not a proxy for it. A hub is only a problem when
    that share is small, which is why a high degree alone is not reported:
    eight hundred neighbours behind one dominant edge waste nothing.
    """
    findings: list[Finding] = []
    for node, targets in graph.adjacency.items():
        weights = [weight for weight in targets.values() if weight > 0.0]
        if len(weights) < cfg.min_hub_degree:
            continue
        shares = [weight**cfg.split_alpha for weight in weights]
        total = sum(shares)
        if total <= 0.0:
            continue
        best = max(shares) / total
        if best > cfg.hub_share_floor:
            continue
        findings.append(
            Finding(
                kind="hub",
                subject=node,
                value=best,
                nodes=(node,),
                message=HUB_TEMPLATE.format(
                    label=_label(graph, node), degree=len(weights), share=best
                ),
            )
        )
    findings.sort(key=lambda f: (f.value, f.subject))
    return findings[: cfg.max_per_kind]


def _duplicates(
    graph: Graph,
    semantic_edges: Iterable[tuple[str, str, float]],
    cfg: CorpusLintConfig,
) -> list[Finding]:
    """Near-identical passages, from the RAW cosine layer.

    Two findings, not one. The pair is what a person checks by eye; the
    per-source count is what actually costs something, because votes are
    counted per document (D7) and repetition inside a single source never
    becomes corroboration - it just burns seed slots.
    """
    pairs: list[Finding] = []
    per_source: dict[str, int] = {}
    for source, target, weight in semantic_edges:
        if source == target or weight < cfg.duplicate_weight:
            continue
        first, second = sorted((source, target))
        pairs.append(
            Finding(
                kind="duplicate",
                subject=first,
                value=float(weight),
                nodes=(first, second),
                message=DUPLICATE_TEMPLATE.format(
                    label=_label(graph, first),
                    other=_label(graph, second),
                    weight=weight,
                ),
            )
        )
        owner_a, owner_b = _source_of(graph, first), _source_of(graph, second)
        if owner_a == owner_b:
            per_source[owner_a] = per_source.get(owner_a, 0) + 1
    pairs.sort(key=lambda f: (-f.value, f.subject))

    sources = [
        Finding(
            kind="duplicate_source",
            subject=owner,
            value=float(count),
            nodes=(),
            message=DUPLICATE_SOURCE_TEMPLATE.format(label=owner, count=count),
        )
        for owner, count in per_source.items()
        if count >= cfg.min_duplicates_per_source
    ]
    sources.sort(key=lambda f: (-f.value, f.subject))
    return pairs[: cfg.max_per_kind] + sources[: cfg.max_per_kind]


def _contradictions(
    negative_edges: Iterable[NegativeEdge],
    graph: Graph,
    cfg: CorpusLintConfig,
) -> list[Finding]:
    """The contradiction map: marked pairs, aggregated by source.

    Per source and not per pair, for the same reason votes are: a document
    that disagrees with the corpus twenty times is one problem, not twenty.
    """
    pairs: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for edge in negative_edges:
        owner_a = _source_of(graph, edge.source)
        owner_b = _source_of(graph, edge.target)
        if owner_a == owner_b:
            continue
        for owner, other in ((owner_a, owner_b), (owner_b, owner_a)):
            pairs.setdefault(owner, set()).add(other)
            counts[owner] = counts.get(owner, 0) + 1
    findings = [
        Finding(
            kind="contradiction",
            subject=owner,
            value=float(counts[owner]),
            nodes=tuple(sorted(others))[: cfg.max_nodes_per_finding],
            message=CONTRADICTION_TEMPLATE.format(
                label=owner, others=len(others), count=counts[owner]
            ),
        )
        for owner, others in pairs.items()
        if counts[owner] >= cfg.min_contradictions_per_source
    ]
    findings.sort(key=lambda f: (-f.value, f.subject))
    return findings[: cfg.max_per_kind]


def _empty_layers(
    graph: Graph,
    layer_edges: Mapping[str, int] | None,
    weights: LayerWeights | None,
) -> list[Finding]:
    """A layer merged at a real weight that carries nothing.

    Not a threshold and not a heuristic: zero edges is zero edges. What
    makes this worth a finding rather than a log line is that BOTH known
    instances went unnoticed for months - the number came out, the layer was
    configured, and nothing connected the two facts.

    Where the cause is knowable from the corpus itself, it is named. A
    corpus of single-chunk documents cannot have within-document relations,
    and saying so turns "your structural weight does nothing" from an
    accusation into a description.
    """
    if not layer_edges:
        return []
    merge = weights if weights is not None else LayerWeights()
    per_source: dict[str, int] = {}
    for data in graph.node_data.values():
        per_source[data.source_id] = per_source.get(data.source_id, 0) + 1
    single_chunk = bool(per_source) and max(per_source.values()) == 1

    findings: list[Finding] = []
    for layer in sorted(layer_edges):
        if layer_edges[layer] != 0:
            continue
        try:
            weight = merge.weight_of(layer)  # type: ignore[arg-type]
        except AttributeError:  # pragma: no cover - an unknown layer name
            continue
        if weight <= 0.0:
            continue  # switched off on purpose; an empty layer is expected
        because = ""
        if layer == "structural" and single_chunk:
            because = SINGLE_CHUNK_REASON
        elif layer == "entity" and len(graph.node_data) < _SMALL_CORPUS:
            because = SMALL_CORPUS_REASON
        findings.append(
            Finding(
                kind="empty_layer",
                subject=layer,
                value=weight,
                nodes=(),
                message=EMPTY_LAYER_TEMPLATE.format(
                    layer=layer, weight=weight, because=because
                ),
            )
        )
    return findings


def _source_of(graph: Graph, node: str) -> str:
    data = graph.node(node)
    return data.source_id if data is not None else node


def _label(graph: Graph, node: str) -> str:
    """`source (atom)` where the graph knows the source, else the atom."""
    owner = _source_of(graph, node)
    return node if owner == node else f"{owner} ({node})"


def source_summary(report: LintReport) -> dict[str, int]:
    """Findings per source id - the roll-up a corpus owner acts on.

    Not a `LintReport` field: it is one of several possible roll-ups, and
    baking one into the report would make the others look secondary.
    """
    totals: dict[str, int] = {}
    for finding in report.findings:
        totals[finding.subject] = totals.get(finding.subject, 0) + 1
    return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))
