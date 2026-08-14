"""The honesty outputs of the §2.5 contract: paths, themes, gaps, refusal.

These are the outputs `top-k` structurally cannot produce, and with the vote
mechanism they are where this project differs from other graph-RAG work:

- **Activation paths** (D19): every node carries the chain that lit it, and
  the chain goes TO THE LLM as an explanation, not just to a debug log.
- **Theme clusters** (D20): connected components of the activated subgraph,
  computed at query time (owner's choice, 2026-08-14) - "these are separate
  themes, they intersect here".
- **Gap warnings** (D18): two dense clusters with no connection between them
  mean the corpus lacks the content that would bridge them - a diagnosis of
  the knowledge base, free, because the split is already in the web's shape.
- **Refusal report** (D35): a template-built, LLM-free account of WHY a weak
  result is weak - which clusters lit, where the bridge is missing, where the
  energy died, what kind of source would close the gap. The library builds
  the report on demand; WHEN confidence counts as low stays the caller's
  policy (D17), so the ablation switch is simply not calling.

Everything here is a pure function over `core` results - no I/O, no model
calls; the optional `edge_labels` mapping (e.g. "shared entity 'X'") comes
from the caller, because the merged adjacency deliberately forgets which
layer a weight came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiyweb.config import OutputConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from spiyweb.core.colors import ColoredResult
    from spiyweb.core.graph import Graph
    from spiyweb.core.polarity import DisputeRecord
    from spiyweb.core.propagate import PropagationResult

REFUSAL_TEMPLATE = (
    "The web activated {cluster_count} theme cluster(s) over {node_count} "
    "node(s) with {total_energy:.2f} total energy, reaching hop {hop_depth}; "
    "the spread stopped on '{stop_reason}'.\n"
    "{gap_line}\n"
    "{missing_line}"
)

GAP_LINE_TEMPLATE = (
    "Gap: the clusters around {top_a} and {top_b} both lit up densely but "
    "share no connection."
)

NO_GAP_LINE = "No gap between dense clusters was detected."

MISSING_SOURCE_TEMPLATE = (
    "Missing: a source that connects {top_a} with {top_b} - nothing in the "
    "corpus bridges them."
)

MISSING_DEPTH_TEMPLATE = (
    "Missing: the energy died at hop {hop_depth} around {deepest} - a source "
    "linking that frontier onward may not exist in the corpus."
)

DISPUTE_TEMPLATE = (
    "The corpus disputes this claim: atom {node} states the opposite and "
    "absorbed {absorbed:.2f} of the query's energy at hop {hop}."
)


@dataclass(frozen=True)
class ActivationPath:
    """How one node was reached - the D19 explanation datum.

    `steps` runs seed-first and ends at `node`. When a node accumulated from
    several contributors, the path follows the STRONGEST one (by contributor
    energy; per-arrival shares are deliberately not recorded by the core) and
    `converging` reports how many fed it - converging evidence is the value
    proposition, so it stays visible even in the single rendered chain.
    """

    node: str
    steps: tuple[str, ...]
    hop: int
    energy: float
    converging: int

    def rendered(self, edge_labels: Mapping[tuple[str, str], str] | None = None) -> str:
        """`A -> shared entity 'X' -> D` when a label exists, else `A -> D`."""
        if len(self.steps) == 1:
            return self.steps[0]
        parts = [self.steps[0]]
        for source, target in zip(self.steps, self.steps[1:], strict=False):
            label = None
            if edge_labels is not None:
                label = edge_labels.get((source, target)) or edge_labels.get(
                    (target, source)
                )
            if label:
                parts.append(f"-> {label} ->")
            else:
                parts.append("->")
            parts.append(target)
        return " ".join(parts)


@dataclass(frozen=True)
class ThemeCluster:
    """One connected component of the activated subgraph (D20).

    Attributes:
        nodes: Member nodes, sorted.
        energy: Summed accumulated energy of the members.
        energy_share: `energy` over the run's total activated energy.
        top_node: The strongest member - the cluster's natural label.
        colors: Colours present in the cluster (colored runs only; empty in
            a plain run). Bridge nodes are where these sets met.
    """

    nodes: tuple[str, ...]
    energy: float
    energy_share: float
    top_node: str
    colors: tuple[str, ...] = ()


@dataclass(frozen=True)
class GapWarning:
    """Two dense clusters, no connection (D18) - a corpus diagnosis."""

    top_a: str
    top_b: str
    nodes_a: tuple[str, ...]
    nodes_b: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class DisputeWarning:
    """The corpus said "no" to the query's claim (D34) - a corpus verdict.

    Built from a `DisputeRecord`: the absorption having fired IS the
    evidence that a negative-polarity atom holds the opposite of what the
    query asserts. The message is template-built, never from an LLM; the
    caller resolves `node` to the atom's text when presenting it.
    """

    node: str
    hop: int
    absorbed: float
    message: str


@dataclass(frozen=True)
class RefusalReport:
    """The D35 structural refusal - why a weak result is weak, without an LLM.

    Built on demand by `build_refusal_report`; the caller decides when to
    show it (D17: the library ships signals, never the policy).
    """

    clusters: tuple[ThemeCluster, ...]
    gaps: tuple[GapWarning, ...]
    stop_reason: str
    hop_depth: int
    deepest_nodes: tuple[str, ...]
    text: str


def activation_paths(result: PropagationResult) -> tuple[ActivationPath, ...]:
    """One path per activated node, strongest node first.

    The chain walks `contributors` backwards, at each step following the
    contributor with the highest accumulated energy (ties break on id, as
    everywhere). Contributors are only recorded at first activation from an
    earlier hop, so the walk terminates at a seed by construction.
    """
    activations = result.activations

    def walk(node: str) -> tuple[str, ...]:
        chain = [node]
        seen = {node}
        current = node
        while True:
            feeders = [
                feeder
                for feeder in activations[current].contributors
                if feeder in activations and feeder not in seen
            ]
            if not feeders:
                return tuple(reversed(chain))
            current = max(feeders, key=lambda f: (activations[f].energy, f))
            chain.append(current)
            seen.add(current)

    paths = [
        ActivationPath(
            node=node,
            steps=walk(node),
            hop=activation.hop,
            energy=activation.energy,
            converging=len(activation.contributors),
        )
        for node, activation in activations.items()
    ]
    return tuple(sorted(paths, key=lambda path: (-path.energy, path.node)))


def entity_edge_labels(
    pairs: Iterable[tuple[str, str]],
    entities: Mapping[str, Sequence[str]],
) -> dict[tuple[str, str], str]:
    """Edge reason labels from shared entities, for `ActivationPath.rendered`.

    For every requested pair the label names the RAREST shared entity - rarity
    measured as document frequency over the supplied `entities` mapping, the
    same instinct as the entity layer's `1/df` edge weighting: the least
    common thing two passages share is the most informative reason they are
    connected. Ties break lexicographically, as everywhere.

    A pair with no shared entity gets NO entry (its edge came from another
    layer; `rendered` prints a plain arrow), and so do self-pairs and nodes
    the mapping does not know. Keys are stored as given - `rendered` already
    looks an edge up in both directions. Callers typically collect pairs from
    path steps: `zip(path.steps, path.steps[1:])`.
    """
    frequency: dict[str, int] = {}
    for found in entities.values():
        for entity in set(found):
            frequency[entity] = frequency.get(entity, 0) + 1

    labels: dict[tuple[str, str], str] = {}
    for source, target in pairs:
        if source == target or (source, target) in labels:
            continue
        shared = set(entities.get(source, ())) & set(entities.get(target, ()))
        if not shared:
            continue
        rarest = min(shared, key=lambda entity: (frequency[entity], entity))
        labels[(source, target)] = f"shared entity '{rarest}'"
    return labels


def theme_clusters(
    result: PropagationResult,
    graph: Graph,
    *,
    colors_of: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[ThemeCluster, ...]:
    """Connected components of the activated subgraph, strongest first.

    Only edges between two ACTIVATED nodes count - the theme structure is a
    property of what this query lit up, not of the whole corpus.
    """
    activations = result.activations
    total = sum(activation.energy for activation in activations.values())
    unvisited = set(activations)
    clusters: list[ThemeCluster] = []
    while unvisited:
        start = min(unvisited)
        component = {start}
        frontier = [start]
        while frontier:
            node = frontier.pop()
            for neighbor, weight in graph.neighbors(node).items():
                if weight > 0.0 and neighbor in unvisited and neighbor not in component:
                    component.add(neighbor)
                    frontier.append(neighbor)
        unvisited -= component
        energy = sum(activations[node].energy for node in component)
        top_node = max(component, key=lambda n: (activations[n].energy, n))
        colors: tuple[str, ...] = ()
        if colors_of is not None:
            colors = tuple(
                sorted(
                    {color for node in component for color in colors_of.get(node, ())}
                )
            )
        clusters.append(
            ThemeCluster(
                nodes=tuple(sorted(component)),
                energy=energy,
                energy_share=energy / total if total > 0.0 else 0.0,
                top_node=top_node,
                colors=colors,
            )
        )
    return tuple(sorted(clusters, key=lambda c: (-c.energy, c.top_node)))


def color_composition(colored: ColoredResult) -> dict[str, tuple[str, ...]]:
    """Node -> sorted colours that reached it, for `theme_clusters`."""
    reached: dict[str, set[str]] = {}
    for color, result in colored.per_color.items():
        for node in result.activations:
            reached.setdefault(node, set()).add(color)
    return {node: tuple(sorted(colors)) for node, colors in reached.items()}


def gap_warnings(
    clusters: tuple[ThemeCluster, ...], config: OutputConfig | None = None
) -> tuple[GapWarning, ...]:
    """One warning per pair of dense clusters (D18).

    Clusters are components, so any two are unconnected by construction -
    the density knobs are what separates a real gap from stray noise.
    """
    cfg = config if config is not None else OutputConfig()
    dense = [
        cluster
        for cluster in clusters
        if len(cluster.nodes) >= cfg.min_cluster_nodes
        and cluster.energy_share >= cfg.min_cluster_energy_share
    ]
    warnings: list[GapWarning] = []
    for i, cluster_a in enumerate(dense):
        for cluster_b in dense[i + 1 :]:
            warnings.append(
                GapWarning(
                    top_a=cluster_a.top_node,
                    top_b=cluster_b.top_node,
                    nodes_a=cluster_a.nodes,
                    nodes_b=cluster_b.nodes,
                    message=GAP_LINE_TEMPLATE.format(
                        top_a=cluster_a.top_node, top_b=cluster_b.top_node
                    ),
                )
            )
    return tuple(warnings)


def dispute_warnings(disputes: Iterable[DisputeRecord]) -> tuple[DisputeWarning, ...]:
    """One template-built warning per dispute record (D34), in firing order.

    Feed it `result.disputes` (plain or coloured). No LLM, no policy: the
    library reports that the corpus objected and how much energy it absorbed;
    whether and how to confront the user with it is the caller's call (D17).
    """
    return tuple(
        DisputeWarning(
            node=record.node,
            hop=record.hop,
            absorbed=record.absorbed,
            message=DISPUTE_TEMPLATE.format(
                node=record.node, absorbed=record.absorbed, hop=record.hop
            ),
        )
        for record in disputes
    )


def build_refusal_report(
    result: PropagationResult,
    graph: Graph,
    *,
    colors_of: Mapping[str, tuple[str, ...]] | None = None,
    config: OutputConfig | None = None,
) -> RefusalReport:
    """Assemble the D35 report from the run's own structure - no LLM.

    Callable on ANY result; the caller invokes it when THEIR confidence
    policy says the answer is weak (D17), and shows or drops it freely.
    """
    clusters = theme_clusters(result, graph, colors_of=colors_of)
    gaps = gap_warnings(clusters, config)
    deepest = tuple(
        sorted(
            node
            for node, activation in result.activations.items()
            if activation.hop == result.hops_used
        )
    )
    if gaps:
        gap_line = gaps[0].message
        missing_line = MISSING_SOURCE_TEMPLATE.format(
            top_a=gaps[0].top_a, top_b=gaps[0].top_b
        )
    else:
        gap_line = NO_GAP_LINE
        missing_line = MISSING_DEPTH_TEMPLATE.format(
            hop_depth=result.hops_used,
            deepest=", ".join(deepest[:3]) if deepest else "the seeds",
        )
    total = sum(activation.energy for activation in result.activations.values())
    text = REFUSAL_TEMPLATE.format(
        cluster_count=len(clusters),
        node_count=len(result.activations),
        total_energy=total,
        hop_depth=result.hops_used,
        stop_reason=result.stop_reason,
        gap_line=gap_line,
        missing_line=missing_line,
    )
    return RefusalReport(
        clusters=clusters,
        gaps=gaps,
        stop_reason=result.stop_reason,
        hop_depth=result.hops_used,
        deepest_nodes=deepest,
        text=text,
    )
