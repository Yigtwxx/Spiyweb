"""Drawing a recorded call, using the same scene builder the live rig uses.

`scene.py` was promoted into the package in Faz 2.2 for exactly this: two
front ends, one picture. The live inspector feeds it a `RetrievalResult`, a
merged graph and a layer index. A trace has none of those objects - it has
what they produced. So this module rebuilds the small amount `build_scene`
needs FROM THE RECORD and hands it over, rather than growing a second
layout, a second ranking and a second set of tooltips that drift.

Two things a record genuinely cannot give back, stated rather than faked:

- **Edge layers.** The merged adjacency forgets which layer a weight came
  from, so the record does too and the scene draws every edge as `merged`.
  Colouring them would mean inventing the answer.
- **Contributor edges.** The core records contributors, but a record keeps
  the strongest chain per node (`paths`), not every arrival. So the
  contributor view is rebuilt from those chains, which is the same picture
  wherever a node had one feeder and a subset of it where it had several.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spiyweb.scene import GraphScene, ViewConfig
    from spiyweb.trace import TraceRecord

__all__ = ["scene_of", "scene_payload"]

MERGED_LAYER = "merged"
"""What a recorded edge's layer is called. Not `semantic` and not a guess."""


def scene_of(record: TraceRecord, view: ViewConfig | None = None) -> GraphScene:
    """Lay out one recorded call for the canvas."""
    from spiyweb.config import LayerWeights
    from spiyweb.core.graph import Graph, Node
    from spiyweb.core.propagate import Activation
    from spiyweb.scene import EdgeLayerIndex, build_scene

    contributors = {path.node: path.steps for path in record.paths}
    activations = {
        node.id: Activation(
            energy=node.energy,
            hop=node.hop,
            # The step before this one in its strongest chain, when there was
            # one. `build_scene` only ever reads this in contributor mode.
            contributors=_feeder(contributors.get(node.id, ())),
        )
        for node in record.nodes
        if node.suppressed_by == ""
    }
    nodes = tuple(
        Node(
            id=node.id,
            layer="proposition" if node.layer == "proposition" else "chunk",
            source_id=node.source_id or node.id,
            # Length is the raw input to the mass formula, which has already
            # run by the time a trace exists; the scene never reads it, so
            # the recorded text length is an honest stand-in for a value the
            # record does not carry.
            length=max(len(node.text), 1),
            polarity=-1 if node.polarity == -1 else 1,
        )
        for node in record.nodes
    )
    graph = Graph.from_edges(
        [(edge.source, edge.target, edge.weight) for edge in record.edges],
        nodes=nodes,
    )
    return build_scene(
        activations=activations,
        seeds={
            node.id: node.seed_similarity
            for node in record.nodes
            if node.seed_similarity is not None
        },
        votes={node.id: node.votes for node in record.nodes},
        suppressed={
            node.id: node.suppressed_by for node in record.nodes if node.suppressed_by
        },
        graph=graph,
        layer_index=EdgeLayerIndex(
            node_ids=(), position={}, codes={}, layer_weights={}, counts={}
        ),
        weights=LayerWeights(),
        view=view,
        salt=record.trace_id,
        texts={node.id: node.text for node in record.nodes},
        bridges=tuple(record.bridges),
        disputed=tuple(node.id for node in record.nodes if node.disputed),
    )


def _feeder(steps: tuple[str, ...]) -> tuple[str, ...]:
    """The step that fed the last one, or nothing for a seed."""
    return (steps[-2],) if len(steps) >= 2 else ()


def scene_payload(record: TraceRecord, view: ViewConfig | None = None) -> dict:
    """The scene in the shape the canvas already draws.

    Identical keys to the live inspector's, because both go through
    `payload.scene_payload_of`. That sharing is the point: the browser has
    one canvas component and must not learn a second dialect for recorded
    calls.
    """
    from spiyweb.scene import EDGE_LAYER_ORDER, hop_ring_layout
    from spiyweb.viewer.payload import scene_payload_of

    scene = scene_of(record, view)
    rings = hop_ring_layout(
        [node.id for node in scene.nodes],
        {node.id: max(node.hop, 0) for node in scene.nodes},
        {node.id: node.energy for node in scene.nodes},
        salt=record.trace_id,
    )
    return scene_payload_of(scene, rings, layer_order=EDGE_LAYER_ORDER)
