"""Index-time NLI contradiction marking - emits negative edges (D26).

A small multilingual NLI model runs at INDEX time over candidate pairs and
marks the contradicting ones; `core/` only ever consumes the pre-marked
result. There is no NLI at query time, and no model call in this module
either: the model arrives injected behind a Protocol, exactly like the spaCy
pipeline and the LLM client elsewhere, so the builder stays pure and the
model choice (open question #10) stays open.

Candidate selection is the CALLER's job: the design targets high-similarity
PROPOSITION pairs (contradiction is blurry on chunks, sharp on propositions -
see memory/contradiction-detection.md) - until the proposition layer lands,
chunk pairs work but with the documented blur.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from spiyweb.config import NLIEdgeConfig
from spiyweb.core.conflict import NegativeEdge

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class NLIModel(Protocol):
    """Minimal NLI interface the edge builder depends on.

    One score per (premise, hypothesis) pair: the model's confidence that the
    hypothesis CONTRADICTS the premise, in [0, 1]. Batch-shaped so a real
    transformer wrapper can batch internally.
    """

    def contradiction_scores(
        self, pairs: Sequence[tuple[str, str]]
    ) -> Sequence[float]: ...


def build_nli_edges(
    candidates: Sequence[tuple[str, str]],
    texts: Mapping[str, str],
    model: NLIModel,
    config: NLIEdgeConfig | None = None,
) -> list[NegativeEdge]:
    """Score candidate node pairs and emit negative edges for contradictions.

    NLI is directional, so each pair is scored BOTH ways and the strength is
    the maximum - a contradiction found in either direction marks the pair.
    A pair reaches the output only when that strength passes the configured
    threshold (inclusive); output order follows the candidate order, with
    each pair's ids in sorted order.

    Raises:
        ValueError: On a candidate id without text, or a self-pair - both
            mean the candidate generator broke, and silence here would
            surface later as inexplicably missing (or absurd) conflicts.
    """
    cfg = config if config is not None else NLIEdgeConfig()
    for id_a, id_b in candidates:
        if id_a == id_b:
            raise ValueError(f"candidate self-pair on {id_a!r}")
        for node_id in (id_a, id_b):
            if node_id not in texts:
                raise ValueError(f"candidate id {node_id!r} has no text")

    directed: list[tuple[str, str]] = []
    for id_a, id_b in candidates:
        directed.append((texts[id_a], texts[id_b]))
        directed.append((texts[id_b], texts[id_a]))
    scores = list(model.contradiction_scores(directed))
    if len(scores) != len(directed):
        raise ValueError(
            f"model returned {len(scores)} scores for {len(directed)} pairs"
        )

    edges: list[NegativeEdge] = []
    for index, (id_a, id_b) in enumerate(candidates):
        strength = max(scores[2 * index], scores[2 * index + 1])
        if strength >= cfg.contradiction_threshold:
            source, target = sorted((id_a, id_b))
            edges.append(NegativeEdge(source=source, target=target, strength=strength))
    return edges
