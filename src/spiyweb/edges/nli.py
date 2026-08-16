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
chunk pairs work but with the documented blur. `shared_subject_pairs` is the
second half of that selection and lives here because it is a property of the
pair, not of the index format.
"""

from __future__ import annotations

from collections import Counter
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


def shared_subject_pairs(
    candidates: Sequence[tuple[str, str]],
    texts: Mapping[str, str],
    entities: Mapping[str, Sequence[str]],
    prefix_chars: int,
    max_df_ratio: float = 1.0,
) -> list[tuple[str, str]]:
    """Keep the candidate pairs whose two texts name the SAME subject.

    An NLI model reads a pair as premise and hypothesis about one thing. When
    the pair is really two things of the same kind - two radio stations, two
    villages, two high schools - it reports a contradiction that exists only
    in its own assumption. That was the single error pattern behind all
    fifteen strongest "contradictions" of the 2026-08-16 audit, and no
    threshold separates it, because those false positives score higher than
    the genuine contradiction found in the same run.

    The test has two parts. The shared entity must appear in the leading
    `prefix_chars` characters of BOTH texts - sharing a mention is not enough,
    since two stations in different states both mention "Jackson", so the
    shared name must sit where the subject sits. And it must be RARE enough
    (`max_df_ratio`, document frequency counted over `entities` itself): a
    name hundreds of passages carry marks a category, which is how two Calgary
    stations stayed paired on the word "Canadian". Matching is
    case-insensitive and substring-based, and a text is compared against ITS
    OWN entity list, so only a name the extractor actually found there can
    qualify it.

    Measured on the audit's recorded edge set (5.418 pairs at cut .90):
    window 40 alone keeps 655, the .005 rarity cut takes it to 393. The
    residual leak is documented and NOT solved here - two members of one
    family ("Scion Fuse" vs "Scion bbX") share a rare name and survive.

    Order is preserved, so a caller that sorted candidates by similarity
    still holds a sorted list afterwards. A node with no entities can never
    qualify, and neither can a node absent from `texts` - both mean the pair
    cannot be judged, and judging it anyway is what this function exists to
    prevent.
    """
    prefixes = {
        node_id: text[:prefix_chars].casefold() for node_id, text in texts.items()
    }
    document_frequency: Counter[str] = Counter()
    for found in entities.values():
        document_frequency.update({name.casefold() for name in found if name})
    # An empty registry would make every ratio a division by zero; it also
    # means nothing can qualify, which is the same answer.
    corpus_size = len(entities) or 1
    subjects: dict[str, set[str]] = {
        node_id: {
            name.casefold()
            for name in found
            if name
            and name.casefold() in prefixes.get(node_id, "")
            and document_frequency[name.casefold()] / corpus_size <= max_df_ratio
        }
        for node_id, found in entities.items()
    }
    return [
        (id_a, id_b)
        for id_a, id_b in candidates
        if subjects.get(id_a, set()) & subjects.get(id_b, set())
    ]


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
