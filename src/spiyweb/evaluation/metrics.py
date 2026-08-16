"""The metrics of the Phase 1 objective - pure functions, hand-checkable.

Every metric is a recall against the question's gold supporting set, so all
of them live on one [0, 1] scale. That shared denominator is the entire
normalisation rule behind the weighted objective (closed open question #1):
`S@k = accuracy_weight * support_recall@k + novelty_weight * novelty@k`
combines two commensurable numbers, so the 65/35 split is real, not nominal.

Novelty@k operationalises serendipity with zero extra annotation (closed open
question #8): a gold paragraph counts as novel for a system when it appears
in that system's top-k while the plain dense top-k - the fixed reference -
does not return it AT ALL. The reference's whole top-k is subtracted, not
just its gold hits: a document the baseline already put in front of the
reader is not novel, however it was ranked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spiyweb.config import EvaluationConfig

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence


def _require_positive_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def passages_at_k(retrieved: Sequence[str], k: int) -> list[str]:
    """The first `k` distinct PASSAGES of a ranking, propositions folded in.

    Gold is annotated per passage, but a two-layer index ranks proposition
    ids (`d00042:0#p3`) alongside chunk ids. Intersecting those with gold
    directly can only ever miss: the score comes out near zero and looks like
    a bad retriever rather than a unit mismatch. Folding a proposition into
    its parent asks the question the gold can answer - "did the passage
    surface?" - and de-duplicating means three propositions from one passage
    occupy one slot, not three.

    A chunk-only ranking passes through unchanged, so every existing number
    is untouched.
    """
    _require_positive_k(k)
    seen: list[str] = []
    for node in retrieved:
        parent = node.split("#", 1)[0]
        if parent not in seen:
            seen.append(parent)
        if len(seen) == k:
            break
    return seen


def nodes_for_k_passages(ranked: Sequence[str], k: int) -> list[str]:
    """The prefix of `ranked` that carries `k` distinct passages.

    `passages_at_k` was written to fold propositions into their parents and
    take the first k DISTINCT passages - but it can only fold what it is
    given, and the harness stores the first `max_k` NODES. On a two-layer
    index those are not the same thing: measured 2026-08-16 on
    `musique_prop200`, ten stored nodes carried **3.81** distinct passages on
    average (5.00 on the chunk-only control) and only 28.5% of queries
    reached five. The metric was silently scoring a handicapped ranking.

    This returns a PREFIX, never a filtered list: nothing is reordered or
    dropped, so anything reading `ranking[:k]` raw sees exactly what it saw
    before. When the ranking cannot supply `k` passages it is returned whole -
    a short web is a finding, not an error.
    """
    _require_positive_k(k)
    seen: set[str] = set()
    for position, node in enumerate(ranked):
        seen.add(node.split("#", 1)[0])
        if len(seen) == k:
            return list(ranked[: position + 1])
    return list(ranked)


def _require_gold(gold: Collection[str], name: str) -> None:
    if not gold:
        raise ValueError(
            f"{name} must not be empty - a MuSiQue question always has "
            "supports, so an empty set means the loader mapping broke"
        )


def support_recall_at_k(
    retrieved: Sequence[str], gold: Collection[str], k: int
) -> float:
    """Fraction of the gold supporting set present in the top-k."""
    _require_positive_k(k)
    _require_gold(gold, "gold")
    hits = set(passages_at_k(retrieved, k)) & set(gold)
    return len(hits) / len(set(gold))


def novelty_at_k(
    retrieved: Sequence[str],
    reference: Sequence[str],
    gold: Collection[str],
    k: int,
) -> float:
    """Fraction of gold found in the top-k that the reference top-k lacks.

    `reference` is the plain dense top-k at the same cutoff. By construction
    the reference's own novelty against itself is 0.0.
    """
    _require_positive_k(k)
    _require_gold(gold, "gold")
    novel = (set(passages_at_k(retrieved, k)) & set(gold)) - set(
        passages_at_k(reference, k)
    )
    return len(novel) / len(set(gold))


def bridge_recall_at_k(
    retrieved: Sequence[str], bridge_gold: Collection[str], k: int
) -> float:
    """Recall over the intermediate documents only - the claim's own metric.

    Standard recall@k rewards finding ANY gold; this one asks specifically
    for the documents of every decomposition step but the last, which is
    where a multi-hop answer actually lives.
    """
    _require_positive_k(k)
    _require_gold(bridge_gold, "bridge_gold")
    hits = set(passages_at_k(retrieved, k)) & set(bridge_gold)
    return len(hits) / len(set(bridge_gold))


def weighted_objective(
    recall: float, novelty: float, config: EvaluationConfig | None = None
) -> float:
    """The single gate number: `S@k` for one system at one cutoff."""
    cfg = config if config is not None else EvaluationConfig()
    for name, value in (("recall", recall), ("novelty", novelty)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}={value!r} must lie in [0, 1]")
    return cfg.accuracy_weight * recall + cfg.novelty_weight * novelty
