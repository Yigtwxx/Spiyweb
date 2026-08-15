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
    hits = set(retrieved[:k]) & set(gold)
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
    novel = (set(retrieved[:k]) & set(gold)) - set(reference[:k])
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
    hits = set(retrieved[:k]) & set(bridge_gold)
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
