"""Paired bootstrap over per-query records - the protocol's interval.

The measurement protocol requires an interval, never a bare point estimate,
and `aggregate()` only produces point estimates. This module closes that gap
inside `evaluation/` rather than in a scratch script, because three consumers
need it (the terminal, the Streamlit tool and the browser server) and a
statistic copied three times is a statistic that will disagree with itself.

Resampling is over QUESTIONS and paired: every system sees the same resample,
which is what makes the interval on a DIFFERENCE honest. The per-question
objective is `0.65 * recall + 0.35 * novelty`; because the objective is linear
in both terms, its mean equals the harness's
`weighted_objective(mean(recall), mean(novelty))`, so these numbers reproduce
`results.json` exactly rather than approximating it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from spiyweb.config import EvaluationConfig
from spiyweb.evaluation.metrics import (
    bridge_recall_at_k as metrics_bridge_recall_at_k,
)
from spiyweb.evaluation.metrics import (
    novelty_at_k,
    support_recall_at_k,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_SEED = 20260816
"""Fixed so two people reading the same run get the same interval."""


@dataclass(frozen=True)
class PairedDifference:
    """One system against one rival, with the interval around the gap."""

    rival: str
    mean: float
    ci_low: float
    ci_high: float
    p: float
    significant: bool


@dataclass(frozen=True)
class HopRow:
    """Per-hop breakdown - where a multi-hop claim lives or dies."""

    hops: int
    questions: int
    scores: dict[str, float]


@dataclass(frozen=True)
class BootstrapReport:
    """Everything a report needs to state a difference honestly."""

    k: int
    iterations: int
    seed: int
    questions: int
    means: dict[str, float]
    bridge: dict[str, float]
    diffs: tuple[PairedDifference, ...]
    by_hop: tuple[HopRow, ...]


def objective_at_k(
    record: Mapping[str, object], system: str, k: int, config: EvaluationConfig
) -> float:
    """One question's S@k for one system, through `evaluation/metrics.py`.

    This used to be a hand-inlined copy of the formula, and the copy was
    WRONG on a two-layer index: it intersected raw node ids with gold, so a
    proposition (`d00042:0#p3`) could never match a passage-level gold label
    and the score came out near zero - the same silent-zero bug the harness
    already had and fixed. Found 2026-08-16 while diagnosing the coloured
    proposition loss: this module reported prop-vs-chunk as -.1232 where the
    folded metric says -.0524. Delegating is the fix AND the guarantee that
    it cannot drift again.
    """
    recall = support_recall_at_k(record[system], record["gold"], k)  # type: ignore[arg-type,index]
    novelty = novelty_at_k(record[system], record["topk"], record["gold"], k)  # type: ignore[arg-type,index]
    return config.accuracy_weight * recall + config.novelty_weight * novelty


def bridge_recall_at_k(record: Mapping[str, object], system: str, k: int) -> float:
    """Bridge recall for one record - folded, for the same reason as above."""
    return metrics_bridge_recall_at_k(record[system], record["bridge_gold"], k)  # type: ignore[arg-type,index]


def paired_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    iterations: int = 10000,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float, float]:
    """`(mean difference, low, high, two-sided p)` for `a - b`, paired.

    The p value is the bootstrap's own: how often a resampled mean lands on
    the other side of zero, doubled. It is a companion to the interval, not a
    replacement for reading it.
    """
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if difference.size == 0:
        return 0.0, 0.0, 0.0, 1.0
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, difference.size, size=(iterations, difference.size))
    means = difference[draws].mean(axis=1)
    observed = float(difference.mean())
    low, high = (float(value) for value in np.percentile(means, [2.5, 97.5]))
    crossings = float(np.mean(means <= 0.0) if observed > 0 else np.mean(means >= 0.0))
    return observed, low, high, min(1.0, 2.0 * crossings)


def bootstrap_report(
    records: Sequence[Mapping[str, object]],
    *,
    k: int = 5,
    iterations: int = 10000,
    seed: int = DEFAULT_SEED,
    config: EvaluationConfig | None = None,
) -> BootstrapReport:
    """Means, paired differences against every rival, and the hop breakdown."""
    cfg = config if config is not None else EvaluationConfig()
    if not records:
        raise ValueError("no per-query records to analyse")

    systems = ["topk", "web"]
    if all(record.get("iterative") is not None for record in records):
        systems.append("iterative")

    scores = {
        system: np.array(
            [objective_at_k(record, system, k, cfg) for record in records],
            dtype=np.float64,
        )
        for system in systems
    }
    bridges = {
        system: float(
            np.mean([bridge_recall_at_k(record, system, k) for record in records])
        )
        for system in systems
    }

    diffs: list[PairedDifference] = []
    for rival in (system for system in systems if system != "web"):
        mean, low, high, p = paired_ci(
            scores["web"], scores[rival], iterations=iterations, seed=seed
        )
        diffs.append(
            PairedDifference(
                rival=rival,
                mean=mean,
                ci_low=low,
                ci_high=high,
                p=p,
                significant=low > 0.0 or high < 0.0,
            )
        )

    by_hop: list[HopRow] = []
    for hops in sorted({int(record["hops"]) for record in records}):  # type: ignore[arg-type]
        rows = [
            index
            for index, record in enumerate(records)
            if int(record["hops"]) == hops  # type: ignore[arg-type]
        ]
        by_hop.append(
            HopRow(
                hops=hops,
                questions=len(rows),
                scores={
                    system: float(scores[system][rows].mean()) for system in systems
                },
            )
        )

    return BootstrapReport(
        k=k,
        iterations=iterations,
        seed=seed,
        questions=len(records),
        means={system: float(scores[system].mean()) for system in systems},
        bridge=bridges,
        diffs=tuple(diffs),
        by_hop=tuple(by_hop),
    )
