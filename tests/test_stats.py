"""Paired-bootstrap statistics - the interval the protocol requires.

The claim under test: `stats` scores a record through `evaluation/metrics.py`
rather than re-deriving the formula, so a two-layer ranking is folded into
passages exactly once and in one place. It used to intersect raw node ids
with passage-level gold, which on a proposition index scored a perfect
ranking as a near-miss: the module reported the coloured proposition loss as
-.1232 where the folded metric says -.0524. Three consumers read this module
(terminal, Streamlit tool, browser server), so the copy was wrong in three
places at once.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from spiyweb.config import EvaluationConfig
from spiyweb.evaluation.stats import bridge_recall_at_k, objective_at_k, paired_ci

CONFIG = EvaluationConfig()

# Five ranked NODES carrying two distinct passages, both of them gold.
TWO_LAYER = {
    "id": "q1",
    "gold": ["d0:0", "d1:0"],
    "bridge_gold": ["d0:0"],
    "topk": ["d2:0", "d3:0", "d4:0", "d5:0", "d6:0"],
    "web": ["d0:0#p1", "d0:0#p2", "d1:0#p0", "d0:0#p3", "d1:0#p4"],
}


def test_proposition_ids_are_folded_into_their_passage() -> None:
    score = objective_at_k(TWO_LAYER, "web", 5, CONFIG)
    assert score == pytest.approx(1.0), (
        "both gold passages are in the ranking and neither is in the dense "
        "reference, so recall and novelty are both 1 - scoring this near "
        "zero is the silent-zero bug"
    )


def test_bridge_recall_is_folded_too() -> None:
    assert bridge_recall_at_k(TWO_LAYER, "web", 5) == pytest.approx(1.0)


def test_a_chunk_only_record_is_unchanged() -> None:
    record = {
        "gold": ["d0:0", "d1:0"],
        "bridge_gold": ["d0:0"],
        "topk": ["d0:0", "d9:0"],
        "web": ["d0:0", "d1:0", "d9:0"],
    }
    # recall 1.0; novelty .5 - d1:0 is the only hit the dense list missed.
    expected = CONFIG.accuracy_weight * 1.0 + CONFIG.novelty_weight * 0.5
    assert objective_at_k(record, "web", 5, CONFIG) == pytest.approx(expected), (
        "every sealed number came from chunk-only runs; the fix must leave "
        "them bit-identical"
    )


def test_paired_ci_is_deterministic_and_paired() -> None:
    import numpy as np

    a = np.array([0.5, 0.6, 0.7, 0.8])
    b = np.array([0.4, 0.5, 0.6, 0.7])
    first = paired_ci(a, b)
    second = paired_ci(a, b)
    assert first == second, "a fixed seed must give two readers the same interval"
    assert first[0] == pytest.approx(0.1), "the mean difference is exact, not sampled"
    assert first[1] <= first[0] <= first[2]
