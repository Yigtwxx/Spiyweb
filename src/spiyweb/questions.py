"""Template-built, LLM-free user questions for surfaced conflicts (D16).

The library never silently picks a winner: when a contradiction survives into
the result, it ships a ready-made question with options so every caller does
not rewrite the same one - and when no answer comes back, both sides enter the
context flagged as disputed (the third option is exactly that outcome).

Templates live here, OUTSIDE `core/`, by design: the core only produces
structured conflict data (`ConflictRecord`); wording is presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spiyweb.core.conflict import ConflictRecord

CONFLICT_QUESTION_TEMPLATE = (
    "Two sources make opposing claims:\n"
    "  (A) {label_a}\n"
    "  (B) {label_b}\n"
    "Which should the answer rely on?"
)

CONFLICT_OPTION_TEMPLATES = (
    "Rely on (A) {label_a}",
    "Rely on (B) {label_b}",
    "Keep both, marked as disputed",
)

KEEP_BOTH_OPTION_INDEX = 2
"""Index of the no-decision option - the D16 default when no answer comes."""


@dataclass(frozen=True)
class ConflictQuestion:
    """One ready-made user question about one surfaced contradiction.

    Attributes:
        text: The question itself, template-built, never from an LLM.
        options: Answer options in order; the last one is always the
            "keep both, disputed" outcome that also applies when the caller
            gets no answer at all.
        record: The structured conflict datum the question was built from,
            so the caller can map the chosen option back to node ids.
    """

    text: str
    options: tuple[str, ...]
    record: ConflictRecord


def build_conflict_question(
    record: ConflictRecord, labels: Mapping[str, str] | None = None
) -> ConflictQuestion:
    """Build the D16 question for one conflict.

    `labels` maps node ids to human-readable names (titles, in practice);
    a node without a label falls back to its id - never an empty string.
    """
    label_of = labels if labels is not None else {}
    label_a = label_of.get(record.node_a, record.node_a)
    label_b = label_of.get(record.node_b, record.node_b)
    return ConflictQuestion(
        text=CONFLICT_QUESTION_TEMPLATE.format(label_a=label_a, label_b=label_b),
        options=tuple(
            template.format(label_a=label_a, label_b=label_b)
            for template in CONFLICT_OPTION_TEMPLATES
        ),
        record=record,
    )
