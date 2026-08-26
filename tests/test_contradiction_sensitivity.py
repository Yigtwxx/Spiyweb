"""The contradiction-sensitivity harness, and the bound it can honestly report.

Phase 2.8's second queue item. Phase 1 recorded 0% on same-passage
contradictions and wrote down the reason as "the proposition layer is
required". This harness exists to test that clause, and the tests below pin
what it can and cannot conclude - because the interesting result turned out
to be about the measuring instrument, not about the mechanism.

The model is injected everywhere, so nothing here downloads mDeBERTa.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spiyweb.evaluation.contradiction import (
    ContradictionCase,
    load_wikicontradict,
    measure_sensitivity,
    split_sentences,
    trace_answers,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DATA = "data/wikicontradict"


class FixedModel:
    """Scores every directed pair the same. The harness is under test here."""

    def __init__(self, score: float) -> None:
        self.score = score
        self.seen: list[tuple[str, str]] = []

    def contradiction_scores(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.seen.extend(pairs)
        return [self.score] * len(pairs)


def _case(**overrides: object) -> ContradictionCase:
    base = {
        "question_id": "1",
        "question": "q",
        "context_a": "Alice moved to Paris in 1990. Bob stayed in Rome.",
        "context_b": "Alice moved to Paris in 1990. Bob stayed in Rome.",
        "answer_a": "Alice moved to Paris",
        "answer_b": "Bob stayed in Rome",
        "same_passage": True,
        "kind": "Explicit",
        "title": "t",
    }
    base.update(overrides)
    return ContradictionCase(**base)  # type: ignore[arg-type]


# --- sentence splitting ----------------------------------------------------


def test_a_passage_splits_on_terminal_punctuation() -> None:
    assert split_sentences("One thing happened. Another thing happened.") == [
        "One thing happened.",
        "Another thing happened.",
    ]


def test_an_abbreviation_does_not_end_a_sentence() -> None:
    """A split here would invent a boundary and inflate the ceiling."""
    assert split_sentences("Dr. Smith arrived. He left.") == [
        "Dr. Smith arrived.",
        "He left.",
    ]


def test_punctuation_debris_is_merged_rather_than_dropped() -> None:
    """No text may be lost: a lost fragment is a claim nobody can find."""
    joined = " ".join(split_sentences("A real sentence here. Ok. And more text."))
    assert "Ok." in joined


def test_a_single_sentence_stays_single() -> None:
    assert len(split_sentences("Only one claim lives in this passage.")) == 1


# --- tracing ---------------------------------------------------------------


def test_two_answers_from_two_sentences_are_traced() -> None:
    found = trace_answers(
        "Alice moved to Paris in 1990. Bob stayed in Rome.",
        "Alice moved to Paris",
        "Bob stayed in Rome",
    )
    assert found == (0, 1)


def test_two_answers_from_one_sentence_are_not_traced() -> None:
    """The whole point of the bound: one sentence cannot be split."""
    assert (
        trace_answers(
            "Alice moved to Paris and Bob stayed in Rome.",
            "Alice moved to Paris",
            "Bob stayed in Rome",
        )
        is None
    )


def test_a_yes_no_answer_traces_to_nothing() -> None:
    """It is a reasoning output, not a span - and 30 of 63 cases are these."""
    assert trace_answers("Alice moved to Paris. Bob stayed.", "Yes", "No") is None


# --- the two buckets are gated differently ---------------------------------


def test_a_different_passage_case_needs_no_split() -> None:
    """The shipped pipeline already has two nodes there; the pair exists."""
    model = FixedModel(1.0)
    report = measure_sensitivity(
        [
            _case(
                same_passage=False,
                context_a="The tower fell in 1917.",
                context_b="The tower stood until 1930.",
                answer_a="1917",
                answer_b="1930",
            )
        ],
        model,
        require_shared_subject=False,
    )
    assert report.different_passage.ceiling == 1
    assert report.different_passage.detected == 1
    assert ("The tower fell in 1917.", "The tower stood until 1930.") in model.seen


def test_a_same_passage_case_is_gated_by_the_split() -> None:
    model = FixedModel(1.0)
    report = measure_sensitivity([_case()], model, require_shared_subject=False)
    assert report.same_passage.ceiling == 1
    assert report.same_passage.detected == 1


def test_an_unsplittable_same_passage_case_never_reaches_the_model() -> None:
    """Nothing downstream can exceed the ceiling, so nothing is scored."""
    model = FixedModel(1.0)
    report = measure_sensitivity(
        [
            _case(
                context_a="Alice moved to Paris and Bob stayed in Rome.",
                context_b="Alice moved to Paris and Bob stayed in Rome.",
            )
        ],
        model,
        require_shared_subject=False,
    )
    assert report.same_passage.ceiling == 0
    assert report.same_passage.detected == 0
    assert model.seen == []


def test_a_score_below_the_shipped_threshold_is_not_a_detection() -> None:
    report = measure_sensitivity(
        [_case()], FixedModel(0.5), require_shared_subject=False
    )
    assert report.same_passage.ceiling == 1
    assert report.same_passage.detected == 0


# --- the dataset itself ----------------------------------------------------


@pytest.mark.skipif(
    not __import__("pathlib").Path(DATA).exists(),
    reason="WikiContradict is a local measurement artifact; data/ is gitignored",
)
def test_the_annotated_set_is_the_one_phase_1_measured() -> None:
    cases = load_wikicontradict(DATA)
    assert len(cases) == 253
    assert sum(case.same_passage for case in cases) == 63


@pytest.mark.skipif(
    not __import__("pathlib").Path(DATA).exists(),
    reason="WikiContradict is a local measurement artifact; data/ is gitignored",
)
def test_the_same_passage_bucket_cannot_answer_the_question_asked_of_it() -> None:
    """The measured finding of 2026-08-26, pinned so it is not re-litigated.

    Of 63 same-passage cases, 21 are a single sentence - unreachable by any
    extractor - and 30 answer yes/no, which is a reasoning output rather than
    a span and cannot be traced to a sentence at all. Seven remain traceable.
    A detection rate over seven cases is not a number, and the harness must
    not pretend otherwise.

    The count is pinned exactly because it moved once: an over-merging bug in
    the splitter reported 22, and it moved in the direction that would have
    flattered the hypothesis. A number that can drift silently is not
    evidence.
    """
    cases = [case for case in load_wikicontradict(DATA) if case.same_passage]
    single_sentence = sum(1 for case in cases if len(split_sentences(case.passage)) < 2)
    traceable = sum(
        1
        for case in cases
        if trace_answers(case.passage, case.answer_a, case.answer_b) is not None
    )
    assert single_sentence == 21, single_sentence
    assert traceable <= 7, traceable
    assert traceable / len(cases) < 0.15


def test_an_unrun_subject_gate_reports_nothing_rather_than_zero() -> None:
    """The artefact this pins actually happened, in the first real run.

    `shared_subject_pairs` qualifies a text by a rare name the extractor
    found IN it, so an empty entity registry drops every pair. The first
    control run passed `{}` and printed `end-to-end 0.0%` - a number that
    said nothing about the filter and everything about the caller. It is
    `None` now, and `None` cannot be mistaken for a measurement.
    """
    report = measure_sensitivity([_case()], FixedModel(1.0), entities=None)
    assert report.same_passage.detected == 1
    assert report.same_passage.survived is None
    assert report.same_passage.end_to_end_rate is None


def test_the_gate_runs_when_an_entity_registry_is_supplied() -> None:
    report = measure_sensitivity(
        [_case(question_id="7")],
        FixedModel(1.0),
        entities={"7:a": ["Alice"], "7:b": ["Bob"]},
    )
    # Two different subjects: the filter is doing its job by dropping it.
    assert report.same_passage.survived == 0
    assert report.same_passage.end_to_end_rate == 0.0
