"""LLM query shaping: decomposition parsing and answer-extraction parsing.

Both parsers face a chatty 8B model, so the tests pin the exact cleanups the
measurement campaign relied on: list markers stripped, scaffolding lines
dropped, the whole-question fallback when nothing survives, the NONE guard
(whose absence measurably HURT in tour 8), and the word cap on the chained
answer. The end-to-end functions are exercised with a scripted fake LLM.
"""

from __future__ import annotations

import pytest

from spiyweb.evaluation.decompose import (
    decompose_question,
    extract_intermediate_answer,
    parse_intermediate_answer,
    parse_subqueries,
)
from spiyweb.prompts import INTERMEDIATE_ANSWER_PROMPT, QUERY_DECOMPOSITION_PROMPT


class ScriptedLLM:
    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.script.pop(0)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            "song Green performer\nperformer spouse",
            ["song Green performer", "performer spouse"],
        ),
        ("1. first fact\n2) second fact", ["first fact", "second fact"]),
        (
            "- first fact\n* second fact\n• third fact",
            ["first fact", "second fact", "third fact"],
        ),
        ("Queries:\nfirst fact\nsecond fact", ["first fact", "second fact"]),
        ("Here are the queries:\nfirst fact", ["first fact"]),
        ("first fact\n\n   \nsecond fact", ["first fact", "second fact"]),
    ],
)
def test_parse_subqueries_strips_markers_and_scaffolding(
    reply: str, expected: list[str]
) -> None:
    assert parse_subqueries(reply, "fallback", 4) == expected


@pytest.mark.parametrize("reply", ["", "   \n  ", "Queries:\nHere you go:"])
def test_parse_subqueries_falls_back_to_the_whole_question(reply: str) -> None:
    assert parse_subqueries(reply, "the whole question", 4) == ["the whole question"], (
        "a question must never be lost to a chatty or empty reply"
    )


def test_parse_subqueries_caps_the_colour_count() -> None:
    reply = "\n".join(f"fact {index}" for index in range(6))
    assert parse_subqueries(reply, "fallback", 4) == [
        "fact 0",
        "fact 1",
        "fact 2",
        "fact 3",
    ]


def test_decompose_question_formats_the_prompt_and_parses_the_reply() -> None:
    llm = ScriptedLLM(["first fact\nsecond fact"])
    result = decompose_question("Who did what?", llm, 4)

    assert result == ["first fact", "second fact"]
    assert llm.prompts == [
        QUERY_DECOMPOSITION_PROMPT.format(question="Who did what?")
    ], "the template is the cache key - it must be used verbatim"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Paris", "Paris"),
        ("Paris\nBecause the passage says so.", "Paris"),
        ("  Paris  ", "Paris"),
        ("NONE", ""),
        ("None of the passage answers this", ""),
        ("", ""),
        ("   \n  ", ""),
    ],
)
def test_parse_intermediate_answer_first_line_and_none_guard(
    reply: str, expected: str
) -> None:
    assert parse_intermediate_answer(reply, 10) == expected


def test_parse_intermediate_answer_caps_the_word_count() -> None:
    reply = "one two three four five six seven eight nine ten eleven twelve"
    assert parse_intermediate_answer(reply, 10) == (
        "one two three four five six seven eight nine ten"
    ), "a rambling extraction would drown the sub-query it is appended to"


def test_extract_intermediate_answer_formats_the_prompt_and_parses() -> None:
    llm = ScriptedLLM(["Wardenclyffe Tower\nextra chatter"])
    result = extract_intermediate_answer(llm, "Alpha", "alpha text", "where is it?", 10)

    assert result == "Wardenclyffe Tower"
    assert llm.prompts == [
        INTERMEDIATE_ANSWER_PROMPT.format(
            title="Alpha", text="alpha text", subquestion="where is it?"
        )
    ], "the template is the cache key - it must be used verbatim"
