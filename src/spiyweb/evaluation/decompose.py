"""LLM query shaping for the coloured web: decomposition and answer chaining.

Two calls per question, both prompt-cached and both with a hard fallback:

1. Decomposition splits the question into at most `max_colors` search queries
   (the colours). An unparseable reply falls back to the whole question as a
   single colour - a question must never be lost to a chatty LLM.
2. Answer extraction reads the primary colour's top passage and pulls out the
   intermediate answer that the later colours' queries are missing. The prompt
   demands the literal reply `NONE` when the passage does not answer the
   sub-question; blind chaining without that guard measurably HURT (tour 8,
   S@5 .404 < .413), because a wrong top-1 passage poisons every later colour.

This module is deliberately outside `core/` (it talks to an LLM) and inside
`evaluation/` for Phase 1: promoting query shaping to a stable public API is
Phase 2 work, and freezing its surface now would do so ahead of the evidence.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from spiyweb.prompts import INTERMEDIATE_ANSWER_PROMPT, QUERY_DECOMPOSITION_PROMPT

if TYPE_CHECKING:
    from spiyweb.llm import LLMClient

_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_CHATTER_PREFIXES = ("query", "queries", "here")


def parse_subqueries(reply: str, fallback: str, max_colors: int) -> list[str]:
    """Sub-queries from a decomposition reply, one per line, chatter stripped.

    List markers (bullets, numbering) are removed per line; lines that are
    obviously scaffolding ("Queries:", "Here are ...") are dropped. An empty
    or fully-chatter reply yields `[fallback]` - never an empty list.
    """
    lines: list[str] = []
    for raw in reply.splitlines():
        line = _LIST_MARKER.sub("", raw).strip()
        if line and not line.lower().startswith(_CHATTER_PREFIXES):
            lines.append(line)
    return lines[:max_colors] if lines else [fallback]


def decompose_question(question: str, llm: LLMClient, max_colors: int) -> list[str]:
    """The question's colours: LLM decomposition with the whole-question
    fallback."""
    reply = llm.complete(QUERY_DECOMPOSITION_PROMPT.format(question=question))
    return parse_subqueries(reply, question, max_colors)


def parse_intermediate_answer(reply: str, max_words: int) -> str:
    """The extracted answer phrase, or `""` when the LLM declined.

    Only the first line counts (later lines are explanation the prompt forbade
    anyway), a reply starting with NONE means "passage does not answer", and
    the phrase is capped at `max_words` words.
    """
    line = reply.strip().splitlines()[0].strip() if reply.strip() else ""
    if not line or line.upper().startswith("NONE"):
        return ""
    return " ".join(line.split()[:max_words])


def extract_intermediate_answer(
    llm: LLMClient, title: str, text: str, subquestion: str, max_words: int
) -> str:
    """The chained fact: answer of `subquestion` read from one passage."""
    reply = llm.complete(
        INTERMEDIATE_ANSWER_PROMPT.format(
            title=title, text=text, subquestion=subquestion
        )
    )
    return parse_intermediate_answer(reply, max_words)
