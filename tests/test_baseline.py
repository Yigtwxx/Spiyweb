"""The baselines must be honest competitors, not strawmen.

The claims under test: the iterative baseline genuinely follows the LLM's
rewritten queries into new corpus regions (that is its whole advantage), its
knobs cap it exactly as configured, and with the LLM ablated it degenerates
to plain `top-k` - the pair that proves the iteration earns its keep. Plus:
the deterministic LLM cache pays each prompt exactly once, across instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spiyweb.config import IterativeBaselineConfig
from spiyweb.evaluation.baseline import (
    IterativeTrace,
    _build_prompt,
    iterative_retrieve,
    topk_retrieve,
)
from spiyweb.evaluation.cache import CachedLLMClient

if TYPE_CHECKING:
    from pathlib import Path

QUESTION = "who built the tower?"
STEP_ONE = "I still need the location of the tower."
ANSWER_STEP = "So the answer is Tesla."

# Scripted embedding table: each query text maps to a distinct vector.
VECTORS = {
    QUESTION: [1.0, 0.0],
    STEP_ONE: [0.0, 1.0],
    "First sentence.": [0.0, 1.0],
}

# Scripted index: each query vector reaches a different corpus region.
ROUTES = {
    (1.0, 0.0): [("p1", 0.9), ("p2", 0.8)],
    (0.0, 1.0): [("p3", 0.95), ("p1", 0.5)],
}

TEXTS = {"p1": "text one", "p2": "text two", "p3": "text three"}
TITLES = {"p1": "One", "p2": "Two", "p3": "Three"}


class FakeEmbedder:
    def __init__(self) -> None:
        self.query_calls: list[str] = []

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.query_calls.extend(texts)
        return [VECTORS[text] for text in texts]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the baseline must never embed passages")


class FakeSeedSource:
    def search(self, query: list[float], k: int) -> list[tuple[str, float]]:
        return ROUTES[tuple(query)][:k]


class FakeLLM:
    """Plays back a script; records the prompts it saw."""

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.script.pop(0)


def run(llm: FakeLLM, config: IterativeBaselineConfig | None = None) -> IterativeTrace:
    return iterative_retrieve(
        QUESTION,
        FakeEmbedder(),
        FakeSeedSource(),
        TEXTS,
        TITLES,
        llm,
        config,
    )


def test_topk_retrieve_returns_the_index_order() -> None:
    assert topk_retrieve([1.0, 0.0], FakeSeedSource(), 2) == ["p1", "p2"]


def test_proposition_ids_are_read_through_their_parent_passage() -> None:
    """A two-layer index ranks propositions, and only passages have text.

    Before the fold this raised `KeyError` on the first proposition contact
    and took the whole run down with it.
    """
    prompt = _build_prompt(QUESTION, ["p1#p0", "p1#p3", "p2"], TEXTS, TITLES, [])
    assert "text one" in prompt and "text two" in prompt
    assert prompt.count("text one") == 1, (
        "two propositions of one passage must not fill the prompt twice"
    )


def test_the_rewritten_query_reaches_a_new_corpus_region() -> None:
    trace = run(FakeLLM([STEP_ONE, ANSWER_STEP]))

    assert "p3" in trace.ranked, (
        "p3 is only reachable through the rewritten query - if it is missing "
        "the loop never followed the LLM, and the baseline is a strawman"
    )
    # Ranked by best cosine seen: p3 (.95) > p1 (.9, not the later .5) > p2.
    assert trace.ranked == ("p3", "p1", "p2")
    assert trace.steps == (STEP_ONE, ANSWER_STEP)
    assert trace.stopped_early is True


def test_the_stop_phrase_is_case_insensitive() -> None:
    trace = run(FakeLLM(["The ANSWER IS Tesla."]))
    assert trace.stopped_early is True
    assert trace.ranked == ("p1", "p2"), (
        "an immediate answer means no second retrieval round ever ran"
    )


def test_only_the_first_sentence_becomes_the_next_query() -> None:
    embedder = FakeEmbedder()
    iterative_retrieve(
        QUESTION,
        embedder,
        FakeSeedSource(),
        TEXTS,
        TITLES,
        FakeLLM(["First sentence. Second sentence never retrieves.", ANSWER_STEP]),
    )
    assert embedder.query_calls == [QUESTION, "First sentence."], (
        "IRCoT keeps exactly the first generated sentence as the next query"
    )


def test_max_steps_caps_the_llm_rounds() -> None:
    llm = FakeLLM([STEP_ONE, STEP_ONE, STEP_ONE])
    trace = run(llm, IterativeBaselineConfig(max_steps=2))
    assert len(llm.prompts) == 2
    assert trace.stopped_early is False


def test_max_steps_zero_is_the_plain_topk_ablation() -> None:
    llm = FakeLLM([])
    trace = run(llm, IterativeBaselineConfig(max_steps=0))
    assert llm.prompts == [], "with zero steps the LLM must never be consulted"
    assert list(trace.ranked) == topk_retrieve([1.0, 0.0], FakeSeedSource(), 5), (
        "the ablated iterative baseline must degenerate to plain top-k - "
        "that equivalence is what isolates the iteration's contribution"
    )


def test_max_collected_admits_no_new_documents_beyond_the_cap() -> None:
    trace = run(
        FakeLLM([STEP_ONE, ANSWER_STEP]),
        IterativeBaselineConfig(per_step_k=2, max_collected=2),
    )
    assert trace.ranked == ("p1", "p2"), (
        "p3 arrives after the cap is full and must be turned away; already "
        "collected documents may still improve their score"
    )


def test_an_empty_llm_sentence_ends_the_loop_quietly() -> None:
    trace = run(FakeLLM(["   \n  "]))
    assert trace.steps == ()
    assert trace.stopped_early is False
    assert trace.ranked == ("p1", "p2")


def test_the_prompt_carries_question_paragraphs_and_reasoning() -> None:
    llm = FakeLLM([STEP_ONE, ANSWER_STEP])
    run(llm)
    first, second = llm.prompts
    assert QUESTION in first
    assert "One: text one" in first
    assert "(nothing yet)" in first
    assert STEP_ONE in second, "later prompts must carry the reasoning so far"
    assert "Three: text three" in second, (
        "later prompts must show the newly retrieved paragraphs"
    )


def test_iterative_config_validation() -> None:
    with pytest.raises(ValueError, match="per_step_k"):
        IterativeBaselineConfig(per_step_k=0)
    with pytest.raises(ValueError, match="max_steps"):
        IterativeBaselineConfig(max_steps=-1)
    with pytest.raises(ValueError, match="stop_phrase"):
        IterativeBaselineConfig(stop_phrase="")
    with pytest.raises(ValueError, match="max_collected"):
        IterativeBaselineConfig(per_step_k=5, max_collected=4)


def test_the_cache_pays_each_prompt_exactly_once(tmp_path: Path) -> None:
    class CountingLLM:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def complete(self, prompt: str) -> str:
            self.calls.append(prompt)
            return f"reply to {prompt}"

    cache_file = tmp_path / "llm_cache.jsonl"
    inner = CountingLLM()
    client = CachedLLMClient(inner, cache_file)

    assert client.complete("alpha") == "reply to alpha"
    assert client.complete("alpha") == "reply to alpha"
    assert inner.calls == ["alpha"], "the second identical prompt reads the cache"

    # A fresh instance over the same file replays without touching the inner
    # client at all - that file IS the crash-resume and the reproducibility.
    later_inner = CountingLLM()
    later = CachedLLMClient(later_inner, cache_file)
    assert later.complete("alpha") == "reply to alpha"
    assert later_inner.calls == []

    assert later.complete("beta") == "reply to beta"
    assert later_inner.calls == ["beta"]
