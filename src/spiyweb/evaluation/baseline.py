"""The two honest competitors: plain `top-k` and IRCoT-style iteration.

`top-k` is the motivating baseline - and a surprisingly strong one. Iterative
retrieval is the real rival: an LLM reads what was retrieved, writes the next
reasoning sentence, and that sentence becomes the next query; much cheaper to
build than a graph and strong on multi-hop. The Phase 1 gate requires beating
BOTH (D28).

Context-budget parity is the harness's job, not this module's: every system
returns a ranked list and the metrics layer cuts all of them at the same k.
The iterative baseline's legitimate advantage is extra retrieval rounds, not
a longer answer list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiyweb.config import IterativeBaselineConfig
from spiyweb.prompts import QUERY_REWRITE_PROMPT

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from spiyweb.embedding import Embedder
    from spiyweb.llm import LLMClient
    from spiyweb.retrieve import SeedSource

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s")


def topk_retrieve(
    query_embedding: Sequence[float], index: SeedSource, k: int
) -> list[str]:
    """Plain dense retrieval: the index's top-k, nothing else."""
    return [node for node, _ in index.search(query_embedding, k)]


@dataclass(frozen=True)
class IterativeTrace:
    """What one iterative run produced, kept for the per-query record.

    Attributes:
        ranked: The union of all rounds' retrievals, ranked by best cosine
            seen, ties broken by earlier discovery step, then id.
        steps: The reasoning sentences the LLM produced, in order.
        stopped_early: True when the stop phrase fired before `max_steps`.
    """

    ranked: tuple[str, ...]
    steps: tuple[str, ...]
    stopped_early: bool


def _first_sentence(text: str) -> str:
    """IRCoT keeps only the first generated sentence; so do we."""
    for line in text.strip().splitlines():
        candidate = line.strip()
        if candidate:
            boundary = _SENTENCE_BOUNDARY.search(candidate)
            return candidate[: boundary.start()] if boundary else candidate
    return ""


def _build_prompt(
    question: str,
    ranked_ids: Sequence[str],
    texts: Mapping[str, str],
    titles: Mapping[str, str],
    steps: Sequence[str],
) -> str:
    paragraphs = "\n\n".join(
        f"{titles.get(chunk_id, chunk_id)}: {texts[chunk_id]}"
        for chunk_id in ranked_ids
    )
    reasoning = "\n".join(steps) if steps else "(nothing yet)"
    return QUERY_REWRITE_PROMPT.format(
        question=question, paragraphs=paragraphs, reasoning=reasoning
    )


def iterative_retrieve(
    question: str,
    embedder: Embedder,
    index: SeedSource,
    texts: Mapping[str, str],
    titles: Mapping[str, str],
    llm: LLMClient,
    config: IterativeBaselineConfig | None = None,
) -> IterativeTrace:
    """Run the IRCoT-style loop and return the ranked union with its trace.

    Round 0 retrieves with the question itself. Each later round hands the
    LLM everything collected so far, keeps the FIRST sentence it writes, and
    either stops (the sentence contains `stop_phrase`, case-insensitively) or
    retrieves with that sentence and unions the results in. An empty LLM
    sentence ends the loop quietly - there is nothing left to retrieve with.

    New documents stop being admitted once `max_collected` is reached;
    already collected documents still improve their best score.
    """
    cfg = config if config is not None else IterativeBaselineConfig()

    # id -> (best cosine seen, step first discovered); the ranking key.
    collected: dict[str, tuple[float, int]] = {}

    def admit(results: Sequence[tuple[str, float]], step: int) -> None:
        for chunk_id, score in results:
            known = collected.get(chunk_id)
            if known is None:
                if len(collected) < cfg.max_collected:
                    collected[chunk_id] = (score, step)
            elif score > known[0]:
                collected[chunk_id] = (score, known[1])

    def ranked_ids() -> list[str]:
        return sorted(
            collected,
            key=lambda chunk_id: (
                -collected[chunk_id][0],
                collected[chunk_id][1],
                chunk_id,
            ),
        )

    query = embedder.embed_queries([question])[0]
    admit(index.search(query, cfg.per_step_k), step=0)

    steps: list[str] = []
    stopped_early = False
    for step in range(1, cfg.max_steps + 1):
        prompt = _build_prompt(question, ranked_ids(), texts, titles, steps)
        sentence = _first_sentence(llm.complete(prompt))
        if not sentence:
            break
        steps.append(sentence)
        if cfg.stop_phrase.lower() in sentence.lower():
            stopped_early = True
            break
        query = embedder.embed_queries([sentence])[0]
        admit(index.search(query, cfg.per_step_k), step=step)

    return IterativeTrace(
        ranked=tuple(ranked_ids()),
        steps=tuple(steps),
        stopped_early=stopped_early,
    )
