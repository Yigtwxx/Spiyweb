"""Polarity detection via the extraction call (open question #11, D34 feed).

The claim under test: with `tag_polarity=True` the extractor asks for `NEG:`
prefixes in the SAME call - zero extra cost - and a tagged line becomes a
`polarity=-1` node with the prefix stripped; the dedup key ignores the tag (a
fact emitted both ways counts once, D6).

The switch DEFAULTS OFF since 2026-08-16: audited on 3.336 real passages, the
piggyback tagged either wrongly (47.2% cue-backed, invented denials in the
dumps) or hardly at all (4 tags in 29.566, misses 160x the catches). The
plumbing below is still exact and still tested - what the audit refuted is the
prompt as a DETECTOR, not the transport it feeds.
"""

from __future__ import annotations

from spiyweb import PropositionConfig, extract_propositions
from spiyweb.nodes import DocumentInput, TextUnit, chunk_documents
from spiyweb.prompts import (
    PROPOSITION_EXTRACTION_POLARITY_PROMPT,
    PROPOSITION_EXTRACTION_PROMPT,
)

CHUNKS = chunk_documents(
    [DocumentInput(source_id="d0", units=(TextUnit(text="Some source text."),))]
)
# The mechanism ships OFF; every test of the mechanism must ask for it.
POLARITY_ON = PropositionConfig(tag_polarity=True)


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_neg_prefix_becomes_a_negative_atom_with_the_prefix_stripped() -> None:
    llm = FakeLLM(
        "The plant produces solar panels.\n"
        "NEG: The plant no longer produces batteries.\n"
    )
    first, second = extract_propositions(CHUNKS, llm, POLARITY_ON)
    assert first.node.polarity == 1
    assert second.node.polarity == -1, "the tag must land as node polarity (D34)"
    assert second.text == "The plant no longer produces batteries.", (
        "the NEG: prefix is transport, never content - embedding and entity "
        "extraction must see the clean sentence"
    )
    assert second.node.length == len(second.text), "length counts the clean text"


def test_the_polarity_prompt_is_sent_only_when_asked_for() -> None:
    llm = FakeLLM("A plain factual sentence here.")
    extract_propositions(CHUNKS, llm, POLARITY_ON)
    assert llm.prompts[0] == PROPOSITION_EXTRACTION_POLARITY_PROMPT.format(
        text="Some source text."
    ), "tag_polarity=True must send the NEG-aware template"


def test_the_measured_default_is_polarity_off() -> None:
    """The 2026-08-16 audit refuted the piggyback in both directions, so the
    shipped default sends the polarity-free prompt. A silent flip back would
    re-enable a detector known to mislabel."""
    llm = FakeLLM("A plain factual sentence here.")
    extract_propositions(CHUNKS, llm)
    assert PropositionConfig().tag_polarity is False
    assert llm.prompts[0] == PROPOSITION_EXTRACTION_PROMPT.format(
        text="Some source text."
    )


def test_tag_polarity_off_reproduces_pre_11_behaviour() -> None:
    llm = FakeLLM("NEG: The plant no longer produces batteries.")
    config = PropositionConfig(tag_polarity=False)
    (only,) = extract_propositions(CHUNKS, llm, config)
    assert llm.prompts[0] == PROPOSITION_EXTRACTION_PROMPT.format(
        text="Some source text."
    ), "the ablation must send the polarity-free template"
    assert only.node.polarity == 1
    assert only.text.startswith("NEG:"), (
        "without the mechanism a NEG: line is ordinary text - the switch must "
        "not half-apply"
    )


def test_dedup_key_ignores_the_tag_and_first_occurrence_wins() -> None:
    llm = FakeLLM(
        "The lab closed in 1999 for good reasons.\n"
        "NEG: The lab closed in 1999 for good reasons.\n"
    )
    kept = extract_propositions(CHUNKS, llm, POLARITY_ON)
    assert len(kept) == 1, "tagged and untagged twins are one fact (D6 votes)"
    assert kept[0].node.polarity == 1, "first occurrence wins, polarity included"


def test_min_chars_filters_on_the_stripped_text() -> None:
    llm = FakeLLM("NEG: tiny.")
    kept = extract_propositions(
        CHUNKS, llm, PropositionConfig(min_chars=15, tag_polarity=True)
    )
    assert kept == [], (
        "a fragment must not survive because the NEG: prefix padded its length"
    )
