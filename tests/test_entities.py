"""Hybrid extraction: label filter, normalisation, LLM fallback routing."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest

from spiyweb import EntityExtractionConfig
from spiyweb.entities import (
    extract_entities,
    load_spacy_pipeline,
    normalize_entity,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class FakeSpan:
    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label_ = label


class FakeDoc:
    def __init__(self, ents: list[FakeSpan]) -> None:
        self.ents = ents


class FakePipeline:
    """Yields pre-scripted spans per text, in call order."""

    def __init__(self, ents_per_text: list[list[FakeSpan]]) -> None:
        self.ents_per_text = ents_per_text
        self.seen_texts: list[str] = []

    def pipe(self, texts: Iterable[str]) -> Iterator[FakeDoc]:
        for text, ents in zip(list(texts), self.ents_per_text, strict=True):
            self.seen_texts.append(text)
            yield FakeDoc(ents)


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_label_filter_keeps_configured_labels_and_drops_the_rest() -> None:
    pipeline = FakePipeline(
        [[FakeSpan("Marie Curie", "PER"), FakeSpan("1903", "DATE")]]
    )
    result = extract_entities({"c1": "text"}, pipeline)
    assert result == {"c1": ["marie curie"]}, (
        "a shared '1903' is not hop fuel; temporal labels stay out"
    )


def test_entities_are_normalized_and_deduped_preserving_first_occurrence() -> None:
    pipeline = FakePipeline(
        [
            [
                FakeSpan("  Marie   Curie ", "PER"),
                FakeSpan("Sorbonne", "ORG"),
                FakeSpan("marie curie", "PER"),
            ]
        ]
    )
    result = extract_entities({"c1": "text"}, pipeline)
    assert result == {"c1": ["marie curie", "sorbonne"]}


def test_no_llm_means_no_fallback_even_for_empty_chunks() -> None:
    pipeline = FakePipeline([[]])
    result = extract_entities({"c1": "text"}, pipeline, llm=None)
    assert result == {"c1": []}, "llm=None is the ablation switch for the path"


def test_llm_is_called_only_for_chunks_below_min_entities() -> None:
    pipeline = FakePipeline([[FakeSpan("Tesla", "ORG")], []])
    llm = FakeLLM("Wardenclyffe Tower")
    result = extract_entities({"rich": "text a", "blind": "text b"}, pipeline, llm=llm)
    assert result == {"rich": ["tesla"], "blind": ["wardenclyffe tower"]}
    assert len(llm.prompts) == 1, "the spaCy-rich chunk must not cost a call"
    assert "text b" in llm.prompts[0]


def test_min_entities_zero_never_calls_the_llm() -> None:
    pipeline = FakePipeline([[]])
    llm = FakeLLM("anything")
    extract_entities(
        {"c1": "text"},
        pipeline,
        config=EntityExtractionConfig(min_entities=0),
        llm=llm,
    )
    assert llm.prompts == []


def test_llm_reply_parsing_tolerates_bullets_numbering_and_blank_lines() -> None:
    pipeline = FakePipeline([[]])
    llm = FakeLLM("- Ada Lovelace\n\n* Babbage\n2. Analytical Engine\n+ London\n")
    result = extract_entities({"c1": "text"}, pipeline, llm=llm)
    assert result == {"c1": ["ada lovelace", "babbage", "analytical engine", "london"]}


def test_llm_reply_with_nothing_usable_yields_an_empty_list() -> None:
    pipeline = FakePipeline([[]])
    llm = FakeLLM("\n   \n")
    result = extract_entities({"c1": "text"}, pipeline, llm=llm)
    assert result == {"c1": []}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Marie Curie", "marie curie"),
        ("  spaced   out  ", "spaced out"),
        ("İSTANBUL", "i̇stanbul"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_entity_truth_table(raw: str, expected: str) -> None:
    assert normalize_entity(raw) == expected


def test_config_rejects_empty_model_and_negative_min_entities() -> None:
    with pytest.raises(ValueError, match="spacy_model"):
        EntityExtractionConfig(spacy_model="")
    with pytest.raises(ValueError, match="min_entities"):
        EntityExtractionConfig(min_entities=-1)


@pytest.mark.skipif(
    find_spec("spacy") is not None, reason="spaCy installed; hint path unreachable"
)
def test_load_spacy_pipeline_without_spacy_names_the_extra() -> None:
    with pytest.raises(ImportError, match=r"spiyweb\[entity\]"):
        load_spacy_pipeline()


@pytest.mark.skipif(
    find_spec("spacy") is None, reason="needs spaCy to reach the model check"
)
def test_load_spacy_pipeline_with_missing_model_names_the_download() -> None:
    config = EntityExtractionConfig(spacy_model="xx_no_such_model")
    with pytest.raises(OSError, match="spacy download xx_no_such_model"):
        load_spacy_pipeline(config)
