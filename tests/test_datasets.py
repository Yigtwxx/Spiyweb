"""MuSiQue loading: pooling must dedup, sampling must be deterministic.

The claims under test: (1) the same Wikipedia paragraph appearing under two
questions becomes ONE corpus node - the dedup mechanism that will eventually
suppress duplicates at query time does not exist yet, so the corpus must not
smuggle exact duplicates into the graph; (2) the question sample is a pure
function of (file content, seed) - never of file order or platform; (3) gold
and bridge-gold labels survive the mapping into corpus chunk ids intact.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from spiyweb.config import EvaluationConfig
from spiyweb.evaluation.datasets import download_musique, load_dataset

if TYPE_CHECKING:
    from pathlib import Path


def make_record(
    question_id: str,
    paragraphs: list[tuple[str, str, bool]],
    support_idxs: list[int],
) -> dict[str, object]:
    """One MuSiQue-Ans record; paragraphs are (title, text, is_supporting)."""
    return {
        "id": question_id,
        "question": f"question of {question_id}",
        "answer": f"answer of {question_id}",
        "answerable": True,
        "paragraphs": [
            {
                "idx": idx,
                "title": title,
                "paragraph_text": text,
                "is_supporting": supporting,
            }
            for idx, (title, text, supporting) in enumerate(paragraphs)
        ],
        "question_decomposition": [
            {
                "id": step,
                "question": f"step {step}",
                "answer": "x",
                "paragraph_support_idx": support_idx,
            }
            for step, support_idx in enumerate(support_idxs)
        ],
    }


RECORDS = [
    make_record(
        "2hop__q1",
        [
            ("Alpha", "alpha text", True),
            ("Beta", "beta text", True),
            ("Noise", "noise text", False),
        ],
        support_idxs=[0, 1],
    ),
    make_record(
        "3hop1__q2",
        [
            ("Alpha", "alpha text", True),  # exact duplicate of q1's paragraph
            ("Gamma", "gamma text", True),
            ("Delta", "delta text", True),
            ("Noise2", "noise2 text", False),
        ],
        support_idxs=[0, 1, 2],
    ),
]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_pooling_collapses_the_shared_paragraph_into_one_node(
    tmp_path: Path,
) -> None:
    dataset = load_dataset(write_jsonl(tmp_path / "dev.jsonl", RECORDS))
    # 7 raw paragraphs, but Alpha appears under both questions: 6 corpus docs.
    assert len(dataset.documents) == 6, (
        "the same (title, text) paragraph under two questions must become ONE "
        "corpus node - query-time dedup does not exist yet to save us"
    )


def test_corpus_ids_are_deterministic_over_sorted_dedup_keys(
    tmp_path: Path,
) -> None:
    dataset = load_dataset(write_jsonl(tmp_path / "dev.jsonl", RECORDS))
    # Keys sort by title: Alpha, Beta, Delta, Gamma, Noise, Noise2.
    assert dataset.titles["d00000:0"] == "Alpha"
    assert dataset.titles["d00003:0"] == "Gamma"
    assert dataset.texts["d00002:0"] == "delta text"
    assert [document.source_id for document in dataset.documents] == [
        f"d{i:05d}" for i in range(6)
    ]


def test_gold_and_bridge_labels_map_to_corpus_chunk_ids(tmp_path: Path) -> None:
    dataset = load_dataset(write_jsonl(tmp_path / "dev.jsonl", RECORDS))
    q1, q2 = dataset.questions

    assert q1.id == "2hop__q1"
    assert q1.hops == 2
    assert set(q1.gold_ids) == {"d00000:0", "d00001:0"}  # Alpha, Beta
    assert q1.bridge_gold_ids == ("d00000:0",), (
        "bridge gold is every decomposition step but the last - for a 2-hop "
        "question exactly the one intermediate document"
    )

    assert q2.hops == 3
    assert set(q2.gold_ids) == {"d00000:0", "d00003:0", "d00002:0"}
    assert q2.bridge_gold_ids == ("d00000:0", "d00003:0")  # Alpha, Gamma


def test_sampling_is_a_pure_function_of_content_and_seed(tmp_path: Path) -> None:
    many = [
        make_record(f"2hop__q{i}", [("T", f"text {i}", True)], support_idxs=[0])
        for i in range(10)
    ]
    path = write_jsonl(tmp_path / "dev.jsonl", many)
    config = EvaluationConfig(sample_size=4, sample_seed=42)

    first = load_dataset(path, config)
    second = load_dataset(path, config)
    assert [q.id for q in first.questions] == [q.id for q in second.questions]
    assert len(first.questions) == 4

    shuffled = write_jsonl(tmp_path / "shuffled.jsonl", list(reversed(many)))
    third = load_dataset(shuffled, config)
    assert [q.id for q in third.questions] == [q.id for q in first.questions], (
        "the draw happens over the SORTED id list, so file order must never "
        "change the sample"
    )

    other_seed = load_dataset(path, EvaluationConfig(sample_size=4, sample_seed=7))
    assert [q.id for q in other_seed.questions] != [q.id for q in first.questions]


def test_sample_size_zero_means_the_full_split(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "dev.jsonl", RECORDS)
    dataset = load_dataset(path, EvaluationConfig(sample_size=0))
    assert len(dataset.questions) == 2


def test_sample_size_beyond_the_split_keeps_everything(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "dev.jsonl", RECORDS)
    dataset = load_dataset(path, EvaluationConfig(sample_size=999))
    assert len(dataset.questions) == 2


def test_a_question_without_supports_is_loader_corruption(tmp_path: Path) -> None:
    broken = [make_record("2hop__q1", [("T", "t", False)], support_idxs=[0])]
    with pytest.raises(ValueError, match="no supporting paragraph"):
        load_dataset(write_jsonl(tmp_path / "dev.jsonl", broken))


def test_an_id_without_a_hop_prefix_is_rejected(tmp_path: Path) -> None:
    broken = [make_record("mystery__q1", [("T", "t", True)], support_idxs=[0])]
    with pytest.raises(ValueError, match="hop prefix"):
        load_dataset(write_jsonl(tmp_path / "dev.jsonl", broken))


def test_an_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        load_dataset(path)


def test_download_writes_the_payload_and_skips_when_present(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return b'{"id": "2hop__x"}\n'

    target = tmp_path / "nested" / "dev.jsonl"
    first = download_musique(target, "https://example.test/dev.jsonl", fetch)
    assert first == target
    assert target.read_bytes() == b'{"id": "2hop__x"}\n'

    download_musique(target, "https://example.test/dev.jsonl", fetch)
    assert calls == ["https://example.test/dev.jsonl"], (
        "an existing file must short-circuit the download - that existence "
        "check IS the resume mechanism"
    )


def test_the_default_fetcher_refuses_plain_http(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="https"):
        download_musique(tmp_path / "dev.jsonl", "http://example.test/dev.jsonl")


# ------------------------------------------------------------- 2Wiki


def make_2wiki_record(
    raw_id: str,
    qtype: str,
    answer: str,
    context: list[tuple[str, list[str]]],
    supporting_titles: list[str],
) -> dict[str, object]:
    """One original-format 2Wiki record; context entries are (title, sents)."""
    return {
        "_id": raw_id,
        "type": qtype,
        "question": f"question of {raw_id}",
        "answer": answer,
        "context": [[title, sentences] for title, sentences in context],
        "supporting_facts": [[title, 0] for title in supporting_titles],
        "evidences": [],
    }


TWOWIKI_RECORDS = [
    make_2wiki_record(
        "abc1",
        "compositional",
        "Paris",
        context=[
            ("Person", ["Born in the capital.", "More."]),
            ("Capital", ["The capital is Paris."]),
            ("Filler", ["Unrelated text."]),
        ],
        supporting_titles=["Person", "Capital"],
    ),
    make_2wiki_record(
        "abc2",
        "comparison",
        "no",
        context=[
            ("Left", ["Left thing."]),
            ("Right", ["Right thing."]),
            ("Filler", ["Unrelated text."]),  # exact duplicate of abc1's filler
        ],
        supporting_titles=["Left", "Right"],
    ),
]


def write_2wiki(path: Path, records: list[dict[str, object]]) -> Path:
    target = path / "2wiki_dev.json"
    target.write_text(json.dumps(records), encoding="utf-8")
    return target


def test_2wiki_pools_and_dedups_across_questions(tmp_path: Path) -> None:
    from spiyweb.evaluation.datasets import load_2wiki

    dataset = load_2wiki(write_2wiki(tmp_path, TWOWIKI_RECORDS))
    assert len(dataset.documents) == 5, (
        "the shared Filler paragraph must become ONE corpus node"
    )
    assert sorted(dataset.titles.values()) == [
        "Capital",
        "Filler",
        "Left",
        "Person",
        "Right",
    ]


def test_2wiki_ids_carry_the_type_prefix_and_hops_count_supports(
    tmp_path: Path,
) -> None:
    from spiyweb.evaluation.datasets import load_2wiki

    dataset = load_2wiki(write_2wiki(tmp_path, TWOWIKI_RECORDS))
    by_id = {question.id: question for question in dataset.questions}
    assert set(by_id) == {"compositional__abc1", "comparison__abc2"}
    assert by_id["compositional__abc1"].hops == 2


def test_2wiki_bridge_excludes_the_answer_bearing_paragraph(
    tmp_path: Path,
) -> None:
    from spiyweb.evaluation.datasets import load_2wiki

    dataset = load_2wiki(write_2wiki(tmp_path, TWOWIKI_RECORDS))
    by_id = {question.id: question for question in dataset.questions}
    compositional = by_id["compositional__abc1"]
    bridge_titles = {dataset.titles[c] for c in compositional.bridge_gold_ids}
    assert bridge_titles == {"Person"}, (
        "'Capital' contains the answer 'Paris' - the intermediate is 'Person'"
    )
    comparison = by_id["comparison__abc2"]
    assert comparison.bridge_gold_ids == comparison.gold_ids, (
        "comparison questions have no intermediate document - bridge = gold"
    )


def test_2wiki_rejects_a_supporting_title_missing_from_context(
    tmp_path: Path,
) -> None:
    from spiyweb.evaluation.datasets import load_2wiki

    broken = [
        make_2wiki_record(
            "bad1",
            "compositional",
            "x",
            context=[("Only", ["Text."])],
            supporting_titles=["Ghost"],
        )
    ]
    with pytest.raises(ValueError, match="Ghost"):
        load_2wiki(write_2wiki(tmp_path, broken))


def test_2wiki_sampling_is_deterministic_over_prefixed_ids(tmp_path: Path) -> None:
    from spiyweb.evaluation.datasets import load_2wiki

    path = write_2wiki(tmp_path, TWOWIKI_RECORDS)
    config = EvaluationConfig(sample_size=1, sample_seed=42)
    first = load_2wiki(path, config)
    second = load_2wiki(path, config)
    assert [q.id for q in first.questions] == [q.id for q in second.questions]
    assert len(first.questions) == 1
