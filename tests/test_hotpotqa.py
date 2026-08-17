"""HotpotQA loader: the untouched third dataset maps like its 2Wiki copy.

The claim under test: `load_hotpotqa` produces type-prefixed ids, pools and
deduplicates context paragraphs, and applies the documented bridge
approximation - comparison and yes/no questions degenerate to all gold, a
bridge question's intermediates are the supporting paragraphs not containing
the answer string. HotpotQA invented the format 2Wiki copied, so the shared
loader body must behave identically on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spiyweb.evaluation.datasets import HOTPOTQA_DEV_URL, load_hotpotqa

BRIDGE_RECORD = {
    "_id": "q1",
    "type": "bridge",
    "question": "Who founded the studio that released Dawnlight?",
    "answer": "Ada Lovel",
    "context": [
        ["Dawnlight", ["Dawnlight is a film released by Orchard Studio."]],
        ["Orchard Studio", ["Orchard Studio was founded by Ada Lovel."]],
        ["Distractor", ["An unrelated paragraph about weather."]],
    ],
    "supporting_facts": [["Dawnlight", 0], ["Orchard Studio", 0]],
}

COMPARISON_RECORD = {
    "_id": "q2",
    "type": "comparison",
    "question": "Were Dawnlight and Duskfall released by the same studio?",
    "answer": "yes",
    "context": [
        ["Dawnlight", ["Dawnlight is a film released by Orchard Studio."]],
        ["Duskfall", ["Duskfall is a film released by Orchard Studio."]],
    ],
    "supporting_facts": [["Dawnlight", 0], ["Duskfall", 0]],
}


def _write(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    target = tmp_path / "hotpot_dev_distractor.json"
    target.write_text(json.dumps(records), encoding="utf-8")
    return target


def test_download_url_is_https() -> None:
    assert HOTPOTQA_DEV_URL.startswith("https://"), (
        "the fetcher rejects non-https urls; the canonical CMU host is http"
    )


def test_ids_carry_the_type_prefix_and_hops_count_gold_titles(
    tmp_path: Path,
) -> None:
    dataset = load_hotpotqa(_write(tmp_path, [BRIDGE_RECORD, COMPARISON_RECORD]))
    ids = sorted(question.id for question in dataset.questions)
    assert ids == ["bridge__q1", "comparison__q2"], (
        "per-type analysis reads the prefix straight from per_query ids"
    )
    for question in dataset.questions:
        assert question.hops == 2, "hotpot always has two supporting titles"


def test_shared_paragraphs_pool_to_one_node(tmp_path: Path) -> None:
    dataset = load_hotpotqa(_write(tmp_path, [BRIDGE_RECORD, COMPARISON_RECORD]))
    # "Dawnlight" appears in both questions' contexts with identical text:
    # 3 unique paragraphs from q1 + 1 new one from q2.
    assert len(dataset.documents) == 4, (
        f"expected 4 pooled paragraphs, got {len(dataset.documents)}"
    )


def test_bridge_question_intermediate_excludes_the_answer_holder(
    tmp_path: Path,
) -> None:
    dataset = load_hotpotqa(_write(tmp_path, [BRIDGE_RECORD]))
    question = dataset.questions[0]
    assert len(question.bridge_gold_ids) == 1, "one intermediate document"
    bridge_title = dataset.titles[question.bridge_gold_ids[0]]
    assert bridge_title == "Dawnlight", (
        "the paragraph containing the answer 'Ada Lovel' is the endpoint, "
        "not the bridge"
    )


def test_comparison_question_bridge_degenerates_to_all_gold(
    tmp_path: Path,
) -> None:
    dataset = load_hotpotqa(_write(tmp_path, [COMPARISON_RECORD]))
    question = dataset.questions[0]
    assert question.bridge_gold_ids == question.gold_ids, (
        "comparison/yes-no questions have no intermediate document (documented "
        "approximation, same rule as 2Wiki)"
    )


def test_a_non_array_file_is_a_hard_error(tmp_path: Path) -> None:
    target = tmp_path / "hotpot_dev_distractor.json"
    target.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="HotpotQA"):
        load_hotpotqa(target)
