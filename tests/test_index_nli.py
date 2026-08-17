"""Index-time NLI stage: candidates, artifact, reload, layer choice.

The claim under test: with an injected NLI model the index gains an
`edges_nli.json` artifact of contradiction edges over its high-cosine pairs -
paired within the proposition layer when the index has one, chunks otherwise -
`load_nli_edges` round-trips it, an index built without the stage loads as
"no conflicts" (not an error), and the stage resumes like every other stage.
"""

from __future__ import annotations

import json
from pathlib import Path

from spiyweb import NLICandidateConfig
from spiyweb.evaluation.datasets import MusiqueDataset
from spiyweb.evaluation.index import IndexPaths, build_index, load_nli_edges
from spiyweb.nodes import DocumentInput, TextUnit

DATASET = MusiqueDataset(
    documents=(
        DocumentInput(
            source_id="d0", units=(TextUnit(text="The lab opened in 1901."),)
        ),
        DocumentInput(source_id="d1", units=(TextUnit(text="The lab never opened."),)),
    ),
    titles={"d0:0": "Lab A", "d1:0": "Lab B"},
    texts={"d0:0": "The lab opened in 1901.", "d1:0": "The lab never opened."},
    questions=(),
)

# All pairs high-cosine: every text embeds to the same direction, so the
# candidate generator is exercised without hand-tuning vector geometry.
# The rarity cut is off here on purpose: in a two-passage corpus every name
# is corpus-wide by definition, so a document-frequency ratio would only
# measure the fixture. Its own behaviour is pinned in `test_nli_subject.py`.
CANDIDATES = NLICandidateConfig(top_k=2, min_similarity=0.0, max_subject_df_ratio=1.0)


class ConstantEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class EmptyPipeline:
    def pipe(self, texts: list[str]) -> list[object]:
        class Doc:
            ents: tuple = ()

        return [Doc() for _ in texts]


class SubjectPipeline:
    """Finds the subject both texts lead with, so the pair can be scored.

    The shared-subject requirement (#10) ships ON, so a fixture without
    entities would silently measure "no candidates" instead of the stage.
    """

    def pipe(self, texts: list[str]) -> list[object]:
        class Span:
            text = "lab"
            label_ = "ORG"

        class Doc:
            ents = (Span(),)

        return [Doc() for _ in texts]


class MarkAllNLI:
    def __init__(self, score: float = 0.95) -> None:
        self.score = score
        self.scored: list[tuple[str, str]] = []

    def contradiction_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.scored.extend(pairs)
        return [self.score] * len(pairs)


class FakeLLM:
    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies

    def complete(self, prompt: str) -> str:
        for needle, reply in self.replies.items():
            if needle in prompt:
                return reply
        return ""


def test_nli_stage_writes_and_reloads_negative_edges(tmp_path: Path) -> None:
    paths = IndexPaths(root=tmp_path / "idx")
    build_index(
        DATASET,
        paths,
        embedder=ConstantEmbedder(),
        entity_pipeline=SubjectPipeline(),
        nli_model=MarkAllNLI(),
        nli_model_name="fake-nli",
        nli_candidates=CANDIDATES,
        log=lambda _: None,
    )
    edges = load_nli_edges(paths)
    assert len(edges) == 1, "two chunks, one candidate pair, one marked edge"
    assert (edges[0].source, edges[0].target) == ("d0:0", "d1:0"), (
        "endpoints are stored in sorted order"
    )
    assert edges[0].strength == 0.95

    meta = json.loads(paths.meta_json.read_text(encoding="utf-8"))
    assert meta["nli_model"] == "fake-nli", "the receipt names the model"
    assert meta["nli_edges"] == 1


def test_nli_pairs_within_the_proposition_layer_when_present(
    tmp_path: Path,
) -> None:
    paths = IndexPaths(root=tmp_path / "idx")
    llm = FakeLLM(
        {
            "1901": "The laboratory opened its doors in 1901.",
            "never": "The laboratory never opened at all.",
        }
    )
    model = MarkAllNLI()
    build_index(
        DATASET,
        paths,
        embedder=ConstantEmbedder(),
        entity_pipeline=SubjectPipeline(),
        llm=llm,
        entity_llm=False,
        propositions=True,
        nli_model=model,
        nli_candidates=CANDIDATES,
        log=lambda _: None,
    )
    edges = load_nli_edges(paths)
    assert {edges[0].source, edges[0].target} == {"d0:0#p0", "d1:0#p0"}, (
        "contradiction is sharp on propositions; chunks must not be paired "
        "once the layer exists (D26)"
    )
    assert model.scored, "the model actually ran on the proposition pair"


def test_nli_stage_resumes_from_the_artifact(tmp_path: Path) -> None:
    paths = IndexPaths(root=tmp_path / "idx")
    common = {
        "embedder": ConstantEmbedder(),
        "entity_pipeline": SubjectPipeline(),
        "nli_candidates": CANDIDATES,
    }
    build_index(DATASET, paths, nli_model=MarkAllNLI(), log=lambda _: None, **common)
    second = MarkAllNLI()
    build_index(DATASET, paths, nli_model=second, log=lambda _: None, **common)
    assert second.scored == [], "an existing artifact skips the model entirely"


def test_the_stage_scores_nothing_when_no_pair_shares_a_subject(
    tmp_path: Path,
) -> None:
    # The audit's error pattern in miniature: two texts that look alike to
    # cosine and share no subject. Without entities nothing can qualify, so
    # the model must never be asked (#10).
    paths = IndexPaths(root=tmp_path / "idx")
    model = MarkAllNLI()
    build_index(
        DATASET,
        paths,
        embedder=ConstantEmbedder(),
        entity_pipeline=EmptyPipeline(),
        nli_model=model,
        nli_candidates=CANDIDATES,
        log=lambda _: None,
    )
    assert model.scored == [], "a subject-less pair must not reach the model"
    assert load_nli_edges(paths) == [], "and it must not become a negative edge"


def test_the_subject_requirement_can_be_switched_off(tmp_path: Path) -> None:
    paths = IndexPaths(root=tmp_path / "idx")
    model = MarkAllNLI()
    build_index(
        DATASET,
        paths,
        embedder=ConstantEmbedder(),
        entity_pipeline=EmptyPipeline(),
        nli_model=model,
        nli_candidates=NLICandidateConfig(
            top_k=2, min_similarity=0.0, require_shared_subject=False
        ),
        log=lambda _: None,
    )
    assert len(load_nli_edges(paths)) == 1, (
        "the ablation switch must restore the pre-#10 candidate pool"
    )


def test_an_index_without_the_stage_has_no_conflicts(tmp_path: Path) -> None:
    paths = IndexPaths(root=tmp_path / "idx")
    build_index(
        DATASET,
        paths,
        embedder=ConstantEmbedder(),
        entity_pipeline=EmptyPipeline(),
        log=lambda _: None,
    )
    assert not paths.nli_json.exists(), "no model, no artifact"
    assert load_nli_edges(paths) == [], "absence means 'no conflicts', not error"
