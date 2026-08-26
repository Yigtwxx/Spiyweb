"""Does the proposition layer actually recover same-passage contradictions?

Phase 1 measured the shipped detector against WikiContradict's 253 annotated
pairs and found 31.6% caught overall and **0% of the 63 same-passage ones**.
The explanation written down at the time was structural: edges run between
nodes, both sides of a same-passage contradiction live in one chunk, so no
pair exists to score - and therefore *the proposition layer is required*.

That last clause is an inference, not a measurement. This module tests it.

The test is free, and that is the point. Proposition extraction costs an LLM
call per passage, but propositions are derived FROM sentences, so splitting a
passage into sentences is a **lower bound on what any proposition extractor
can deliver**: two claims inside one sentence cannot be separated by either.
Measuring the sentence-level ceiling therefore answers "can this class of fix
work at all" before anyone spends a night on extraction.

Three numbers per bucket, in the order they gate each other:

1. **Structural ceiling** - pairs whose two annotated answers trace to
   DIFFERENT sentences of the passage. Nothing downstream can exceed this.
2. **Detection** - of those, how many the NLI model marks as a contradiction
   at the shipped threshold. No tuning; the constant is the shipped one.
3. **Survival** - how many then pass the shared-subject candidate filter,
   which is what the real pipeline would apply.

The two buckets are gated differently on purpose. A `Different` case is two
passages, so the shipped pipeline ALREADY has two nodes and the pair already
exists - its ceiling is the bucket itself, and it runs as the control that
must reproduce Phase 1's 31.6%. A control that misses is a broken harness,
not a finding. Only the `Same` bucket needs the sentence split, because only
there is the pair missing.

**Measured 2026-08-26, and the answer is that this dataset cannot answer it.**
Of the 63 same-passage cases: **21 (33.3%) are a single sentence**, where no
extractor of any kind can separate the two claims; **30 (47.6%) answer
yes/no**, which is a reasoning output rather than a span and cannot be traced
to a sentence at all; 4 more share no content word with their passage, and 1
puts both answers in one sentence. **Seven remain traceable.** A detection
rate over seven cases is not a number, so this module reports the breakdown
and refuses to compute one.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from spiyweb.config import NLIEdgeConfig
    from spiyweb.edges.nli import NLIModel

__all__ = [
    "BucketReport",
    "ContradictionCase",
    "SensitivityReport",
    "load_wikicontradict",
    "measure_sensitivity",
    "split_sentences",
    "trace_answers",
]

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")
"""Split on terminal punctuation followed by whitespace and something that
starts a sentence. Deliberately crude and deliberately dependency-free: a
smarter splitter would make the ceiling look better without making any real
extractor better, and the ceiling is the number under test."""

_ABBREVIATIONS = (
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "St.",
    "Jr.",
    "Sr.",
    "vs.",
    "e.g.",
    "i.e.",
    "No.",
    "Fig.",
)
"""Trailing dots that end a word rather than a sentence. Not exhaustive and
not meant to be: an abbreviation that slips through splits one sentence into
two, which can only INFLATE the ceiling - so the bound stays honest in the
direction that matters."""

_DEBRIS = re.compile(r"[A-Za-z0-9]")
"""A fragment with no letter or digit in it is punctuation debris, and gets
merged back into the sentence before it rather than dropped.

This used to be a minimum LENGTH, and that was a real defect: at fifteen
characters it swallowed "He left." - a whole sentence - into its neighbour.
Over-merging lowers the sentence count, which LOWERS the structural ceiling
this module exists to measure, so the bug pushed the answer in the direction
that would have confirmed the hypothesis. Caught by a test on 2026-08-26
before any number was reported."""

_MIN_TOKEN_CHARS = 4
"""Shortest word an answer contributes to the overlap test. Below this the
token is a preposition or an article and matches everywhere."""


@dataclass(frozen=True)
class ContradictionCase:
    """One annotated pair from WikiContradict."""

    question_id: str
    question: str
    context_a: str
    context_b: str
    answer_a: str
    answer_b: str
    same_passage: bool
    kind: str
    title: str

    @property
    def passage(self) -> str:
        """The text a chunk-level index would hold for this case.

        For a same-passage case the two contexts are one passage (60 of the
        63 are literally identical strings), which is exactly why the shipped
        detector cannot see it: one chunk, one node, no pair.
        """
        if self.same_passage and self.context_a.strip() == self.context_b.strip():
            return self.context_a.strip()
        return f"{self.context_a.strip()}\n{self.context_b.strip()}"


@dataclass(frozen=True)
class BucketReport:
    """What happened to one bucket of cases, gate by gate."""

    name: str
    total: int
    ceiling: int
    detected: int
    survived: int | None
    """How many survived the shared-subject filter, or `None` when it could
    not be run. `None` and not `0`: the filter needs an entity registry, and
    without one it drops everything - reporting that as a measured zero would
    be a harness artefact wearing a finding's clothes."""

    @property
    def ceiling_rate(self) -> float:
        return self.ceiling / self.total if self.total else 0.0

    @property
    def detection_rate(self) -> float:
        """Detected over TOTAL, so it compares with Phase 1's number."""
        return self.detected / self.total if self.total else 0.0

    @property
    def end_to_end_rate(self) -> float | None:
        """`None` when the subject filter was not run - see `survived`."""
        if self.survived is None or not self.total:
            return None
        return self.survived / self.total


@dataclass(frozen=True)
class SensitivityReport:
    same_passage: BucketReport
    different_passage: BucketReport
    threshold: float

    @property
    def overall_detection_rate(self) -> float:
        total = self.same_passage.total + self.different_passage.total
        found = self.same_passage.detected + self.different_passage.detected
        return found / total if total else 0.0


def load_wikicontradict(path: Path | str) -> list[ContradictionCase]:
    """Read the 253 annotated pairs. No network, no download step."""
    from pathlib import Path as _Path

    target = _Path(path)
    if target.is_dir():
        target = target / "wikicontradict.csv"
    with target.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        ContradictionCase(
            question_id=str(row["question_ID"]),
            question=row["question"],
            context_a=row["context1"],
            context_b=row["context2"],
            answer_a=row["answer1"],
            answer_b=row["answer2"],
            same_passage=row["samepassage"].strip().lower().startswith("same"),
            kind=row["contradictType"],
            title=row["WikipediaArticleTitle"],
        )
        for row in rows
    ]


def split_sentences(text: str) -> list[str]:
    """Split a passage into sentences, deterministically and without a model.

    The lower bound on proposition extraction: an extractor works from these,
    so two claims sharing a sentence cannot be separated by either.
    """
    guarded = text
    for abbreviation in _ABBREVIATIONS:
        guarded = re.sub(
            r"\b" + re.escape(abbreviation),
            abbreviation.replace(".", "\x00"),
            guarded,
        )
    parts = [part.strip() for part in _SENTENCE_END.split(guarded) if part.strip()]

    merged: list[str] = []
    for part in parts:
        restored = part.replace("\x00", ".")
        if merged and not _DEBRIS.search(restored):
            merged[-1] = f"{merged[-1]} {restored}"
        else:
            merged.append(restored)
    return merged


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(word) >= _MIN_TOKEN_CHARS
    }


def trace_answers(passage: str, answer_a: str, answer_b: str) -> tuple[int, int] | None:
    """Which sentences the two annotated answers came from, or `None`.

    Overlap on content words, not an exact match: the annotators wrote the
    answers as prose ("President Oler took the stand in 1911"), not as spans.
    A pair is only counted when the two answers land on DIFFERENT sentences
    and each has some real overlap - a tie means the evidence for separating
    them is not there, and counting it would inflate the ceiling this
    function exists to bound.
    """
    sentences = split_sentences(passage)
    if len(sentences) < 2:
        return None
    scored = [_tokens(sentence) for sentence in sentences]

    def best(answer: str) -> tuple[int, int]:
        wanted = _tokens(answer)
        if not wanted:
            return (-1, 0)
        overlaps = [len(wanted & sentence) for sentence in scored]
        top = max(overlaps)
        return (overlaps.index(top), top) if top else (-1, 0)

    index_a, _ = best(answer_a)
    index_b, _ = best(answer_b)
    if index_a < 0 or index_b < 0 or index_a == index_b:
        return None
    return (index_a, index_b)


def measure_sensitivity(
    cases: Sequence[ContradictionCase],
    model: NLIModel,
    *,
    config: NLIEdgeConfig | None = None,
    entities: Mapping[str, Sequence[str]] | None = None,
    require_shared_subject: bool = True,
) -> SensitivityReport:
    """Run the three gates over both buckets and report the counts.

    The model is injected, so the harness is testable with a fake and the
    real mDeBERTa run is the same code path.

    `entities` maps a case's two sides to the names an extractor found in
    them, keyed `"<question_id>:a"` / `"<question_id>:b"`. Without it the
    shared-subject gate cannot run - it qualifies a text only by a rare name
    the extractor found IN that text - and the report says `survived=None`
    rather than a zero somebody could mistake for a measurement.
    """
    from spiyweb.config import NLICandidateConfig, NLIEdgeConfig
    from spiyweb.edges.nli import shared_subject_pairs

    cfg = config if config is not None else NLIEdgeConfig()
    candidate_cfg = NLICandidateConfig()

    buckets: dict[bool, list[ContradictionCase]] = {True: [], False: []}
    for case in cases:
        buckets[case.same_passage].append(case)

    reports: dict[bool, BucketReport] = {}
    for same, group in buckets.items():
        traced: list[tuple[ContradictionCase, str, str]] = []
        for case in group:
            if not same:
                # Two passages: the shipped pipeline already sees two nodes,
                # so the pair exists and the ceiling is the bucket itself.
                # Splitting sentences here would measure a different, easier
                # question than the one the detector actually faces.
                traced.append((case, case.context_a.strip(), case.context_b.strip()))
                continue
            sentences = split_sentences(case.passage)
            found = trace_answers(case.passage, case.answer_a, case.answer_b)
            if found is None:
                continue
            traced.append((case, sentences[found[0]], sentences[found[1]]))

        directed: list[tuple[str, str]] = []
        for _, first, second in traced:
            directed.append((first, second))
            directed.append((second, first))
        scores = list(model.contradiction_scores(directed)) if directed else []

        detected: list[tuple[ContradictionCase, str, str]] = []
        for index, entry in enumerate(traced):
            strength = max(scores[2 * index], scores[2 * index + 1])
            if strength >= cfg.contradiction_threshold:
                detected.append(entry)

        survived: int | None = len(detected)
        if require_shared_subject:
            if entities is None:
                # Unmeasured, not zero. `shared_subject_pairs` qualifies a
                # text by a rare name the extractor found in it, so an empty
                # registry drops every pair - which says nothing about the
                # filter and everything about the caller.
                survived = None
            elif detected:
                texts: dict[str, str] = {}
                pairs: list[tuple[str, str]] = []
                for case, first, second in detected:
                    id_a, id_b = f"{case.question_id}:a", f"{case.question_id}:b"
                    texts[id_a], texts[id_b] = first, second
                    pairs.append((id_a, id_b))
                survived = len(
                    shared_subject_pairs(
                        pairs,
                        texts,
                        entities,
                        candidate_cfg.subject_prefix_chars,
                        candidate_cfg.max_subject_df_ratio,
                    )
                )

        reports[same] = BucketReport(
            name="same passage" if same else "different passages",
            total=len(group),
            ceiling=len(traced),
            detected=len(detected),
            survived=survived,
        )

    return SensitivityReport(
        same_passage=reports[True],
        different_passage=reports[False],
        threshold=cfg.contradiction_threshold,
    )
