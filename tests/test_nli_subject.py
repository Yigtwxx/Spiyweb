"""Shared-subject candidate filter for index-time NLI (open question #10).

The claim under test: a candidate pair survives only when both texts NAME the
same subject - a rare-enough entity that both texts carry and that appears in
the leading (subject) region of both. This is the filter the 2026-08-16 audit
asked for: cosine pairs two same-kind, different-entity texts (two radio
stations, two villages), NLI silently assumes they share a subject, and the
resulting false contradictions score HIGHER than the one genuine
contradiction in the run, so no threshold can separate them.

The examples below are the audit's own pairs, shortened; the windows are the
shipped defaults, so a change to `NLICandidateConfig` breaks these tests
rather than quietly changing what gets scored.
"""

from __future__ import annotations

from spiyweb import NLICandidateConfig
from spiyweb.edges import shared_subject_pairs

WINDOW = NLICandidateConfig().subject_prefix_chars

# Two radio stations in two different Jacksons: cosine-similar, both mention
# "Jackson", neither is ABOUT Jackson. The audit's dominant error.
STATION_A = "WMTQ is a radio station licensed to Jackson, New Hampshire."
STATION_B = "WJXN-FM is a radio station licensed to Jackson, Mississippi."

# Two versions of one article: the genuine contradiction the same run found.
ASSEMBLY_A = "The National Assembly of Pakistan has 342 members in total."
ASSEMBLY_B = "The National Assembly of Pakistan is composed of 332 members."

TEXTS = {"a": STATION_A, "b": STATION_B, "c": ASSEMBLY_A, "d": ASSEMBLY_B}
ENTITIES = {
    "a": ["wmtq", "jackson", "new hampshire"],
    "b": ["wjxn-fm", "jackson", "mississippi"],
    "c": ["national assembly of pakistan"],
    "d": ["national assembly of pakistan"],
}


def test_a_shared_mention_outside_the_subject_region_does_not_qualify() -> None:
    kept = shared_subject_pairs([("a", "b")], TEXTS, ENTITIES, WINDOW)
    assert kept == [], (
        "both texts mention Jackson, neither is about it - this is exactly "
        "the pair class that produced .9995-confidence false contradictions"
    )


def test_a_shared_subject_keeps_the_pair() -> None:
    kept = shared_subject_pairs([("c", "d")], TEXTS, ENTITIES, WINDOW)
    assert kept == [("c", "d")], (
        "two texts about the same assembly can genuinely contradict; the "
        "filter must not cost the run its true positives"
    )


def test_the_window_is_what_separates_subject_from_object() -> None:
    # Widen the window past "Jackson" and the station pair qualifies again -
    # the known failure mode, pinned so a later change to the default is a
    # decision rather than a surprise.
    kept = shared_subject_pairs([("a", "b")], TEXTS, ENTITIES, 200)
    assert kept == [("a", "b")]


def test_a_corpus_wide_name_cannot_be_a_subject() -> None:
    # "Canadian" in the audit corpus: 1.4% of passages, and the only thing
    # keeping two Calgary radio stations paired.
    texts = {
        "a": "CFGQ-FM is a Canadian radio station in Calgary.",
        "b": "CKMP-FM is a Canadian radio station in Calgary.",
    }
    entities = {f"n{i}": ["canadian"] for i in range(100)}
    entities["a"] = ["cfgq-fm", "canadian"]
    entities["b"] = ["ckmp-fm", "canadian"]
    generous = shared_subject_pairs([("a", "b")], texts, entities, WINDOW, 1.0)
    assert generous == [("a", "b")], "without the rarity cut the leak is real"
    kept = shared_subject_pairs([("a", "b")], texts, entities, WINDOW, 0.005)
    assert kept == [], "a name 100 of 102 nodes carry marks a category"


def test_a_rare_name_survives_the_rarity_cut() -> None:
    entities = {**ENTITIES, **{f"n{i}": ["unrelated"] for i in range(500)}}
    kept = shared_subject_pairs([("c", "d")], TEXTS, entities, WINDOW, 0.005)
    assert kept == [("c", "d")], (
        "the cut must remove categories, not the subjects it exists to keep"
    )


def test_an_entity_absent_from_its_own_text_cannot_qualify() -> None:
    entities = {
        "c": ["national assembly of pakistan", "ghost entity"],
        "d": ["ghost entity"],
    }
    kept = shared_subject_pairs([("c", "d")], TEXTS, entities, WINDOW)
    assert kept == [], (
        "a name only the extractor believes in must not qualify a pair - the "
        "test is what the TEXT says, not what the entity list claims"
    )


def test_matching_ignores_case() -> None:
    kept = shared_subject_pairs(
        [("c", "d")], {"c": ASSEMBLY_A.upper(), "d": ASSEMBLY_B}, ENTITIES, WINDOW
    )
    assert kept == [("c", "d")]


def test_unknown_nodes_are_dropped_not_guessed() -> None:
    kept = shared_subject_pairs([("c", "zz")], TEXTS, ENTITIES, WINDOW)
    assert kept == [], "a node with no text cannot be judged, so it is not"


def test_candidate_order_survives_the_filter() -> None:
    # The caller sorts by similarity before capping; a filter that reorders
    # would silently change which pairs the cap keeps.
    kept = shared_subject_pairs(
        [("c", "d"), ("a", "b"), ("d", "c")], TEXTS, ENTITIES, WINDOW
    )
    assert kept == [("c", "d"), ("d", "c")]
