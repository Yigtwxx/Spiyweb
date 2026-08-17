"""Query profiles (D13): a profile is a knob package, not a config tree.

The claim under test: `Profile` overlays EXACTLY damping, threshold ratio
and seed width onto a base retrieval config - everything else the base
carries (the measured coloured winner's `split_alpha` and chain settings
above all) survives untouched - and the preset packages actually behave as
designed: the explore ball rolls further than the precise one.
"""

from __future__ import annotations

import pytest

from spiyweb import (
    COMPARE,
    EXPLORE,
    PRECISE,
    PROFILES,
    ColoredRetrievalConfig,
    Graph,
    Profile,
    RetrievalConfig,
    propagate,
)

# A bare chain - the simplest graph where reach depends on the profile.
# Long enough that damping, not the `max_hop` brake, is what stops the walk:
# eight hops is exactly the default cap, so the explore ball dies of old age
# rather than being cut off.
CHAIN = Graph.from_edges(
    [(chr(65 + i), chr(66 + i), 1.0) for i in range(8)]  # A-B, B-C, ... H-I
)


def test_as_retrieval_overrides_the_three_knobs_only() -> None:
    base = RetrievalConfig()
    config = PRECISE.as_retrieval(base)
    assert config.seed_width == 3
    assert config.propagation.damping == pytest.approx(0.45)
    assert config.propagation.threshold_ratio == pytest.approx(0.01)
    assert config.propagation.split_alpha == base.propagation.split_alpha
    assert config.propagation.max_hop == base.propagation.max_hop
    assert config.contact_overfetch == base.contact_overfetch, (
        "a profile is damping + threshold + seed width, nothing more"
    )


def test_as_colored_preserves_the_measured_winner() -> None:
    config = COMPARE.as_colored()
    winner = ColoredRetrievalConfig()
    assert config.propagation.split_alpha == winner.propagation.split_alpha
    assert config.chain_mode == winner.chain_mode
    assert config.decomposition_model == winner.decomposition_model
    assert config.max_colors == winner.max_colors, (
        "overlaying a profile must never silently discard the grid winner"
    )
    assert config.seed_width == COMPARE.seed_width
    assert config.propagation.damping == pytest.approx(COMPARE.damping)


def test_explore_ball_rolls_further_than_the_precise_one() -> None:
    # Both profiles now stop on the same threshold (.01 of the injected 1.0),
    # so the reach difference is damping alone - which is the point of the
    # 2026-08-16 re-derivation: threshold decides WHETHER the web leaves hop
    # 0, damping decides how far it then travels.
    # PRECISE .45: .45**5 = .0184 lives, .45**6 = .0083 dies -> A..F.
    # EXPLORE .75: .75**8 = .1001 still lives -> the whole chain.
    precise = propagate(CHAIN, {"A": 1.0}, PRECISE.as_retrieval().propagation)
    explore = propagate(CHAIN, {"A": 1.0}, EXPLORE.as_retrieval().propagation)
    assert len(explore.activations) > len(precise.activations), (
        "a fact lookup wants a small fast-dying ball, exploration a large one"
    )
    assert "I" in explore.activations, "the explore ball reaches the far end"
    assert "G" not in precise.activations, "the precise ball dies of damping"
    assert precise.stop_reason == "threshold", (
        "the brake must not be what stopped it - that would test max_hop"
    )


def test_the_presets_share_the_measured_threshold() -> None:
    # The 2026-08-16 measurement rejected the hand triple (.25/.05/.10):
    # at those ratios the activated set equalled seed_width, i.e. the web
    # never left hop 0. Equal thresholds are the finding, so pin them.
    assert {profile.threshold_ratio for profile in PROFILES.values()} == {0.01}


def test_profiles_map_carries_the_three_presets_by_name() -> None:
    assert set(PROFILES) == {"precise", "explore", "compare"}
    assert PROFILES["explore"] is EXPLORE


def test_profile_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="damping"):
        Profile(name="p", damping=1.0, threshold_ratio=0.1, seed_width=1)
    with pytest.raises(ValueError, match="threshold_ratio"):
        Profile(name="p", damping=0.5, threshold_ratio=1.0, seed_width=1)
    with pytest.raises(ValueError, match="seed_width"):
        Profile(name="p", damping=0.5, threshold_ratio=0.1, seed_width=0)
    with pytest.raises(ValueError, match="name"):
        Profile(name="", damping=0.5, threshold_ratio=0.1, seed_width=1)
