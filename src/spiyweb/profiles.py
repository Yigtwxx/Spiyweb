"""Query profiles (D13) - precise / explore / compare knob packages.

One global damping cannot serve a fact lookup and an exploratory sweep at
once: a precise question wants a small, fast-dying energy ball, an
exploratory one wants a large, slow-dying ball. A profile is exactly the
"damping + threshold + seed width" package the design names, and this module
is a FACTORY over the existing retrieval configs, not a parallel config
tree: `as_retrieval` / `as_colored` overlay the three knobs onto a base
config and preserve everything else it carries.

The caller picks the profile - never an LLM, and never anything inside
`core/`. The preset values below are provisional hand defaults (open
question #6), to be measured.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from spiyweb.config import ColoredRetrievalConfig, RetrievalConfig


@dataclass(frozen=True)
class Profile:
    """One query profile: the D13 "damping + threshold + seed width" package.

    Applying a profile overrides EXACTLY these three knobs on a base config;
    every other field of the base (`split_alpha`, `max_hop`, `max_nodes`,
    `contact_overfetch`, the coloured chain settings, ...) survives
    untouched. Bounds mirror `PropagationConfig`.

    Attributes:
        name: Identifier of the profile (UI selectors key on it).
        damping: Fraction of its energy a node forwards; low = the ball dies
            near the seed, high = it rolls far.
        threshold_ratio: Relative stop threshold; high = early termination.
        seed_width: First-contact atoms. On the coloured path this is PER
            COLOUR - the measured winner uses 2 there, so a plain-path width
            applied via `as_colored` widens every colour's contact.
    """

    name: str
    damping: float
    threshold_ratio: float
    seed_width: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not 0.0 < self.damping < 1.0:
            raise ValueError("damping must lie strictly between 0 and 1")
        if not 0.0 <= self.threshold_ratio < 1.0:
            raise ValueError("threshold_ratio must lie in [0, 1)")
        if self.seed_width < 1:
            raise ValueError("seed_width must be at least 1")

    def as_retrieval(self, base: RetrievalConfig | None = None) -> RetrievalConfig:
        """The profile overlaid onto a plain retrieval config."""
        base = base if base is not None else RetrievalConfig()
        return replace(
            base,
            seed_width=self.seed_width,
            propagation=replace(
                base.propagation,
                damping=self.damping,
                threshold_ratio=self.threshold_ratio,
            ),
        )

    def as_colored(
        self, base: ColoredRetrievalConfig | None = None
    ) -> ColoredRetrievalConfig:
        """The profile overlaid onto a coloured retrieval config.

        The default base is `ColoredRetrievalConfig()`, which carries the
        measured 2026-08-14 grid winner - its `split_alpha`, chain settings
        and colour cap all survive the overlay. Only the three profile knobs
        change, and `seed_width` is per colour here (winner: 2).
        """
        base = base if base is not None else ColoredRetrievalConfig()
        return replace(
            base,
            seed_width=self.seed_width,
            propagation=replace(
                base.propagation,
                damping=self.damping,
                threshold_ratio=self.threshold_ratio,
            ),
        )


PRECISE = Profile(name="precise", damping=0.45, threshold_ratio=0.25, seed_width=3)
"""Small, fast-dying ball: forwards little, stops early, touches few atoms."""

EXPLORE = Profile(name="explore", damping=0.75, threshold_ratio=0.05, seed_width=8)
"""Large, slow-dying ball: forwards most of its energy and keeps rolling."""

COMPARE = Profile(name="compare", damping=0.60, threshold_ratio=0.10, seed_width=6)
"""Two-sided question: wide enough contact to hold both compared regions.

The natural pairing is the coloured multi-seed path (each side its own
colour, the bridge is the answer); applied there via `as_colored`, remember
the width is per colour.
"""

PROFILES: dict[str, Profile] = {
    profile.name: profile for profile in (PRECISE, EXPLORE, COMPARE)
}
"""Name -> profile map for UI selectors and config files."""
