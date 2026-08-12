"""Tunable parameters of the propagation core.

No magic numbers are allowed inside the algorithm modules: every knob lives here
as a documented dataclass field, so the developer UI can build its controls from
this object and every mechanism stays individually switchable for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PropagationConfig:
    """Settings for a single spreading-activation run.

    Attributes:
        seed_energy: Total energy injected into the graph by one query.
        damping: Fraction of its energy a node forwards to its neighbours; the
            rest stays behind. Decay is multiplicative, never subtractive, so a
            weak edge fades faster than a strong one.
        threshold_ratio: Stop condition, expressed relative to the injected
            energy rather than as an absolute number. Energy arriving at a node
            below `threshold_ratio * seed_energy` dies there. Relative because
            thermal memory and query profiles both change the injected total.
        max_hop: Hard overflow guard on propagation depth. The threshold is the
            real stop condition; this only prevents surprises. Provisional
            default, to be revisited after the first measurement.
        max_nodes: Hard overflow guard on the size of the activated set. Same
            status as `max_hop`: safety brake, not a `top-k` in disguise.
    """

    seed_energy: float = 10.0
    damping: float = 0.60
    threshold_ratio: float = 0.15
    max_hop: int = 6
    max_nodes: int = 512

    def __post_init__(self) -> None:
        if self.seed_energy <= 0.0:
            raise ValueError("seed_energy must be positive")
        if not 0.0 < self.damping < 1.0:
            raise ValueError("damping must lie strictly between 0 and 1")
        if not 0.0 <= self.threshold_ratio < 1.0:
            raise ValueError("threshold_ratio must lie in [0, 1)")
        if self.max_hop < 0:
            raise ValueError("max_hop must not be negative")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be at least 1")

    @property
    def threshold(self) -> float:
        """Absolute energy floor implied by `threshold_ratio` for this run."""
        return self.threshold_ratio * self.seed_energy
