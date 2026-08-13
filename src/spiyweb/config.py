"""Tunable parameters of the propagation core.

No magic numbers are allowed inside the algorithm modules: every knob lives here
as a documented dataclass field, so the developer UI can build its controls from
this object and every mechanism stays individually switchable for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

EdgeLayer = Literal["semantic", "entity", "structural", "learned"]
"""The four edge layers of the hybrid graph; layer choices live here, not in core."""


@dataclass(frozen=True)
class LayerWeights:
    """Relative strength of each edge layer when merging into one adjacency.

    These are THE home of the Phase 1 hand weights - no other module may
    restate them. A weight of `0.0` disables the layer entirely: its edges and
    any nodes only it mentions never enter the merged graph, which is what
    keeps every layer individually switchable for ablation.

    Attributes:
        semantic: Cosine-similarity edges; seed contact and fallback only.
            Hopping along paraphrases returns repetition, not new information.
        entity: Shared entity / concept edges - the main hop fuel.
        structural: Same document, same section, adjacent chunk.
        learned: Hebbian usage-reinforced edges. Disabled by default in
            Phase 1; the layer must never mutate the base graph.
    """

    semantic: float = 0.5
    entity: float = 1.0
    structural: float = 0.3
    learned: float = 0.0

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            if value < 0.0:
                raise ValueError(
                    f"layer weight {spec.name}={value!r} must not be negative"
                )

    def weight_of(self, layer: EdgeLayer) -> float:
        """Weight of `layer`; field names and layer names coincide by design."""
        return float(getattr(self, layer))


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
