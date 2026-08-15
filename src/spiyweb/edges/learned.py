"""Hebbian usage reinforcement in a separate, disableable edge layer.

The spider thickens the thread it uses and lets the unused one thin out:
edges that carried energy in a propagation run gain strength, everything
else ages. The learning lives in its OWN layer (the fourth `EdgeLayer`,
"learned") and the base graph is never mutated - that keeps the bias
reversible (delete the layer, not the index), measurable
(`LayerWeights.learned = 0.0` is the read-side ablation) and isolated
(popularity drift can at worst spoil one layer).

Unlike the stateless builders in this package, the layer is a stateful
accumulator; like them, it never calls a model, never touches I/O, and
emits a raw `(u, v, weight)` list for `Graph.from_layers` - persistence is
the caller's via `to_dict` / `from_dict`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spiyweb.config import LearnedLayerConfig

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from spiyweb.core.propagate import PropagationResult


class LearnedLayer:
    """Accumulates usage strength per undirected edge across runs.

    One reinforcement round (owner's 2026-08-14 decisions):

    1. Every stored edge is aged: `strength *= forgetting` (mandatory - the
       layer must keep forgetting or it collapses onto popular regions).
    2. Every edge that carried energy in `result` - each (contributor, node)
       pair of the activations - gains `learning_rate * min(1.0, carried /
       injected_energy)`, capped at `max_strength`: energy-proportional with
       saturation, so strong paths thicken faster but never without bound.

    `carried` is approximated as the node's accumulated energy split equally
    over its contributors - the core deliberately does not record per-edge
    flows, and the approximation errs toward evenness rather than inventing
    precision. Seeds have no contributors and produce no edges.

    Colored runs: reinforce each `per_color` result separately - each colour
    is its own propagation and its own usage evidence.
    """

    def __init__(self, config: LearnedLayerConfig | None = None) -> None:
        self._config = config if config is not None else LearnedLayerConfig()
        self._strength: dict[tuple[str, str], float] = {}

    def reinforce(
        self,
        result: PropagationResult,
        *,
        accepted: Collection[str] | None = None,
    ) -> int:
        """Age the layer, then strengthen the edges `result` actually used.

        `accepted` narrows the evidence to edges whose TARGET node the caller
        confirmed useful (e.g. passages the LLM cited); `None` reinforces
        every energy-carrying edge. Returns the number of edges strengthened;
        a disabled layer returns 0 and touches nothing, ageing included.
        """
        config = self._config
        if not config.enabled:
            return 0
        for key in self._strength:
            self._strength[key] *= config.forgetting

        injected = result.injected_energy
        touched = 0
        for node, activation in result.activations.items():
            if accepted is not None and node not in accepted:
                continue
            contributors = [
                feeder for feeder in activation.contributors if feeder != node
            ]
            if not contributors or injected <= 0.0:
                continue
            carried = activation.energy / len(contributors)
            increment = config.learning_rate * min(1.0, carried / injected)
            if increment <= 0.0:
                continue
            for feeder in contributors:
                key = (feeder, node) if feeder < node else (node, feeder)
                current = self._strength.get(key, 0.0)
                self._strength[key] = min(config.max_strength, current + increment)
                touched += 1
        return touched

    def edges(self) -> list[tuple[str, str, float]]:
        """The layer as a raw edge list for `Graph.from_layers` ("learned")."""
        return [
            (source, target, strength)
            for (source, target), strength in sorted(self._strength.items())
            if strength > 0.0
        ]

    def prune(self, min_strength: float) -> int:
        """Drop edges below `min_strength` (the consolidation hook, D23).

        Returns how many edges were removed. Pruning is the periodic offline
        counterpart of per-round forgetting: ageing thins, pruning discards.
        """
        doomed = [
            key for key, strength in self._strength.items() if strength < min_strength
        ]
        for key in doomed:
            del self._strength[key]
        return len(doomed)

    def to_dict(self) -> dict[str, float]:
        """Serialisable snapshot; keys are `"u\\tv"` with `u < v`."""
        return {
            f"{source}\t{target}": strength
            for (source, target), strength in sorted(self._strength.items())
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, float],
        config: LearnedLayerConfig | None = None,
    ) -> LearnedLayer:
        """Rebuild a layer from a `to_dict` snapshot."""
        layer = cls(config)
        for key, strength in payload.items():
            source, separator, target = key.partition("\t")
            if not separator or not source or not target:
                raise ValueError(f"malformed learned-layer key: {key!r}")
            layer._strength[(source, target)] = float(strength)
        return layer
