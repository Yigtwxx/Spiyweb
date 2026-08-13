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
class StructuralEdgeConfig:
    """Raw within-layer weights of the structural edge builder.

    These are relation weights INSIDE the structural layer; the layer's overall
    strength against other layers stays in `LayerWeights.structural`. The three
    relations are strictly nested (adjacent pairs are also same-section pairs,
    which are also same-document pairs), so they are not independent evidence:
    per pair the strongest enabled relation WINS - weights never sum. A weight
    of `0.0` disables that relation; a pair with no enabled relation is omitted
    entirely, never emitted at `0.0` (that value is reserved for
    dedup-suppressed edges). Defaults are provisional hand values, subject to
    the same grid search as the layer weights.

    Attributes:
        adjacent: Consecutive chunks of the same document in reading order.
        same_section: Chunks sharing a non-None section within one document.
        same_document: Any two chunks of one document. Off by default: it
            builds an O(n^2) clique per document, and proportional splitting
            turns dense cliques straight into the known hub penalty.
    """

    adjacent: float = 1.0
    same_section: float = 0.6
    same_document: float = 0.0

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            if value < 0.0:
                raise ValueError(
                    f"structural relation weight {spec.name}={value!r} "
                    "must not be negative"
                )


@dataclass(frozen=True)
class SemanticEdgeConfig:
    """Settings of the semantic (cosine kNN) edge builder.

    The semantic layer is deliberately weak - seed contact and fallback only -
    so a small `k` keeps it sparse. Defaults are provisional hand values,
    subject to the same grid search as the layer weights.

    Attributes:
        k: Neighbours considered per node. A pair is emitted once when either
            endpoint ranks the other in its top-k (union kNN).
        min_similarity: Emission floor; a pair needs `similarity > floor` to be
            emitted (strict, so a cosine of exactly 0.0 never leaks in as a
            fake suppressed edge). The non-negative bound on this floor is the
            mechanism that keeps negative cosine out of the graph, whose edge
            weights must never be negative.
    """

    k: int = 5
    min_similarity: float = 0.0

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if not 0.0 <= self.min_similarity < 1.0:
            raise ValueError("min_similarity must lie in [0, 1)")


@dataclass(frozen=True)
class EntityEdgeConfig:
    """Settings of the entity (shared entity / concept) edge builder.

    The edge weight itself carries no knob: for every entity shared by two
    chunks the pair gains `1 / df(entity)`, where `df` is the number of chunks
    mentioning that entity - a rare entity is strong evidence of a real link,
    a ubiquitous one is barely any. There is deliberately no `min_shared`
    threshold: the rarity sum already fades weak overlap, and a second cutoff
    would be a knob with no evidence behind it.

    Attributes:
        max_df_ratio: Entities mentioned by more than `max_df_ratio * n_chunks`
            chunks are dropped before any pair is built. The 1/df damping
            bounds a stopword entity's *weight* but not its *edge count* - an
            entity in 80% of n chunks still emits ~0.32*n^2 pairs, and dense
            cliques feed the known hub penalty (the same rationale that keeps
            `same_document` off by default). `1.0` disables the guard.
            Provisional hand value, subject to the same grid search as the
            layer weights.
    """

    max_df_ratio: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.max_df_ratio <= 1.0:
            raise ValueError("max_df_ratio must lie in (0, 1]")


@dataclass(frozen=True)
class EntityExtractionConfig:
    """Settings of the hybrid (spaCy + LLM) entity extraction pipeline.

    spaCy handles the bulk; only chunks where it finds fewer than
    `min_entities` entities are routed to the LLM. Passing no LLM client (or
    `min_entities=0`) disables the hybrid entirely - the mandatory ablation
    switch for the LLM path.

    Attributes:
        spacy_model: Pipeline name to load. The default is the multilingual
            WikiNER model (Turkish + English corpora are both in scope).
        labels: Entity labels kept after NER. Deliberately the UNION of the
            WikiNER scheme (PER/ORG/LOC/MISC) and the OntoNotes scheme used by
            the English models (PERSON/GPE/...), so swapping the model never
            silently drops every entity. Numeric and temporal labels (DATE,
            CARDINAL, PERCENT, ...) are excluded on purpose: a shared "2019"
            is not hop fuel, it floods document frequency. A custom set that
            matches neither scheme extracts nothing - and the LLM fallback
            would then mask the mistake, one paid call per chunk.
        min_entities: Chunks where spaCy yields fewer entities than this go to
            the LLM. Provisional hand value; the default of 1 sends only
            spaCy-blind chunks.
    """

    spacy_model: str = "xx_ent_wiki_sm"
    labels: frozenset[str] = frozenset(
        {
            "PER",
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "NORP",
            "FAC",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "MISC",
        }
    )
    min_entities: int = 1

    def __post_init__(self) -> None:
        if not self.spacy_model:
            raise ValueError("spacy_model must not be empty")
        if self.min_entities < 0:
            raise ValueError("min_entities must not be negative")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Settings of the embedding model wrapper (index time, outside core/).

    Attributes:
        model: Sentence-transformers model name. The default is the Phase 1
            decision: multilingual, Turkish + English both in scope.
        batch_size: Encoding batch size.
        device: Explicit device string, or `None` to auto-resolve in the
            fixed order CUDA -> MPS -> CPU.
    """

    model: str = "intfloat/multilingual-e5-large"
    batch_size: int = 32
    device: str | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass(frozen=True)
class LLMConfig:
    """Settings of the LLM provider used at index time (never inside core/).

    One OpenAI-compatible chat-completions code path covers the local-first
    default (Ollama) and the optional free APIs (Groq, OpenRouter, ...): they
    all speak the same protocol, only `base_url`, `model` and the key differ.

    Attributes:
        base_url: API root ending before `/chat/completions`. The default is
            Ollama's local OpenAI-compatible endpoint, which needs no key.
        model: Model name as the provider expects it.
        api_key_env: NAME of the environment variable holding the API key
            (e.g. "GROQ_API_KEY"), never the key itself - secrets stay out of
            source and out of every log. `None` sends no Authorization header.
        timeout_seconds: Per-request network timeout.
        temperature: Sampling temperature; extraction wants determinism, so
            the default is 0.
        max_tokens: Completion length cap per request.
        max_retries: Additional attempts after the first failed request.
        retry_backoff_seconds: Base of the exponential backoff between
            retries (`backoff * 2**attempt`).
    """

    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3.1:8b"
    api_key_env: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 512
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if self.temperature < 0.0:
            raise ValueError("temperature must not be negative")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.retry_backoff_seconds < 0.0:
            raise ValueError("retry_backoff_seconds must not be negative")


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
