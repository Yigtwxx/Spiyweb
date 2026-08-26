"""Embedding wrapper: e5 prefixes baked in, device order CUDA -> MPS -> CPU.

The e5 model family requires a role prefix on every input - `"query: "` for
questions, `"passage: "` for corpus text - and silently degrades without it.
The prefix contract is therefore baked into the API: there is no un-prefixed
encode method, so callers cannot forget it.

Vectors are always L2-normalised, which is what makes the store's inner
product equal cosine similarity - that contract lives here, not in the store.

This module imports cleanly with nothing installed: the pure `resolve_device`
and the Protocols need no torch. Only constructing a real
`SentenceTransformerEmbedder` without an injected model touches the optional
`embed` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from spiyweb.config import EmbeddingConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

Device = Literal["cuda", "mps", "cpu"]

_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


def resolve_device(*, cuda_available: bool, mps_available: bool) -> Device:
    """Pure device policy: CUDA -> MPS -> CPU, in that fixed order."""
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def detect_device() -> Device:
    """Probe torch for the available backends and apply `resolve_device`."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "torch is required for device detection; "
            "install it with `pip install spiyweb[embed]`"
        ) from error
    return resolve_device(
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
    )


class EncoderLike(Protocol):
    """The single sentence-transformers method the wrapper depends on."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> Sequence[Sequence[float]]: ...


class Embedder(Protocol):
    """Role-aware embedding interface the index pipeline depends on."""

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """e5-style embedder over any `EncoderLike` (a real model or a test fake).

    Without an injected model the sentence-transformers dependency loads
    lazily on first construction, on the device resolved by `detect_device`
    unless the config names one explicitly.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        model: EncoderLike | None = None,
    ) -> None:
        self._config = config if config is not None else EmbeddingConfig()
        self._model = model if model is not None else self._load_model()

    @property
    def model_name(self) -> str:
        """Which model produced these vectors - receipt data, not behaviour.

        `build_index` records it in the store so a query embedded by a
        different model can be refused instead of silently answered: two
        unrelated models can share a dimension, and cosine across two spaces
        returns confident nonsense.
        """
        return self._config.model

    def _load_model(self) -> EncoderLike:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "sentence-transformers is required for embedding; "
                "install it with `pip install spiyweb[embed]`"
            ) from error
        device = self._config.device
        if device is None:
            device = detect_device()
        return SentenceTransformer(self._config.model, device=device)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed question-side texts with the `"query: "` role prefix."""
        return self._encode([_QUERY_PREFIX + text for text in texts])

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus-side texts with the `"passage: "` role prefix."""
        return self._encode([_PASSAGE_PREFIX + text for text in texts])

    def _encode(self, prefixed: list[str]) -> list[list[float]]:
        rows = self._model.encode(
            prefixed,
            batch_size=self._config.batch_size,
            normalize_embeddings=True,
        )
        return [[float(value) for value in row] for row in rows]
