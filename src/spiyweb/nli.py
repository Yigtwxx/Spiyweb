"""Real NLI model wrapper: mDeBERTa XNLI, device order CUDA -> MPS -> CPU.

The concrete `NLIModel` behind `edges/nli.py`'s Protocol (open question #10,
owner's 2026-08-15 choice). It lives OUTSIDE `core/` - like `embedding.py`
and `llm.py`, the heavy dependency loads lazily on first construction, and
tests inject a fake scorer instead.

The wrapper never hardcodes the contradiction class index: NLI heads order
their labels differently across checkpoints, so the index is resolved from
the model's own `id2label` mapping - a wrong guess here would silently score
entailment as contradiction, which is the one failure mode this module must
make impossible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from spiyweb.config import NLIModelConfig
from spiyweb.embedding import detect_device

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class PairScorerLike(Protocol):
    """The single method a backend must offer: contradiction probability per
    (premise, hypothesis) pair, batched."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


def contradiction_label_index(id2label: Mapping[int, str]) -> int:
    """Locate the contradiction class in a model's own label mapping.

    Matching is case-insensitive on the substring "contradiction", so both
    plain (`"contradiction"`) and prefixed (`"CONTRADICTION"`, `"LABEL_2:
    contradiction"`) conventions resolve.

    Raises:
        ValueError: When no label (or more than one) mentions contradiction -
            either way the head is not a standard 3-way NLI classifier and
            guessing would corrupt every score downstream.
    """
    matches = [
        index
        for index, label in id2label.items()
        if "contradiction" in str(label).casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cannot locate the contradiction class in id2label {dict(id2label)!r}; "
            "expected exactly one label mentioning 'contradiction'"
        )
    return matches[0]


def batched(items: Sequence[tuple[str, str]], size: int) -> list[list[tuple[str, str]]]:
    """Split `items` into consecutive batches of at most `size` pairs."""
    if size < 1:
        raise ValueError("batch size must be at least 1")
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


class _TransformersPairScorer:
    """The real backend: a Hugging Face sequence-classification NLI model."""

    def __init__(self, config: NLIModelConfig) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:
            raise ImportError(
                "torch and transformers are required for the real NLI model; "
                "install them with `pip install spiyweb[nli]`"
            ) from error
        self._torch = torch
        self._config = config
        self._device = config.device if config.device is not None else detect_device()
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        with torch.no_grad():
            self._model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name
            )
        self._model.to(self._device)
        self._model.eval()
        self._contradiction = contradiction_label_index(self._model.config.id2label)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for batch in batched(pairs, self._config.batch_size):
            encoded = self._tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                truncation=True,
                padding=True,
                max_length=self._config.max_length,
                return_tensors="pt",
            ).to(self._device)
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
            probabilities = self._torch.softmax(logits, dim=-1)
            scores.extend(
                float(value) for value in probabilities[:, self._contradiction]
            )
        return scores


class TransformersNLIModel:
    """`NLIModel` implementation over any `PairScorerLike` backend.

    Without an injected scorer, the transformers dependency loads lazily on
    first construction (config's model, device CUDA -> MPS -> CPU). The
    class exists so `build_nli_edges` and the index stage depend on the
    Protocol alone - swapping the model is a config edit, never a code edit.
    """

    def __init__(
        self,
        config: NLIModelConfig | None = None,
        scorer: PairScorerLike | None = None,
    ) -> None:
        cfg = config if config is not None else NLIModelConfig()
        self._scorer = scorer if scorer is not None else _TransformersPairScorer(cfg)

    def contradiction_scores(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        """Contradiction confidence in [0, 1] per (premise, hypothesis) pair."""
        if not pairs:
            return []
        scores = self._scorer.score_pairs(pairs)
        if len(scores) != len(pairs):
            raise ValueError(
                f"backend returned {len(scores)} scores for {len(pairs)} pairs"
            )
        return scores
