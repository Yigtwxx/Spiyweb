"""Deterministic disk cache around any LLM client.

The first measurement must be reproducible and crash-resumable: with
temperature 0 the only remaining nondeterminism in the LLM paths (entity
fallback at index time, the iterative baseline's ~thousands of rewrite calls)
is *whether the call happens at all*. This wrapper makes every prompt's
response a one-time cost: keyed by the SHA-256 of the prompt, appended to a
JSONL file, replayed on every later run. The cache file is an experiment
artifact - the receipt (`meta.json`) records its location.

Never keyed by anything but the prompt content: same prompt, same answer, on
any machine that has the file.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from spiyweb.llm import LLMClient


class CachedLLMClient:
    """Wraps an `LLMClient`; satisfies the same protocol structurally."""

    def __init__(self, inner: LLMClient, path: Path) -> None:
        self._inner = inner
        self._path = path
        self._responses: dict[str, str] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self._responses[record["key"]] = record["response"]

    def complete(self, prompt: str) -> str:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = self._responses.get(key)
        if cached is not None:
            return cached
        response = self._inner.complete(prompt)
        self._responses[key] = response
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"key": key, "response": response}, ensure_ascii=False)
                + "\n"
            )
        return response
