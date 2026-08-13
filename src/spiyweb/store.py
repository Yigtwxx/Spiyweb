"""Single-file vector store: numpy + FAISS, exact search, outside core/.

The index is a `faiss.IndexFlatIP` - exact inner product, deterministic, no
approximation at Phase 1 corpus scale. With L2-normalised vectors (the
embedder's contract, not enforced here - the store is metric-agnostic) inner
product equals cosine similarity.

Persistence keeps ONE source of truth: `save` writes ids + vectors to a single
compressed `.npz` file and `load` rebuilds the FAISS index from the vectors.
Serialised FAISS indexes are a version/platform compatibility liability, and a
flat-index rebuild is a single O(n*d) add.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

try:
    import faiss
    import numpy as np
except ImportError as error:
    raise ImportError(
        "numpy and faiss are required for the vector store; "
        "install them with `pip install spiyweb[store]`"
    ) from error

if TYPE_CHECKING:
    from collections.abc import Sequence


class VectorStore:
    """Exact-search vector store mapping string ids to embeddings."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        self._dimension = dimension
        self._ids: list[str] = []
        self._known: set[str] = set()
        self._index = faiss.IndexFlatIP(dimension)

    @property
    def dimension(self) -> int:
        """Dimensionality every stored and queried vector must match."""
        return self._dimension

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        """Add vectors under their ids; ids are permanent and unique."""
        if len(ids) != len(vectors):
            raise ValueError(
                f"got {len(ids)} ids for {len(vectors)} vectors; "
                "they must map one-to-one"
            )
        if not ids:
            return
        incoming: set[str] = set()
        for chunk_id in ids:
            if not chunk_id:
                raise ValueError("vector id must not be empty")
            if chunk_id in self._known or chunk_id in incoming:
                raise ValueError(f"duplicate vector id {chunk_id!r}")
            incoming.add(chunk_id)
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dimension:
            raise ValueError(
                f"vectors have shape {matrix.shape}; expected (n, {self._dimension})"
            )
        self._index.add(matrix)
        self._ids.extend(ids)
        self._known.update(incoming)

    def search(self, query: Sequence[float], k: int) -> list[tuple[str, float]]:
        """Return up to `k` (id, inner product) pairs, best first."""
        if k < 1:
            raise ValueError("k must be at least 1")
        row = np.ascontiguousarray([query], dtype=np.float32)
        if row.shape[1] != self._dimension:
            raise ValueError(
                f"query has dimension {row.shape[1]}; expected {self._dimension}"
            )
        if not self._ids:
            return []
        scores, positions = self._index.search(row, min(k, len(self._ids)))
        return [
            (self._ids[position], float(score))
            for position, score in zip(positions[0], scores[0], strict=True)
            if position >= 0
        ]

    def save(self, path: str | Path) -> None:
        """Write ids and vectors to one compressed `.npz` file.

        The file handle form sidesteps numpy's suffix magic: the file lands at
        exactly `path`, so `load(path)` always finds what `save(path)` wrote.
        """
        vectors = self._index.reconstruct_n(0, len(self._ids))
        with Path(path).open("wb") as handle:
            np.savez_compressed(
                handle, ids=np.array(self._ids, dtype=np.str_), vectors=vectors
            )

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        """Rebuild a store (and its index) from a `save`d file."""
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"vector store file {str(resolved)!r} not found")
        with np.load(resolved) as payload:
            ids = [str(chunk_id) for chunk_id in payload["ids"]]
            vectors = payload["vectors"]
        store = cls(int(vectors.shape[1]))
        store.add(ids, vectors)
        return store
