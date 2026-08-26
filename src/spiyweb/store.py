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

from spiyweb.config import SemanticEdgeConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

_STORE_SCHEMA_VERSION = 1
"""Version of the `.npz` layout. Bumped only when a reader must behave
differently; a file without it reads as 0 and loads unchanged."""


def build_semantic_edges_fast(
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    config: SemanticEdgeConfig | None = None,
) -> list[tuple[str, str, float]]:
    """FAISS-backed twin of `spiyweb.edges.semantic.build_semantic_edges`.

    Same semantics as the pure builder - union kNN, strict
    `similarity > min_similarity`, canonical `u < v` pairs, sorted output -
    but neighbour selection runs through a throwaway `IndexFlatIP`, which is
    what makes a corpus-scale semantic layer feasible (the pure builder is
    O(n^2 * d) Python and exists as the semantics oracle, not the workhorse).
    Input is L2-normalised internally, exactly like the pure builder, so
    arbitrary callers get cosine without a pre-normalisation trap. Emitted
    edge weights are recomputed as float64 cosines; only the neighbour
    *selection* runs in FAISS's float32.

    It lives here and not in `edges/` because `edges/` eagerly imports every
    builder module and must stay importable with zero dependencies; this
    module already owns the faiss/numpy import and its install hint.

    Known divergence from the pure builder: on EXACTLY equal similarities
    FAISS breaks ties by insertion order while the pure builder breaks them
    by id. Real embeddings make that a measure-zero event.
    """
    cfg = config if config is not None else SemanticEdgeConfig()

    if len(ids) != len(embeddings):
        raise ValueError(
            f"got {len(ids)} ids but {len(embeddings)} embeddings; "
            "they must pair up one-to-one"
        )
    seen: set[str] = set()
    for node_id in ids:
        if node_id in seen:
            raise ValueError(f"duplicate node id {node_id!r}")
        seen.add(node_id)
    if len(ids) < 2:
        return []

    dimension = len(embeddings[0])
    for node_id, vector in zip(ids, embeddings, strict=True):
        if len(vector) != dimension:
            raise ValueError(
                f"embedding of node {node_id!r} has dimension {len(vector)}; "
                f"expected {dimension}"
            )

    matrix = np.ascontiguousarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    zero_rows = np.flatnonzero(norms == 0.0)
    if zero_rows.size:
        raise ValueError(
            f"embedding of node {ids[int(zero_rows[0])]!r} has zero norm; "
            "a real embedding model never emits one"
        )
    unit = matrix / norms[:, np.newaxis]

    index = faiss.IndexFlatIP(dimension)
    unit32 = np.ascontiguousarray(unit, dtype=np.float32)
    index.add(unit32)
    # k+1 because every vector's own row ranks itself first (or among exact
    # duplicates); self is dropped below, leaving the k true neighbours.
    _, positions = index.search(unit32, min(cfg.k + 1, len(ids)))

    emitted: set[tuple[str, str]] = set()
    for i, row in enumerate(positions):
        neighbours = [int(p) for p in row if p >= 0 and p != i][: cfg.k]
        for j in neighbours:
            u, v = ids[i], ids[j]
            emitted.add((u, v) if u < v else (v, u))

    position_of = {node_id: i for i, node_id in enumerate(ids)}
    edges: list[tuple[str, str, float]] = []
    for u, v in sorted(emitted):
        similarity = float(unit[position_of[u]] @ unit[position_of[v]])
        if similarity > cfg.min_similarity:
            edges.append((u, v, similarity))
    return edges


class VectorStore:
    """Exact-search vector store mapping string ids to embeddings."""

    def __init__(self, dimension: int, *, model_name: str | None = None) -> None:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        self._dimension = dimension
        self._ids: list[str] = []
        self._known: set[str] = set()
        self._model_name = model_name
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

    @property
    def model_name(self) -> str | None:
        """Which embedding model produced these vectors, when recorded.

        `None` for a store built in memory and for every file written before
        2026-08-25 - including the four sealed Phase 1 indexes. Absence means
        "unknown", never "mismatch": a check that cannot be made is not a
        failure, and `SpiywebIndex` treats it that way.
        """
        return self._model_name

    def save(self, path: str | Path, *, model_name: str | None = None) -> None:
        """Write ids and vectors to one compressed `.npz` file.

        The file handle form sidesteps numpy's suffix magic: the file lands at
        exactly `path`, so `load(path)` always finds what `save(path)` wrote.

        `model_name` and the schema version ride along as extra arrays. They
        are what turns "queried with a different embedding model" from a
        silently wrong answer into a stated one: dimensions can match across
        two unrelated models, and cosine against the wrong space returns
        confident nonsense.
        """
        vectors = self._index.reconstruct_n(0, len(self._ids))
        name = model_name if model_name is not None else self._model_name
        with Path(path).open("wb") as handle:
            np.savez_compressed(
                handle,
                ids=np.array(self._ids, dtype=np.str_),
                vectors=vectors,
                schema_version=np.array(_STORE_SCHEMA_VERSION),
                model_name=np.array("" if name is None else name, dtype=np.str_),
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
            # Both keys are absent in files written before 2026-08-25; an
            # older artifact loads unchanged rather than needing a migration.
            recorded = (
                str(payload["model_name"]) if "model_name" in payload.files else ""
            )
        store = cls(int(vectors.shape[1]))
        store.add(ids, vectors)
        store._model_name = recorded or None
        return store
