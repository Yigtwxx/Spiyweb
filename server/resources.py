"""Process-wide cache of the heavy objects a query needs.

Loading the MuSiQue graph is about a second and roughly a hundred megabytes;
the vector matrix is another forty. A browser that re-loads those on every
keystroke would be unusable, and — worse on this machine — it would compete
with a six-hour measurement run for RAM.

Everything here is keyed by strings and plain numbers, never by a dataclass
instance: the cache key should read as "which weights", not "whatever this
object hashes to". That is the same reasoning the Streamlit tool's
`cached_graph` documents.

Nothing in this module mutates what it hands out. The objects are shared, and
`Graph` / `VectorStore` are treated as read-only by every caller.
"""

from __future__ import annotations

import gc
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from server._paths import DATA_ROOT
from server.settings import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Callable

    from spiyweb.core.graph import Graph
    from spiyweb.scene import EdgeLayerIndex, VectorMatrix

T = TypeVar("T")

DATASET_FILES: dict[str, str] = {
    "musique": "musique_ans_v1.0_dev.jsonl",
    "2wiki": "2wiki_dev.json",
    "hotpotqa": "hotpot_dev_distractor.json",
}
EDGE_LAYERS: tuple[str, ...] = (
    "semantic",
    "entity",
    "structural",
    "derivation",
    "learned",
)


class MissingExtra(RuntimeError):
    """An optional dependency is absent; the message names the install line."""

    def __init__(self, what: str, extra: str, cause: Exception) -> None:
        super().__init__(f"{what} needs the `{extra}` extra ({cause})")
        self.extra = extra
        self.hint = f'uv pip install -e ".[{extra}]"'


@dataclass(frozen=True)
class CacheEntry:
    """One cached object, with a measured (not guessed) size."""

    kind: str
    key: str
    approx_bytes: int


class _Lru(OrderedDict[str, T]):
    """Smallest possible LRU: an OrderedDict with a cap."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.maxsize = max(1, maxsize)

    def touch(self, key: str, build: Callable[[], T]) -> T:
        if key in self:
            self.move_to_end(key)
            return self[key]
        value = build()
        self[key] = value
        while len(self) > self.maxsize:
            self.popitem(last=False)
        return value


class ResourceCache:
    """The single instance every request goes through."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._graphs: _Lru[Graph] = _Lru(SETTINGS.cache_graphs)
        self._stores: _Lru[object] = _Lru(SETTINGS.cache_stores)
        self._vectors: _Lru[VectorMatrix] = _Lru(SETTINGS.cache_vectors)
        self._layers: _Lru[EdgeLayerIndex] = _Lru(SETTINGS.cache_layers)
        self._entities: _Lru[dict[str, list[str]]] = _Lru(SETTINGS.cache_entities)
        self._texts: _Lru[dict[str, dict[str, str]]] = _Lru(SETTINGS.cache_texts)
        self._embedder: tuple[str, object] | None = None

    # -- paths ----------------------------------------------------------------

    def index_root(self, name: str) -> Path:
        """Resolve an index name to a directory, refusing anything outside it.

        The name arrives from the browser, so it is treated as untrusted: only
        a direct child of `data/` that carries a node registry is accepted.
        """
        candidate = (DATA_ROOT / name).resolve()
        if candidate.parent != DATA_ROOT.resolve():
            raise ValueError(f"index {name!r} is not a directory under data/")
        if not (candidate / "nodes.json").exists():
            raise ValueError(f"index {name!r} has no nodes.json")
        return candidate

    def available(self) -> list[Path]:
        """Index directories, sorted, newest artifacts irrelevant to order."""
        if not DATA_ROOT.exists():
            return []
        return sorted(
            path
            for path in DATA_ROOT.iterdir()
            if path.is_dir() and (path / "nodes.json").exists()
        )

    def dataset_kind(self, root: Path) -> str | None:
        """Which loader this index's dataset file belongs to, if it has one."""
        for kind, filename in DATASET_FILES.items():
            if (root / filename).exists():
                return kind
        return None

    # -- artifacts ------------------------------------------------------------

    def meta(self, root: Path) -> dict[str, object]:
        path = root / "meta.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    def node_records(self, root: Path) -> list[dict[str, object]]:
        payload = json.loads((root / "nodes.json").read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []

    def layer_counts(self, root: Path) -> dict[str, int]:
        """Edge count per layer, so a UI can grey out a slider that does nothing."""
        counts: dict[str, int] = {}
        for layer in EDGE_LAYERS:
            path = root / f"edges_{layer}.json"
            if not path.exists():
                counts[layer] = 0
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            counts[layer] = len(payload) if isinstance(payload, list) else 0
        return counts

    # -- heavy objects --------------------------------------------------------

    def graph(self, root: Path, weights: tuple[float, ...]) -> Graph:
        from spiyweb.config import LayerWeights
        from spiyweb.evaluation.index import IndexPaths, load_graph

        key = f"{root}|{weights}"
        with self._lock:
            return self._graphs.touch(
                key,
                lambda: load_graph(
                    IndexPaths(root=root),
                    LayerWeights(
                        semantic=weights[0],
                        entity=weights[1],
                        structural=weights[2],
                        derivation=weights[3],
                        learned=weights[4],
                    ),
                ),
            )

    def store(self, root: Path) -> object:
        try:
            from spiyweb.evaluation.index import IndexPaths, load_store
        except ImportError as error:
            raise MissingExtra("the vector store", "store", error) from error
        with self._lock:
            return self._stores.touch(
                str(root), lambda: load_store(IndexPaths(root=root))
            )

    def vectors(self, root: Path) -> VectorMatrix:
        from spiyweb.scene import vector_matrix

        def build() -> VectorMatrix:
            with np.load(root / "vectors.npz") as payload:
                ids = [str(node_id) for node_id in payload["ids"]]
                rows = payload["vectors"]
            return vector_matrix(ids, rows)

        with self._lock:
            return self._vectors.touch(str(root), build)

    def layer_index(self, root: Path) -> EdgeLayerIndex:
        from spiyweb.scene import build_layer_index

        def build() -> EdgeLayerIndex:
            ids = [str(record["id"]) for record in self.node_records(root)]
            layers: dict[str, list[tuple[str, str, float]]] = {}
            for layer in EDGE_LAYERS:
                path = root / f"edges_{layer}.json"
                if not path.exists():
                    layers[layer] = []
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                layers[layer] = [(str(u), str(v), float(w)) for u, v, w in payload]
            return build_layer_index(ids, layers)

        with self._lock:
            return self._layers.touch(str(root), build)

    def entities(self, root: Path) -> dict[str, list[str]]:
        def build() -> dict[str, list[str]]:
            path = root / "entities.json"
            if not path.exists():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {str(key): list(value) for key, value in payload.items()}

        with self._lock:
            return self._entities.touch(str(root), build)

    def corpus(
        self, root: Path, sample_size: int, seed: int
    ) -> dict[str, dict[str, str]]:
        """Titles and texts for an index, or empty maps when unavailable.

        `meta.json` does not record the sampling, so the caller has to supply
        the draw the index was built with. A mismatch is reported by the route
        rather than guessed at here.
        """
        kind = self.dataset_kind(root)
        if kind is None:
            return {"titles": {}, "texts": {}}

        def build() -> dict[str, dict[str, str]]:
            from spiyweb.config import EvaluationConfig
            from spiyweb.evaluation.datasets import (
                load_2wiki,
                load_dataset,
                load_hotpotqa,
            )

            loaders = {
                "musique": load_dataset,
                "2wiki": load_2wiki,
                "hotpotqa": load_hotpotqa,
            }
            dataset = loaders[kind](
                root / DATASET_FILES[kind],
                EvaluationConfig(sample_size=sample_size, sample_seed=seed),
            )
            return {"titles": dict(dataset.titles), "texts": dict(dataset.texts)}

        with self._lock:
            return self._texts.touch(f"{root}|{kind}|{sample_size}|{seed}", build)

    def embedder(self, model: str, device: str) -> object:
        """The e5 wrapper - loaded lazily and never pre-warmed.

        torch is a two-gigabyte dependency and the GPU belongs to whatever is
        measuring; the corpus-atom query mode exists precisely so this is
        optional.
        """
        key = f"{model}|{device}"
        with self._lock:
            if self._embedder is not None and self._embedder[0] == key:
                return self._embedder[1]
            try:
                from spiyweb.config import EmbeddingConfig
                from spiyweb.embedding import SentenceTransformerEmbedder
            except ImportError as error:
                raise MissingExtra("free-text queries", "embed", error) from error
            built = SentenceTransformerEmbedder(
                EmbeddingConfig(model=model, device=device)
            )
            self._embedder = (key, built)
            return built

    # -- introspection --------------------------------------------------------

    def entries(self) -> list[CacheEntry]:
        """What is loaded right now, with measured sizes."""
        found: list[CacheEntry] = []
        with self._lock:
            for key, graph in self._graphs.items():
                edges = sum(len(neighbors) for neighbors in graph.adjacency.values())
                found.append(CacheEntry("graph", key, edges * 96 + len(graph) * 200))
            for key in self._stores:
                found.append(CacheEntry("store", key, 0))
            for key, vectors in self._vectors.items():
                found.append(CacheEntry("vectors", key, int(vectors.matrix.nbytes)))
            for key, layers in self._layers.items():
                size = sum(int(codes.nbytes) for codes in layers.codes.values())
                size += sum(int(w.nbytes) for w in layers.layer_weights.values())
                found.append(CacheEntry("layer_index", key, size))
            for key, entities in self._entities.items():
                found.append(CacheEntry("entities", key, len(entities) * 120))
            for key, corpus in self._texts.items():
                size = sum(len(text) for text in corpus["texts"].values())
                found.append(CacheEntry("corpus", key, size))
        return found

    @property
    def embedder_loaded(self) -> bool:
        return self._embedder is not None

    def clear(self) -> None:
        """Drop everything and collect - the operator may want RAM back."""
        with self._lock:
            self._graphs.clear()
            self._stores.clear()
            self._vectors.clear()
            self._layers.clear()
            self._entities.clear()
            self._texts.clear()
            self._embedder = None
        gc.collect()


CACHE = ResourceCache()
