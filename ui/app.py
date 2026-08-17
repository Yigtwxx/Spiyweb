"""Single-page, read-only Streamlit inspector for the activated web.

A developer tool, never a product surface: the propagation's knobs (damping,
threshold, layer weights) cannot be tuned blind, and this page exists so the
web can be watched spreading. It reads index artifacts and writes nothing.

`ui/` is not a package dependency - `pip install spiyweb` must not drag in
Streamlit. Run it from a checkout:

    pip install -e ".[ui]"
    streamlit run ui/app.py

Every algorithm lives in `graph_view`, which imports no Streamlit at all, so
the logic stays testable in a CI that never installs this extra. This file is
the shell: caches, widgets, error paths, wiring.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TypeVar

import numpy as np
import streamlit as st

from graph_view import (
    EDGE_LAYER_ORDER,
    LAYER_COLORS,
    Comparison,
    EdgeLayerIndex,
    GraphScene,
    LayoutConfig,
    VectorMatrix,
    ViewConfig,
    build_comparison,
    build_layer_index,
    build_scene,
    get_renderer,
    make_similarity,
    vector_matrix,
)
from spiyweb.config import (
    ConflictConfig,
    DedupConfig,
    EmbeddingConfig,
    EvaluationConfig,
    LayerWeights,
    MassConfig,
    OutputConfig,
    PolarityConfig,
    PropagationConfig,
    RetrievalConfig,
    ThermalConfig,
)
from spiyweb.core.graph import Graph
from spiyweb.output import (
    activation_paths,
    build_refusal_report,
    dispute_warnings,
    entity_edge_labels,
    gap_warnings,
    theme_clusters,
)
from spiyweb.profiles import PROFILES, Profile
from spiyweb.retrieve import RetrievalResult, retrieve

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

T = TypeVar("T")

DATA_ROOT = Path("data")
INSTALL_HINTS: dict[str, str] = {
    "store": 'pip install "spiyweb[store]"',
    "embed": 'pip install "spiyweb[embed]"',
}
DATASET_LOADERS: dict[str, str] = {
    "musique": "musique_ans_v1.0_dev.jsonl",
    "2wiki": "2wiki_dev.json",
    "hotpotqa": "hotpot_dev_distractor.json",
}


# --------------------------------------------------------------------------
# Missing-extra handling: the page reports, it never shows a traceback.
# --------------------------------------------------------------------------


def fail_with_install_hint(what: str, extra: str, error: Exception) -> NoReturn:
    """`store.py`-style install message, rendered instead of a traceback."""
    st.error(f"{what} needs the `{extra}` extra ({error}).")
    st.code(INSTALL_HINTS[extra], language="bash")
    st.stop()
    raise SystemExit  # pragma: no cover - st.stop() does not return


def guarded(loader: Callable[[], T], what: str, extra: str) -> T:
    """Run a loader, turning a missing extra into a page-level message."""
    try:
        return loader()
    except ImportError as error:
        fail_with_install_hint(what, extra, error)


# --------------------------------------------------------------------------
# Cached loaders. `cache_resource` = one shared heavy object (never mutated
# here); `cache_data` = serialisable derived data, copied per caller.
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def cached_meta(root: str) -> dict[str, object]:
    """The experiment receipt written by the index stage."""
    path = Path(root) / "meta.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


@st.cache_data(show_spinner="reading the edge layers ...")
def cached_layer_index(root: str) -> EdgeLayerIndex:
    """Per-layer edge memory, rebuilt from the raw artifacts."""
    base = Path(root)
    node_ids = [
        str(record["id"])
        for record in json.loads((base / "nodes.json").read_text(encoding="utf-8"))
    ]
    layers: dict[str, list[tuple[str, str, float]]] = {}
    for layer in EDGE_LAYER_ORDER:
        path = base / f"edges_{layer}.json"
        if not path.exists():
            layers[layer] = []
            continue
        layers[layer] = [
            (str(u), str(v), float(w))
            for u, v, w in json.loads(path.read_text(encoding="utf-8"))
        ]
    return build_layer_index(node_ids, layers)


@st.cache_data(show_spinner=False)
def cached_entities(root: str) -> dict[str, list[str]]:
    """Chunk -> entities, for the "shared entity 'X'" path labels."""
    path = Path(root) / "entities.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): list(value) for key, value in payload.items()}


@st.cache_resource(show_spinner="loading vectors ...")
def cached_vectors(root: str) -> VectorMatrix:
    """Node embeddings straight from `vectors.npz` - the store keeps them private."""
    with np.load(Path(root) / "vectors.npz") as payload:
        ids = [str(node_id) for node_id in payload["ids"]]
        rows = payload["vectors"]
    return vector_matrix(ids, rows)


@st.cache_resource(show_spinner="loading the vector store ...")
def cached_store(root: str) -> object:
    """The seed source (FAISS-backed), rebuilt from the same artifact."""
    from spiyweb.evaluation.index import IndexPaths, load_store

    return load_store(IndexPaths(root=Path(root)))


@st.cache_resource(show_spinner="merging edge layers ...", max_entries=4)
def cached_graph(
    root: str,
    semantic: float,
    entity: float,
    structural: float,
    derivation: float,
    learned: float,
) -> Graph:
    """Re-merge the layers under these weights.

    The weights arrive as five floats rather than a `LayerWeights` so the
    cache key reads as "which weights" and never depends on how Streamlit
    hashes a dataclass. The returned graph is shared and never mutated.
    """
    from spiyweb.evaluation.index import IndexPaths, load_graph

    weights = LayerWeights(
        semantic=semantic,
        entity=entity,
        structural=structural,
        derivation=derivation,
        learned=learned,
    )
    return load_graph(IndexPaths(root=Path(root)), weights)


@st.cache_resource(show_spinner=False)
def cached_conflicts(root: str) -> dict[str, dict[str, float]]:
    """Pre-marked contradiction edges, empty when the NLI stage never ran."""
    from spiyweb.core.conflict import conflict_adjacency
    from spiyweb.evaluation.index import IndexPaths, load_nli_edges

    marked = load_nli_edges(IndexPaths(root=Path(root)))
    return conflict_adjacency(marked) if marked else {}


@st.cache_resource(show_spinner="loading the corpus texts ...")
def cached_dataset(root: str, kind: str, sample_size: int, sample_seed: int) -> object:
    """Titles and passage texts; the index artifacts carry neither."""
    from spiyweb.evaluation.datasets import load_2wiki, load_dataset, load_hotpotqa

    loaders = {
        "musique": load_dataset,
        "2wiki": load_2wiki,
        "hotpotqa": load_hotpotqa,
    }
    path = Path(root) / DATASET_LOADERS[kind]
    config = EvaluationConfig(sample_size=sample_size, sample_seed=sample_seed)
    return loaders[kind](path, config)


@st.cache_resource(show_spinner="loading the embedder ...")
def cached_embedder(model: str, device: str) -> object:
    """e5 on the CPU by default - the GPU belongs to whatever is measuring."""
    from spiyweb.embedding import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(EmbeddingConfig(model=model, device=device))


# --------------------------------------------------------------------------
# Sidebar. Every control's bounds come from `config.py`'s own validation -
# the project forbids magic numbers, and the UI builds its sliders from them.
# --------------------------------------------------------------------------


def available_indexes() -> list[Path]:
    """Directories under `data/` that carry a node registry."""
    if not DATA_ROOT.exists():
        return []
    return sorted(
        path for path in DATA_ROOT.iterdir() if (path / "nodes.json").exists()
    )


def sidebar_index() -> tuple[Path, str, int, int]:
    """Which index, and how its dataset was sampled."""
    st.sidebar.header("Index")
    roots = available_indexes()
    if not roots:
        st.sidebar.error("no index directories under `data/`")
        st.stop()
    root = st.sidebar.selectbox("directory", roots, format_func=lambda p: p.name)
    kind = "musique"
    for candidate, filename in DATASET_LOADERS.items():
        if (root / filename).exists():
            kind = candidate
            break
    st.sidebar.caption(f"dataset format: **{kind}**")
    # meta.json does not record the sampling, so the texts can only be joined
    # when the operator supplies the same draw the index was built with.
    size = st.sidebar.number_input("sample_size", 0, 100000, 1000, step=100)
    seed = st.sidebar.number_input("sample_seed", 0, 999999, 42, step=1)
    return root, kind, int(size), int(seed)


def sidebar_profile() -> Profile | None:
    """A profile overlays exactly damping, threshold and seed width."""
    st.sidebar.header("Profile")
    names = ["(none)", *PROFILES]
    chosen = st.sidebar.selectbox("preset", names, help="D13 knob package")
    if chosen == "(none)":
        return None
    profile = PROFILES[chosen]
    st.sidebar.caption(
        f"damping {profile.damping} · threshold {profile.threshold_ratio} "
        f"· seed width {profile.seed_width} (provisional, open question #6)"
    )
    return profile


def sidebar_propagation(profile: Profile | None) -> PropagationConfig:
    """The core knobs, bounded exactly as `PropagationConfig` validates them."""
    st.sidebar.header("Propagation")
    damping = st.sidebar.slider(
        "damping", 0.01, 0.99, profile.damping if profile else 0.60, 0.01
    )
    ratio = st.sidebar.slider(
        "threshold_ratio", 0.0, 0.99, profile.threshold_ratio if profile else 0.01, 0.01
    )
    energy = st.sidebar.slider("seed_energy", 0.1, 100.0, 10.0, 0.1)
    alpha = st.sidebar.slider("split_alpha", 0.1, 5.0, 3.0, 0.1)
    # Read the brake defaults instead of retyping them: these numbers moved
    # once (max_hop 6 -> 8) and three separate copies stayed behind.
    brakes = PropagationConfig()
    max_hop = st.sidebar.slider("max_hop", 0, 12, brakes.max_hop)
    max_nodes = st.sidebar.slider("max_nodes", 1, 2048, brakes.max_nodes)
    st.sidebar.caption(f"absolute floor = {ratio * energy:.3f} energy units")
    with st.sidebar.expander("node mass (D11, off by default)"):
        mass_on = st.checkbox("enabled", value=False, key="mass_on")
        exponent = st.slider("exponent", 0.0, 3.0, 1.0, 0.1)
        floor = st.slider("mass floor", 0.01, 1.0, 0.5, 0.01)
        cap = st.slider("mass cap", 1.0, 5.0, 2.0, 0.1)
    return PropagationConfig(
        seed_energy=energy,
        damping=damping,
        threshold_ratio=ratio,
        max_hop=max_hop,
        max_nodes=max_nodes,
        split_alpha=alpha,
        mass=MassConfig(enabled=mass_on, exponent=exponent, floor=floor, cap=cap),
    )


def sidebar_layer_weights(counts: Mapping[str, int]) -> LayerWeights:
    """Layer weights; an empty layer's slider is disabled, not silently inert."""
    st.sidebar.header("Edge layers")
    defaults = LayerWeights()
    values: dict[str, float] = {}
    for layer in EDGE_LAYER_ORDER:
        present = counts.get(layer, 0)
        values[layer] = st.sidebar.slider(
            f"{layer} ({present} edges)",
            0.0,
            2.0,
            float(getattr(defaults, layer)),
            0.05,
            disabled=present == 0,
            help="0.0 removes the layer from the merged graph entirely",
        )
        if present == 0:
            st.sidebar.caption(f"no `edges_{layer}.json` content in this index")
    return LayerWeights(**values)


def sidebar_ablations(
    *, has_conflicts: bool
) -> tuple[DedupConfig | None, ConflictConfig | None, PolarityConfig | None]:
    """Mechanism switches; every one of them must be individually disableable."""
    st.sidebar.header("Mechanisms")
    with st.sidebar.expander("redundancy -> vote (dedup)", expanded=True):
        dedup_on = st.checkbox("enabled", value=True, key="dedup_on")
        sigma = st.slider("sigma", 0.0, 5.0, 2.0, 0.1)
        floor = st.slider("floor", 0.01, 1.0, 0.95, 0.01)
        min_pairs = st.slider("min_pairs", 1, 64, 8)
        include_seeds = st.checkbox("include_seeds", value=True)
        st.caption("costs one pairwise pass per hop; watch the elapsed metric")
    dedup = (
        DedupConfig(
            enabled=True,
            sigma=sigma,
            floor=floor,
            min_pairs=min_pairs,
            include_seeds=include_seeds,
        )
        if dedup_on
        else None
    )
    with st.sidebar.expander("contradiction (D15)"):
        if not has_conflicts:
            st.caption("no `edges_nli.json` in this index - nothing to fire")
        conflict_on = st.checkbox(
            "enabled", value=has_conflicts, disabled=not has_conflicts, key="conf_on"
        )
        coefficient = st.slider("coefficient", 0.01, 1.0, 1.0, 0.01)
    conflict = (
        ConflictConfig(enabled=True, coefficient=coefficient)
        if (conflict_on and has_conflicts)
        else None
    )
    with st.sidebar.expander("negative-knowledge atoms (D34)"):
        polarity_on = st.checkbox("enabled", value=True, key="pol_on")
        absorb = st.slider("absorbed fraction", 0.01, 1.0, 1.0, 0.01)
    polarity = PolarityConfig(enabled=True, coefficient=absorb) if polarity_on else None
    return dedup, conflict, polarity


def sidebar_view() -> tuple[ViewConfig, LayoutConfig, str, int]:
    """What the picture may contain, and who draws it."""
    st.sidebar.header("Picture")
    mode = st.sidebar.radio(
        "edges",
        ["contributors", "induced"],
        help="contributors = the links energy actually crossed",
    )
    max_nodes = st.sidebar.slider("drawn atoms", 10, 800, 300, 10)
    max_edges = st.sidebar.slider("drawn edges", 0, 6000, 1500, 100)
    labels = st.sidebar.slider("labelled atoms", 0, 60, 15)
    iterations = st.sidebar.slider("layout iterations", 20, 500, 200, 10)
    renderer = st.sidebar.selectbox("renderer", ["vega-lite", "plotly"])
    top_k = st.sidebar.slider("comparison k", 1, 20, 5)
    view = ViewConfig(
        max_nodes=max_nodes,
        max_edges=max_edges,
        edge_mode=mode,
        label_top_n=labels,
    )
    return view, LayoutConfig(iterations=iterations), renderer, top_k


# --------------------------------------------------------------------------
# Query side.
# --------------------------------------------------------------------------


def resolve_query(
    vectors: VectorMatrix, titles: Mapping[str, str]
) -> tuple[list[float] | None, str]:
    """A query vector plus a label, from free text or from a corpus atom.

    The corpus-atom mode needs no embedder at all, which is what lets the
    inspector be fully usable with only the `store` extra installed - torch
    is a two-gigabyte dependency and this page should not force it.
    """
    st.sidebar.header("Query")
    mode = st.sidebar.radio("source", ["corpus atom", "free text"])
    if mode == "corpus atom":
        options = list(vectors.ids)
        node = st.sidebar.selectbox(
            "atom",
            options,
            format_func=lambda node_id: f"{node_id} — {titles.get(node_id, '')[:40]}",
        )
        row = vectors.position[node]
        return vectors.matrix[row].tolist(), node
    text = st.sidebar.text_area("question", "", height=90)
    if not text.strip():
        return None, ""
    model = st.sidebar.text_input("model", EmbeddingConfig().model)
    device = st.sidebar.selectbox("device", ["cpu", "cuda", "mps"])
    embedder = guarded(
        lambda: cached_embedder(model, device), "free-text queries", "embed"
    )
    return embedder.embed_queries([text])[0], text  # type: ignore[attr-defined]


def run_query(
    query: Sequence[float],
    store: object,
    graph: Graph,
    config: RetrievalConfig,
    *,
    similarity: object | None,
    dedup: DedupConfig | None,
    source_of: Mapping[str, str],
    negative: Mapping[str, Mapping[str, float]] | None,
    conflict: ConflictConfig | None,
    polarity: PolarityConfig | None,
) -> tuple[RetrievalResult, float]:
    """One retrieval, timed - the cost of a toggle must be visible."""
    started = time.perf_counter()
    result = retrieve(
        query,
        store,  # type: ignore[arg-type]
        graph,
        config,
        similarity=similarity,  # type: ignore[arg-type]
        dedup=dedup,
        source_of=source_of,
        negative=negative,
        conflict=conflict,
        polarity=polarity,
    )
    return result, (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------
# Body.
# --------------------------------------------------------------------------


def render_metrics(
    result: RetrievalResult, elapsed_ms: float, *, dedup_on: bool
) -> None:
    """The six numbers that describe a run, tau among them by design."""
    confidence = result.confidence
    columns = st.columns(6)
    columns[0].metric("total energy", f"{confidence.total_energy:.2f}")
    columns[1].metric("atoms", confidence.node_count)
    columns[2].metric("hop depth", confidence.hop_depth)
    columns[3].metric("stopped by", result.propagation.stop_reason)
    if result.contact_tau is None:
        columns[4].metric("dedup tau", "—")
        columns[4].caption(
            "dedup off" if not dedup_on else "no duplicate candidates this query"
        )
    else:
        columns[4].metric("dedup tau", f"{result.contact_tau:.4f}")
        columns[4].caption("adaptive, computed from this query's contacts")
    columns[5].metric("elapsed", f"{elapsed_ms:.0f} ms")
    thresholds = result.propagation.dedup_thresholds
    if thresholds:
        with st.expander(f"per-stage duplicate cuts ({len(thresholds)})"):
            st.write([round(value, 4) for value in thresholds])


def render_comparison(comparison: Comparison, k: int) -> None:
    """The panel the whole tool exists for: the web against plain top-k."""
    st.subheader(f"Activated web vs plain top-k (both cut at k={k})")
    left, right = st.columns(2, gap="large")
    left.markdown("**plain top-k — RIVAL**")
    left.dataframe(
        [
            {
                "#": row.rank,
                "id": row.node_id,
                "cosine": round(row.score, 4),
                "also in web": row.in_other,
                "title": row.title,
            }
            for row in comparison.baseline
        ],
        hide_index=True,
        width="stretch",
    )
    right.markdown("**SPIYWEB activated web**")
    right.dataframe(
        [
            {
                "#": row.rank,
                "id": row.node_id,
                "energy": round(row.score, 3),
                "hop": row.hop,
                "votes": row.votes,
                "also in top-k": row.in_other,
                "badges": " ".join(row.badges),
                "title": row.title,
            }
            for row in comparison.web
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"only in web: **{len(comparison.only_in_web)}** · "
        f"only in top-k: **{len(comparison.only_in_baseline)}** · "
        f"overlap: **{len(comparison.overlap)}** — the novelty term, visible"
    )


def render_graph(scene: GraphScene, renderer_name: str, height: int) -> None:
    """Draw the scene, falling back when an optional backend is absent."""
    from graph_view import RendererUnavailable

    st.subheader("The web")
    try:
        get_renderer(renderer_name).render(scene, height=height)
    except RendererUnavailable as error:
        st.info(f"{renderer_name} is not installed ({error}); using vega-lite")
        get_renderer("vega-lite").render(scene, height=height)
    legend = " · ".join(
        f":{'red' if layer == 'learned' else 'blue'}[{layer}]" for layer in LAYER_COLORS
    )
    st.caption(
        f"{scene.caption}. Edge colour = dominant layer ({legend}); "
        "**dashed = redundancy link cut by dedup**, its share redistributed "
        "and the survivor voted. Atom size = accumulated energy."
    )


def render_tabs(
    result: RetrievalResult,
    graph: Graph,
    config: RetrievalConfig,
    entities: Mapping[str, list[str]],
    texts: Mapping[str, str],
    output: OutputConfig,
) -> None:
    """Everything the output contract promises, one tab each."""
    names = [
        "Seeds",
        "Votes",
        "Paths",
        "Clusters & gaps",
        "Conflicts & disputes",
        "Refusal report",
        "Receipt",
    ]
    seeds, votes, paths, clusters, conflicts, refusal, receipt = st.tabs(names)

    with seeds:
        total = sum(result.seeds.values()) or 1.0
        st.dataframe(
            [
                {
                    "atom": node,
                    "cosine": round(score, 4),
                    "injected": round(
                        config.propagation.seed_energy * score / total, 3
                    ),
                    "title": texts.get(node, "")[:80],
                }
                for node, score in result.seeds.items()
            ],
            hide_index=True,
            width="stretch",
        )
        if result.contact_suppressed:
            st.markdown("**Contact-stage duplicates (elastic refill)**")
            st.dataframe(
                [
                    {"duplicate": duplicate, "voted survivor": survivor}
                    for duplicate, survivor in result.contact_suppressed.items()
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("no duplicate contact was suppressed for this query")

    with votes:
        counted = result.votes()
        if counted:
            st.dataframe(
                [
                    {"idea (source)": key, "votes": value}
                    for key, value in sorted(counted.items(), key=lambda item: -item[1])
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption("votes count corpus support per SOURCE, never per chunk")
        else:
            st.caption("no idea absorbed a duplicate: every vote is the implicit 1")

    with paths:
        labels = entity_edge_labels(
            [
                (contributor, node)
                for node, activation in result.propagation.activations.items()
                for contributor in activation.contributors
            ],
            entities,
        )
        st.dataframe(
            [
                {
                    "atom": path.node,
                    "energy": round(path.energy, 3),
                    "hop": path.hop,
                    "converging": path.converging,
                    "path": path.rendered(labels),
                }
                for path in activation_paths(result.propagation)[:40]
            ],
            hide_index=True,
            width="stretch",
        )

    with clusters:
        found = theme_clusters(result.propagation, graph)
        st.dataframe(
            [
                {
                    "top atom": cluster.top_node,
                    "atoms": len(cluster.nodes),
                    "energy": round(cluster.energy, 3),
                    "share": f"{cluster.energy_share:.1%}",
                }
                for cluster in found
            ],
            hide_index=True,
            width="stretch",
        )
        for warning in gap_warnings(found, output):
            st.warning(warning.message)

    with conflicts:
        if result.conflicts:
            st.dataframe(
                [
                    {
                        "a": record.node_a,
                        "b": record.node_b,
                        "hop": record.hop,
                        "absorbed each": round(record.absorbed_each, 3),
                    }
                    for record in result.conflicts
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(f"disputed and still ranked: {sorted(result.disputed)}")
        else:
            st.caption("no contradiction fired in this run")
        for warning in dispute_warnings(result.disputes):
            st.error(warning.message)

    with refusal:
        report = build_refusal_report(result.propagation, graph, config=output)
        st.code(report.text)

    with receipt:
        st.caption(
            "the settings behind this run; a measurement without them is not one"
        )
        st.json(
            {
                "retrieval": asdict(config),
                "output": asdict(output),
            }
        )


def sidebar_thermal() -> ThermalConfig | None:
    """Conversation warmth across turns; the caller owns the reset (D32)."""
    st.sidebar.header("Thermal memory")
    enabled = st.sidebar.checkbox("keep the previous turn warm", value=False)
    if not enabled:
        st.session_state.pop("thermal_residue", None)
        return None
    ratio = st.sidebar.slider("residue_ratio", 0.01, 0.99, 0.25, 0.01)
    if st.sidebar.button("reset warmth"):
        st.session_state.pop("thermal_residue", None)
    warm = st.session_state.get("thermal_residue", {})
    st.sidebar.caption(f"warm atoms held: {len(warm)}")
    return ThermalConfig(enabled=True, residue_ratio=ratio, auto_reset=False)


def main() -> None:
    """Wire the page: load, query, compare, draw, explain."""
    st.set_page_config(page_title="Spiyweb inspector", layout="wide")
    st.title("Spiyweb — activated web inspector")
    st.caption(
        "Read-only developer tool. Nothing here writes to disk, and the "
        "coloured multi-seed path is deliberately absent: decomposition needs "
        "an LLM, which does not belong in an inspector."
    )

    root, kind, sample_size, sample_seed = sidebar_index()
    meta = cached_meta(str(root))
    layer_index = cached_layer_index(str(root))
    vectors = guarded(lambda: cached_vectors(str(root)), "the vector store", "store")
    store = guarded(lambda: cached_store(str(root)), "the vector store", "store")
    negative = guarded(lambda: cached_conflicts(str(root)), "the index", "store")
    entities = cached_entities(str(root))

    titles: dict[str, str] = {}
    texts: dict[str, str] = {}
    try:
        dataset = cached_dataset(str(root), kind, sample_size, sample_seed)
        titles = dict(dataset.titles)  # type: ignore[attr-defined]
        texts = dict(dataset.texts)  # type: ignore[attr-defined]
    except (FileNotFoundError, KeyError, ValueError) as error:
        st.warning(f"corpus texts unavailable ({error}); showing ids only")

    weights = sidebar_layer_weights(layer_index.counts)
    graph = guarded(
        lambda: cached_graph(
            str(root),
            weights.semantic,
            weights.entity,
            weights.structural,
            weights.derivation,
            weights.learned,
        ),
        "the graph",
        "store",
    )
    if titles and not set(texts) >= {node for node in graph.nodes if "#" not in node}:
        st.warning(
            "the loaded sample does not cover this index - meta.json does not "
            "record sample_seed, so check the sampling inputs; ids still work"
        )

    profile = sidebar_profile()
    propagation = sidebar_propagation(profile)
    dedup, conflict, polarity = sidebar_ablations(has_conflicts=bool(negative))
    thermal = sidebar_thermal()
    view, layout, renderer_name, top_k = sidebar_view()
    st.sidebar.header("Output")
    min_nodes = st.sidebar.slider("gap: min cluster atoms", 1, 20, 3)
    min_share = st.sidebar.slider("gap: min energy share", 0.0, 1.0, 0.15, 0.01)
    output = OutputConfig(
        min_cluster_nodes=min_nodes, min_cluster_energy_share=min_share
    )

    seed_width = st.sidebar.slider(
        "seed_width", 1, 20, profile.seed_width if profile else 5
    )
    overfetch = st.sidebar.slider("contact_overfetch", 1, 10, 3)
    config = RetrievalConfig(
        seed_width=seed_width, propagation=propagation, contact_overfetch=overfetch
    )

    query, label = resolve_query(vectors, titles)
    st.markdown(
        f"**index** `{root.name}` · {meta.get('corpus_chunks', '?')} chunks · "
        f"propositions {meta.get('propositions')} · "
        f"nli edges {meta.get('nli_edges')} · llm `{meta.get('llm_model')}`"
    )
    if query is None:
        st.info("enter a question or pick a corpus atom in the sidebar")
        return

    similarity = make_similarity(vectors) if dedup is not None else None
    source_of = {node_id: node.source_id for node_id, node in graph.node_data.items()}
    residue = st.session_state.get("thermal_residue") if thermal else None
    try:
        started = time.perf_counter()
        result = retrieve(
            query,
            store,  # type: ignore[arg-type]
            graph,
            config,
            similarity=similarity,
            dedup=dedup,
            source_of=source_of,
            negative=negative or None,
            conflict=conflict,
            polarity=polarity,
            residue=residue or None,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
    except ValueError as error:
        st.error(str(error))
        return

    if thermal is not None:
        st.session_state["thermal_residue"] = {
            node: activation.energy * thermal.residue_ratio
            for node, activation in result.propagation.activations.items()
            if activation.energy > 0.0
        }

    render_metrics(result, elapsed, dedup_on=dedup is not None)
    baseline = [node for node, _ in store.search(query, top_k)]  # type: ignore[attr-defined]
    baseline_scores = dict(store.search(query, top_k))  # type: ignore[attr-defined]
    comparison = build_comparison(
        web_ranked=result.ranked(),
        baseline_ids=baseline,
        k=top_k,
        activations=result.propagation.activations,
        votes=result.votes(),
        graph=graph,
        seeds=set(result.seeds),
        disputed=result.disputed,
        contact_tau=result.contact_tau,
        dedup_enabled=dedup is not None,
        titles=titles,
        texts=texts,
        baseline_scores=baseline_scores,
    )
    render_comparison(comparison, top_k)

    scene = build_scene(
        activations=result.propagation.activations,
        seeds=result.seeds,
        votes=result.votes(),
        suppressed={**result.contact_suppressed, **result.propagation.suppressed},
        graph=graph,
        layer_index=layer_index,
        weights=weights,
        view=view,
        layout=layout,
        salt=label,
        texts=texts,
        disputed=result.disputed,
    )
    render_graph(scene, renderer_name, view.height)
    render_tabs(result, graph, config, entities, texts, output)


if __name__ == "__main__":
    main()
