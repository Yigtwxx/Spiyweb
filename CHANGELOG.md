# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Public API policy

`import spiyweb` is the **query-time** contract: everything in
`spiyweb.__all__`, and nothing else. `spiyweb.indexing` is the **index-time**
contract, on the same terms. Anything reachable only through a submodule path
— `spiyweb.core.*`, `spiyweb.evaluation.*`, `spiyweb.config.*`, `server.*`,
`ui/` — is internal and may change in any release without a note here.

Two rules hold from the first tagged release onwards:

1. Removing or renaming a declared name is a MINOR bump while the project is
   below 1.0, and it lands here under **Removed** with the replacement path.
2. A change that moves a measured retrieval number is never a silent one. It
   lands under **Changed** with the measurement that justifies it, because a
   benchmark comparison across two versions is only honest when the diff
   between them is written down.

`spiyweb.evaluation` is the measurement harness, not the library. It is the
regression suite, and its CLI flags are not covered by the policy above.

## [Unreleased]

### Added

- `Stats/phase1_results.png` and the script that regenerates it. One figure
  with six panels replaces nine separate charts that lived outside the
  repository, where a reader could not see which of them disagreed. The
  folder is tracked now.
- `RESULTS.md`: the complete Phase 1 measurement record as a single table -
  gate, metric decomposition, per-hop breakdown, all five failed rescue
  rounds, every mechanism ablation, detection quality, and the limits. It
  replaces a folder of separate charts that lived outside the repository,
  where nobody could see which of them disagreed. No run was executed to
  produce it: groups A-D and H were recomputed from the sealed artifacts,
  the intervals come from the closing report.
- `spiyweb.scene`: the render-agnostic layout, scene and comparison code,
  promoted out of the repository-rooted Streamlit tool. numpy only, behind
  the new `spiyweb[view]` extra, never imported eagerly. `server/_paths.py`
  had written down the condition for this promotion in advance; both halves
  of it came true at once, so the `sys.path` adjustment it existed for is
  gone rather than grandfathered.

- `py.typed`: the package ships its type information, so `mypy` and `pyright`
  see the annotations that were always there instead of `Any`.
- `spiyweb.indexing`: the index-time facade — chunking, the edge builders, the
  embedder, entity extraction, the LLM clients and `VectorStore` in one
  namespace. Importable with nothing installed; only touching `VectorStore` or
  `build_semantic_edges_fast` asks for `spiyweb[store]`.
- `RetrievalResult.dedup_mode`, `ColoredRetrievalResult.dedup_mode` and a
  `dedup_mode` entry in the harness receipt (`results.json`): which
  duplicate rules actually ran — `"off"`, `"sources_only"`, `"cosine_only"` or
  `"full"`. See **Changed** for why this exists.
- `tests/test_public_api.py`, `tests/wheel_smoke.py`, and a CI job that builds
  the wheel and imports it in an environment with no extras installed. Until
  now nothing anywhere checked that `dependencies = []` was true.

### Changed

- `retrieve()` and `retrieve_colored()` refuse a `DedupConfig` no rule can
  run. Two rules suppress a duplicate and they need different things: the
  cosine twin test needs a `similarity` backend, the source test does not.
  With an enabled config and neither of them live, `dedup=` was a no-op the
  caller believed in — which is how the 2026-08 measurement campaign ran tour
  after tour with the mechanism silently off. That combination is now a
  `ValueError`; the source-rule-only combination warns and reports
  `dedup_mode="sources_only"`.

  **No sealed number is affected.** The evaluation harness and the inspector
  both pass `similarity` and `dedup` as a pair or pass neither - the
  `load_similarity` call in `evaluation/run.py` and the `make_similarity`
  call in `server/inspect_api.py` are each guarded by `dedup is not None`
  - so neither reaches the refused path. `distinct_sources=False` is
  constructed nowhere outside the tests.
- The version has a single source: `src/spiyweb/__init__.py`. `pyproject.toml`
  reads it through `dynamic = ["version"]` instead of keeping a second copy.

### Removed

- The Streamlit inspector (`ui/`) and the `spiyweb[ui]` extra. The browser
  face (`server/` + `web/`) does the same job against the same artifacts and
  the same `retrieve()`, and maintaining two front ends bought nothing once
  they already shared the scene code.
- With it, the scene's renderer layer - `SceneRenderer`, `VegaLiteRenderer`,
  `PlotlyRenderer`, `RENDERERS`, `get_renderer`, `RendererUnavailable`. Those
  imported Streamlit, and after the promotion they would have shipped that
  import inside the wheel with no caller left to use it. `build_vega_spec`
  stays: it is a pure scene-to-spec function, and its tests pin scene
  semantics that nothing else covers.

- `EvaluationConfig` and `IterativeBaselineConfig` from `spiyweb.__all__`
  (84 → 82 names). They configure the benchmark, not the library. They have
  not moved: `from spiyweb.config import EvaluationConfig` is unchanged and is
  the path every caller in this repository already used.

### Known gaps

- `propagate()` still accepts an enabled `DedupConfig` without a `similarity`
  backend and runs without dedup. `core/` takes arrays and a config and
  returns numbers; the guard lives one layer up in `retrieve()`, which is
  where the trap actually bit. Stated rather than left to be discovered.
- A `UserWarning` raised through `ThermalSession.retrieve` points at
  `thermal.py` rather than at the caller's line.
