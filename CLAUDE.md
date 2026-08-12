# Spiyweb

Graph-based retrieval for RAG. Instead of cutting retrieval off at `top-k`, the
query is injected into a vector graph as an **energy seed** and spreads outward
with decay. Strongly related nodes light up first; weakly related but genuinely
connected nodes light up later, through multiple hops — like a spider web.

**Status:** Phase 1 — research prototype. Public repo, Apache-2.0.

---

## 1. Mental model

| Metaphor | Technical mapping |
|---|---|
| Box of oxygen atoms | Vector space / index |
| One atom | One node — a chunk **or** a proposition |
| Living ball entering the box | Query embedding = seed node |
| Strong bond | High-weight edge |
| Weak bond | Low-weight edge, 2–3 hops out |
| Energy reaching zero | Activation decay → natural termination |
| The spider web | Activated node set, ranked by accumulated energy |

Formally: **spreading activation** over a sparse graph, mathematically equivalent
to **Personalized PageRank** with a damping factor. It reduces to a repeated
sparse matrix–vector product, so it is cheap and numerically stable.

---

## 2. Algorithm invariants

Settled design decisions. These are the identity of the project, not tuning
knobs. Full rationale for each lives in `memory/`.

### 2.1 Propagation

| Rule | Value / behaviour | Why |
|---|---|---|
| Decay | **Multiplicative**: `E_out = E_in * damping` | A weak edge must fade faster than a strong one |
| Damping | `0.60` default, overridden per query profile | Node forwards 60% of its energy, keeps 40% |
| Split | Energy is **divided** among neighbours, **proportional to edge weight** | One neighbour gets ~60%; five equidistant neighbours get ~12% each |
| Accumulation | **Additive** — energy arriving via multiple paths sums | Gives converging evidence for free |
| Dedup renormalisation | After a duplicate's edge is zeroed, the remaining neighbour weights are **renormalised** — the duplicate's share is redistributed | Visible in the worked example; stated here as a rule, not an accident |
| Energy conservation | Dedup **redistributes** (conserves) energy; negative seeds and contradictions **absorb** (destroy) it; nothing else creates or destroys energy | Makes every mechanism's effect on the energy ledger auditable |
| Stop condition | **Relative energy threshold** — 15% of total injected energy (`1.5` against a seed of `10.0`) — *not* a token budget | Scales consistently when thermal residue or profiles change the injected total; keeps the mechanism pure and context-window independent |
| Safety caps | `max_nodes` and `max_hop` as hard overflow guards | The threshold is the primary stop; these only prevent surprises |
| Node mass | Proportional to length, **normalised within its own layer** | Without per-layer normalisation the proposition layer silently dies |
| Freshness | **Tie-breaker only**, never a continuous multiplier | A continuous boost quietly promotes recent-but-irrelevant content |

### 2.2 Nodes and edges

| Rule | Value / behaviour |
|---|---|
| Node layers | **Two**: chunk nodes and proposition nodes, linked to each other |
| Edge layers | **Layered hybrid** — `semantic` (cosine, seed contact + fallback only), `entity` (main hop fuel), `structural` (same doc/section/adjacent), `learned` (usage-reinforced) |
| Learned layer | Hebbian reinforcement lives in a **separate, disableable layer**; the base graph is never mutated |
| Consolidation | Periodic offline **pruning** of never-used edges. Node merging is deferred to Phase 2 (irreversible) |

### 2.3 Redundancy, contradiction, negation

| Rule | Value / behaviour |
|---|---|
| Redundancy | Near-duplicate neighbour: **edge weight → 0**, source idea's **vote count += 1** |
| Duplicate detection | **Dynamic**, at query time, among currently active nodes |
| Duplicate threshold | **Adaptive**, computed from the active set's similarity distribution — the computed value must be visible in the UI |
| Vote granularity | Per **document/source**, never per chunk |
| Contradiction | Modelled as **negative charge** — opposing atoms damp each other instead of reinforcing |
| Contradiction detection | **Index-time NLI** — a small multilingual NLI model runs inside `edges/` and emits negative edges; `core/` only consumes pre-marked data |
| Contradiction surfacing | The library emits a **template-built, LLM-free question with options** for the user |
| No answer given | **Both sides enter the context, flagged as disputed.** Never silently pick a winner |
| Negative requests | "excluding X", "without Y" → **energy-absorbing negative seed**, not a post-filter |
| Negative knowledge | Negated propositions ("X does not…") become permanently **negative-polarity atoms**: they absorb the energy of queries asserting the opposite and emit a "corpus disputes this" warning. Designed now (schema `polarity` field + config flag); implemented as an ablation **after** the first Phase 1 measurement |
| Supersession | NLI contradiction + ordered timestamps on the same subject = **update, not conflict** — older atom damped, newer marked current, no user question emitted. Freshness tie-break stays separate: supersession requires NLI evidence, not mere recency. Implementation in Phase 2 |

### 2.4 Query shaping

| Rule | Value / behaviour |
|---|---|
| Multi-seed colours | The query is decomposed; each part is a differently **coloured** seed. A node where two colours meet is a **bridge** — that is where a multi-hop answer lives. Runs as an ablation in Phase 1 |
| Query profiles | `precise` / `explore` / `compare` — each carries its own damping, threshold and seed width. The caller picks; never an LLM inside `core/` |
| Conversation memory | **20–30%** of the previous turn's energy persists; follow-up questions land on warm ground. Reset is **hybrid**: caller-controlled `reset()` by default, optional topic-change auto-detection behind a config flag |

### 2.5 Output contract

`retrieve()` returns, in one structure:

- ranked nodes with accumulated energy
- vote counts per idea
- **activation paths** — passed to the LLM as explanations, not just debug data
- **theme clusters** — "these are separate themes, they intersect here"
- **confidence score** — total activated energy, activated node count and reached hop depth; the caller decides what counts as "I don't know"
- **corpus gap warnings** — two dense clusters with no bridge between them
- **structural refusal report** — when confidence is low, a template-built,
  LLM-free explanation of *why*: which entity clusters activated, where the
  missing bridge is, where the energy died, what kind of source is missing
- **contradiction records** plus the ready-made user question

### 2.6 Worked example (the canonical trace)

Seed `Q = 10.0`, damping `0.60`, threshold `1.5`.

```
HOP 0   A = 10.0 * .9/(.9+.7) = 5.60      C = 10.0 * .7/(.9+.7) = 4.40

HOP 1   A distributes 5.60 * .60 = 3.36   neighbours: A'(.95 DUPLICATE) B(.8) D(.4)
          A' -> edge zeroed, excluded from context, "idea A" VOTES = 2
          renormalise without A' -> B = 3.36 * .8/1.2 = 2.24
                                    D = 3.36 * .4/1.2 = 1.12
        C distributes 4.40 * .60 = 2.64   neighbours: D(.6) E(.3)
                                    D = 2.64 * .6/0.9 = 1.76
                                    E = 2.64 * .3/0.9 = 0.88   -> below threshold, dies

HOP 2   D = 1.12 + 1.76 = 2.88            <- converging evidence
        D distributes 2.88 * .60 = 1.73 -> F = 1.73

HOP 3   F distributes 1.73 * .60 = 1.04   -> below threshold, web stops

RESULT  A 5.60 | C 4.40 | D 2.88 | B 2.24 | F 1.73     (+ idea A: 2 votes)
```

The ledger rounds hop 0 to two decimals and carries that rounding downwards;
the unrounded values are `A 5.625 | C 4.375 | D 2.875 | B 2.25 | F 1.725`, and
those are what `tests/test_propagate.py` pins. The ordering is identical, which
is the part that matters.

`D` is never the single most similar node to `Q`, yet it ranks third. **That gap
is the entire value proposition of this project.** Any change that destroys it
is a regression, whatever the benchmark says.

---

## 3. Architecture

```
src/spiyweb/
├── core/                 # pure computation - ZERO I/O, zero heavy dependencies
│   ├── graph.py          # sparse adjacency, multi-layer edge merging, node layers
│   ├── propagate.py      # damping, proportional split, accumulation, threshold, mass
│   ├── dedup.py          # dynamic redundancy suppression: edge -> 0, vote += 1
│   ├── colors.py         # multi-seed coloured activation, bridge detection
│   └── conflict.py       # contradiction as negative charge
├── edges/                # hybrid edge builders
│   ├── semantic.py       # cosine kNN (seed contact + fallback only)
│   ├── entity.py         # shared entity / concept edges (main hop fuel)
│   ├── structural.py     # same document, same section, adjacent chunk
│   ├── nli.py            # index-time NLI: emits negative (contradiction) edges
│   └── learned.py        # Hebbian reinforcement, separate and disableable
├── nodes/
│   ├── chunks.py         # chunk-level nodes
│   └── propositions.py   # proposition-level nodes (LLM extraction at index time)
├── llm.py                # LLM provider abstraction: Ollama default, free APIs optional
├── profiles.py           # precise / explore / compare propagation profiles
├── config.py             # dataclass: damping, threshold, max_hop, max_nodes, weights
├── store.py              # numpy + FAISS single-file vector store (outside core/)
├── output.py             # result structure, paths, clusters, confidence, conflicts
├── retrieve.py           # seed injection -> propagate -> structured result
└── evaluation/           # renamed from eval/ — avoids shadowing the Python builtin
    ├── datasets.py       # MuSiQue loader
    ├── baseline.py       # plain top-k + iterative retrieval
    └── run.py            # side-by-side metric report

ui/                       # developer tool, optional extra: pip install spiyweb[ui]
├── app.py                # single-page Streamlit inspector
└── graph_view.py         # force-directed view of the activated web
```

### Boundary rules — the single most important thing in this file

1. **`core/` knows nothing about the outside world.** No vector store, no LLM, no
   embedding model, no filesystem, no network. Arrays and a config in, numbers
   out. This is what keeps every later phase from becoming a rewrite.
2. **`edges/` and `nodes/` are plural on purpose.** Layer choices live in config,
   not in code. Adding a layer must never require touching `core/`.
3. **`evaluation/` is the Phase 1 product.** It becomes the regression suite
   later. It is never throwaway code. (Named `evaluation`, not `eval`, to avoid
   shadowing the Python builtin.)
4. **`ui/` is not a package dependency.** `pip install spiyweb` must not pull in
   Streamlit; the UI ships as an optional extra — `pip install spiyweb[ui]`.
5. **Contradiction question templates live outside `core/`.** The library ships
   them so callers do not rewrite them, but the core only produces structured
   conflict data.

---

## 4. Phase 1 settings

| Item | Decision |
|---|---|
| Objective | **65% multi-hop accuracy + 35% serendipity** (weighted) |
| Serendipity metric | **Novelty@k** — nodes the web returns that `top-k` never returns, and that are still relevant |
| Extra metric | **Bridge-node recall** — where does the required intermediate document rank? Standard `recall@k` does not reward this |
| Benchmark | **MuSiQue** |
| Baselines | **`top-k`** and **iterative retrieval** (LLM query rewriting) — the Phase 1 gate requires beating **both** |
| Reference comparison | **HippoRAG** results reported alongside — informative, not a gate criterion |
| Entity extraction | **spaCy + LLM hybrid** — spaCy for the bulk, LLM for ambiguous cases |
| Embedding | **multilingual-e5-large** (Turkish + English) |
| Vector store | **numpy + FAISS**, single file, no server |
| Layer weights | Start by hand: `semantic .5 / entity 1.0 / structural .3`, then a small grid search |
| Seed width | **5 atoms** |
| Environment | **Python 3.11 + uv** |
| Platforms | **macOS + Windows + Linux** — no OS-specific paths or calls; device order CUDA → MPS → CPU |
| LLM provider | **Local-first (Ollama)**; free APIs (Gemini, OpenRouter, Groq) optional via config — the provider abstraction lives outside `core/` |
| Packaging | **`src/spiyweb/` layout**; `eval/` → **`evaluation/`**; `ui/` as optional extra **`spiyweb[ui]`** |
| Repo / license | **Public from day one, Apache-2.0** |

Because the repo is public from day one, `CLAUDE.local.md` must stay in
`.gitignore` permanently.

---

## 5. Roadmap

| Phase | Contents | Gate to leave it |
|---|---|---|
| **1. Simple working version** | Graph, propagation, dedup, conflicts, eval harness, dev UI | Beat both baselines on MuSiQue by a meaningful margin |
| **2. Browser face (UI)** | Real UI on top of a stable public API — which means the library work happens here whether or not it is called a library. Also lands here: **supersession handling** and the **corpus-lint diagnostic mode** (orphan clusters, overloaded hubs, contradiction map, duplicate density) — the project's plan B if the multi-hop gain proves marginal | Real external demand |
| **3. Framework / ecosystem** | Ingestion, LLM calls, orchestration; distributable as a skill | — |

The order above is a **tentative revision** from the owner and may change. Two
things do not change: gates are triggered by signals, never by the calendar; and
Phase 3 abstractions must never leak backwards into Phase 1 or 2 code.

---

## 6. Code standards

- **Python**, type annotations mandatory on every function signature.
- `ruff` for lint and format. All identifiers, comments and docstrings in
  **English**.
- Device selection order is fixed everywhere a tensor is placed:
  **CUDA → MPS → CPU**.
- No magic numbers in code. Every tunable lives in `config.py` as a dataclass
  field with a documented default — the UI builds its sliders from these.
- Every mechanism in section 2 must be **individually disableable** from config.
  Ablation is impossible otherwise, and ablation is how this project proves
  itself.
- No secrets in source. Read from environment (`os.environ.get`) or `.env`.
  Never log credentials, not even at `DEBUG`.

### Verification

```
ruff check .
ruff format --check .
pytest
```

All three must be clean before any change is considered done.

---

## 7. Things not to do

- Do not add an LLM call to `core/`.
- Do not replace multiplicative decay with a fixed subtraction "for simplicity".
- Do not turn duplicate suppression into plain MMR — dropping duplicates loses
  the vote signal, which is the project's most original idea.
- Do not compute node mass from raw length across layers; propositions are short
  by definition and would never activate.
- Do not silently resolve a contradiction by picking the stronger side.
- Do not let the learned layer write into the base graph.
- Do not evaluate on a private corpus and call it evidence.
- Do not commit, push, or open PRs unless explicitly asked.

---

## 8. Known open risks

- **Prior art.** PPR-based graph RAG already exists (HippoRAG, GraphRAG,
  LightRAG, RAPTOR). Differentiation rests on dynamic redundancy-to-vote
  conversion, coloured multi-seed bridging, and the honesty outputs — not on the
  propagation itself.
- **Hub penalty.** Proportional splitting punishes information-dense nodes.
  Softening option: weight by `sim ** alpha`.
- **Repetition is not truth.** Vote counts measure corpus support. Document-level
  counting limits but does not eliminate this.
- **Index cost.** Proposition extraction and hybrid entity extraction both need
  LLM calls at index time. Fine on a fixed benchmark corpus; must be measured
  before it is pointed at a real one.
- **Learned-layer drift.** Reinforcement without a forgetting factor collapses
  the graph toward whatever was asked most often.

Full design: `docs/specs/2026-08-10-spiyweb-design.md`.
Decision history and rationale: `memory/`.
