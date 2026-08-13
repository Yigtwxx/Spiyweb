# Spiyweb — Design Specification

- **Date:** 2026-08-10
- **Revised:** 2026-08-12 — stale sections aligned with `CLAUDE.md`; new
  decisions D26–D33 recorded in §8.1b
- **Status:** Approved for Phase 1 (research prototype)
- **Owner:** Yigit Erdogan

---

## 1. Problem

Classic RAG retrieval cuts off at `top-k`: the N chunks most similar to the query
embedding are returned, everything else is discarded. This fails whenever the
answer is **distributed** across several documents, because no single one of them
is among the most similar chunks. Raising `k` does not fix it — it dilutes the
context with paraphrases of the same top hit and burns tokens.

Two structural weaknesses:

1. **A hard cutoff has no notion of indirect relevance.** A chunk that is only
   weakly similar to the query, but strongly connected to a chunk that *is*
   relevant, is invisible to `top-k`.
2. **Repetition is treated as content.** Ten near-identical chunks fill ten
   context slots and tell the model one thing, badly.

---

## 2. Concept

Picture a sealed box full of oxygen atoms. Every atom is one vector in the
database, and all of them are inert. A user question is a **living ball** thrown
into that box from outside. Wherever it touches, atoms come alive; each living
atom passes part of its life to the atoms it is bonded with; the energy fades at
every step and the web stops growing when there is nothing left to pass on.

Strongly related atoms are bound by strong threads and light up immediately.
Weakly related ones are reached later, through thinner threads — and if enough
thin threads converge on the same atom, it lights up anyway. The answer is built
from the whole web, not from one cluster.

| Metaphor | Technical mapping |
|---|---|
| Box of oxygen atoms | Vector space / index |
| One atom | One graph node — a chunk **or** a proposition (D10) |
| Living ball entering the box | Query embedding = seed node |
| Strong bond | High-weight edge |
| Weak bond | Low-weight edge, 2–3 hops out |
| Energy reaching zero | Activation decay → natural termination |
| The spider web | Activated node set, ranked by accumulated energy |

Formally: **spreading activation** over a sparse graph, mathematically equivalent
to **Personalized PageRank** with a damping factor. It reduces to a repeated
sparse matrix–vector product, so it is cheap and numerically stable.

---

## 3. Algorithm

### 3.1 Decisions

| # | Decision | Value | Rationale |
|---|---|---|---|
| D1 | Decay model | **Multiplicative** (`E_out = E_in * damping`) | Subtractive decay charges a weak edge the same as a strong one. Multiplication makes weak bonds fade fast and strong chains reach far — which is the whole intent |
| D2 | Damping factor | **0.60** | A node forwards 60% of its energy, retains 40% |
| D3 | Fan-out handling | **Split**, weighted by edge similarity | One neighbour receives ~60%; five equidistant neighbours receive ~12% each. Copying instead of splitting would activate thousands of nodes within three hops |
| D4 | Accumulation | **Additive** | Energy arriving through independent paths sums, producing converging evidence at no extra cost |
| D5 | Termination | **Relative energy threshold** — 15% of total injected energy (`1.5` against a seed of `10.0`; revised by D27) | The web self-limits and the rule scales with thermal residue and profiles. There is deliberately no "return N results" parameter |
| D6 | Redundancy handling | Near-duplicate neighbour: **edge weight → 0**, source idea's **vote count += 1** | Repetition becomes confidence instead of context bloat, and pruning the edge also cuts propagation cost |
| D7 | Duplicate detection timing | **Dynamic**, at query time, among currently active nodes | "Similar" is query-relative: two chunks may be interchangeable for one question and distinct for another. Cost is low — active nodes number in the hundreds at most |
| D8 | Vote granularity | Per **document / source**, never per chunk | Otherwise a corpus containing 50 copies of one blurb elects the most-duplicated idea, not the best-supported one |
| D9 | Edge definition | **Layered hybrid**: cosine for the seed's first contact, entity + structural edges for subsequent hops | Cosine neighbours are paraphrases. Hopping along them returns repetition, not new information. Entity edges are what make multi-hop retrieval actually multi-hop |

### 3.1b Extended decisions

Settled after the first design pass. Each has its rationale in `memory/`.

| # | Decision | Value | Rationale |
|---|---|---|---|
| D10 | Node layers | **Two: chunks and propositions**, linked | "Says the same thing" is vague for a chunk, precise for a proposition — which is what redundancy voting and contradiction detection both need |
| D11 | Node mass | Proportional to length, **normalised within its own layer** | Cross-layer raw length would leave the proposition layer permanently dark |
| D12 | Multi-seed colours | Query decomposed into differently **coloured** seeds; a node where two colours meet is a **bridge** | Distinguishes "reached from two sub-questions" from "reached twice from one" — the first is the multi-hop answer, the second is just density. Also makes ranking explainable |
| D13 | Query profiles | `precise` / `explore` / `compare`, each with its own damping, threshold and seed width | One global damping cannot serve a fact lookup and an exploratory sweep at once. Caller picks; no LLM in `core/` |
| D14 | Negative seeds | "excluding X" injects an **energy-absorbing** seed | A post-filter removes X but cannot remove the five chunks that arrived only because they neighbour X |
| D15 | Contradiction | Modelled as **negative charge**; opposing atoms damp rather than reinforce | Closes the design's most serious blind spot |
| D16 | Contradiction surfacing | Library emits a **template-built, LLM-free question with options**; if unanswered, **both sides enter the context flagged as disputed** | Never silently picks a winner — the minority-but-correct source would vanish |
| D17 | Confidence | Total activated energy and web breadth returned as a **confidence score**; the caller decides what counts as abstention | `top-k` structurally cannot produce this. Library ships the signal, not the policy |
| D18 | Corpus gaps | Two dense clusters with no bridge → **gap warning** | Free by-product of propagation; diagnoses the knowledge base, not just the query |
| D19 | Activation paths | Returned **and passed to the LLM** as explanations | Makes retrieval self-explaining — the most visible benefit of being graph-based |
| D20 | Result shape | Grouped into **theme clusters**, not a flat list | The web's shape carries information a ranked list discards |
| D21 | Learned layer | Hebbian reinforcement in a **separate, disableable** layer; base graph never mutated | Reversible, measurable, and popularity bias stays isolated |
| D22 | Conversation memory | **20–30%** of the previous turn's energy persists | Follow-up questions live in the same region of the box |
| D23 | Consolidation | Periodic offline **pruning** of never-used edges; node merging deferred | Merging is irreversible and must not precede measurement |
| D24 | Duplicate threshold | **Adaptive**, from the active set's similarity distribution | More robust than a constant, but hard to debug — the computed value must be visible in the UI |
| D25 | Freshness | **Tie-breaker only** | A continuous recency multiplier promotes recent-but-irrelevant content in a way that is nearly impossible to notice |

### 3.2 Accepted trade-offs

- **Hub penalty.** Splitting energy proportionally punishes information-dense
  nodes that have many neighbours. Accepted for now; a `sim ** alpha` softening
  is the known escape hatch.
- **Contradiction blindness — resolved.** Two chunks at cosine `0.9` can assert
  opposite things — embeddings do not encode negation, so D6 alone would score
  them as agreement and *amplify* a false consensus. Closed by D15/D16
  (negative charge + surfacing) and D26 (detection): a small multilingual NLI
  model runs at **index time inside `edges/`** and emits negative edges, so
  `core/` stays LLM-free. Remaining watch item: NLI recall must be measured.
- **"Strong" is not "true".** Vote counts measure corpus support, not
  correctness. D8 limits, but does not eliminate, this.

### 3.3 Worked example

Parameters: seed `Q = 10.0`, damping `0.60`, threshold `1.5`.

```
┌──────────────────────── VECTOR BOX ────────────────────────────┐
│   ·       ·        ·       ·        ·       ·       ·      ·   │
│                                                                │
│                      [A'] ⋯⋯ duplicate: edge = 0,              │
│                       ╎        excluded from context,          │
│                       ╎        idea A -> VOTES = 2             │
│               .9   ┌──┴───┐   .8    ┌──────┐                   │
│            ╭──────▶│  A   │────────▶│  B   │                   │
│            │       │ 5.60 │         │ 2.24 │                   │
│   ┌──────┐ │       └───┬──┘         └──────┘         ·         │
│   │  Q   │─┤           │ .4                                    │
│   │ 10.0 │ │           ╰──────╮                                │
│   └──────┘ │                  ▼                                │
│    living  │   ┌──────┐  .6  ┌──────┐  .5   ┌──────┐           │
│      ball  ╰──▶│  C   │─────▶│  D   │──────▶│  F   │           │
│             .7 │ 4.40 │      │ 2.88 │       │ 1.73 │           │
│                └───┬──┘      └──────┘       └───┬──┘           │
│           ·        │ .3       ▲ ACCUMULATED     ╎ 1.04         │
│                    ▼          (two paths)       ▼              │
│                 ┌──────┐                      dies ✗           │
│                 │  E   │ 0.88 -> dies ✗                        │
│                 └──────┘                                       │
│   ·       ·        ·       ·        ·       ·       ·      ·   │
└────────────────────────────────────────────────────────────────┘
   (Q) living ball     [X] activated atom     ·  inert atom
```

```
LEDGER

HOP 0   Q = 10.0  ->  first contact (cosine), split proportional to similarity
        A = 10.0 * .9/(.9+.7) = 5.60
        C = 10.0 * .7/(.9+.7) = 4.40

HOP 1   A distributes:  5.60 * 0.60 = 3.36   neighbours: A'(.95 DUPLICATE) B(.8) D(.4)
          A' -> edge zeroed, never enters context, "idea A" VOTES = 2
          renormalise excluding A' -> B = 3.36 * .8/1.2 = 2.24
                                      D = 3.36 * .4/1.2 = 1.12
        C distributes:  4.40 * 0.60 = 2.64   neighbours: D(.6) E(.3)
                                      D = 2.64 * .6/0.9 = 1.76
                                      E = 2.64 * .3/0.9 = 0.88   ✗ below threshold

HOP 2   D = 1.12 + 1.76 = 2.88            <- converging evidence
        D distributes:  2.88 * 0.60 = 1.73 -> F = 1.73

HOP 3   F distributes:  1.73 * 0.60 = 1.04  ✗ below threshold, the web stops

RESULT  A 5.60 · C 4.40 · D 2.88 · B 2.24 · F 1.73    (+ idea A: 2 votes)
```

**The point of the example:** `D` is never the single most similar node to `Q`,
yet it ranks third — because two independent weak paths converged on it. That
promotion is the only thing this project sells. `A'` never consumed a context
slot, but it strengthened `A`. Any future change that erases either behaviour is
a regression regardless of what the benchmark says.

---

## 4. Architecture

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

### 4.1 Boundary rules

1. **`core/` knows nothing about the outside world** — no vector store, no LLM,
   no embedding model, no filesystem, no network. Arrays and a config in, numbers
   out. This purity is what makes the Phase 1 → Phase 2 promotion free rather
   than a rewrite.
2. **`edges/` is plural on purpose.** The hybrid decision lives in configuration,
   not in code. A new edge type must never require touching `core/`.
3. **`evaluation/` is the Phase 1 product.** In Phase 2 it becomes the
   regression suite. It is never throwaway code. (Named `evaluation`, not
   `eval`, to avoid shadowing the Python builtin.)
4. **`ui/` is not a package dependency.** `pip install spiyweb` must not pull in
   Streamlit; the UI ships as an optional extra — `pip install spiyweb[ui]`.

### 4.2 Data flow

**Index time**
1. Chunk the corpus, embed the chunks (device order: CUDA → MPS → CPU).
2. Build each edge layer independently: `semantic`, `entity`, `structural`.
3. Merge the layers into one weighted sparse adjacency matrix using the
   configured layer weights.

**Query time**
1. Embed the query.
2. **Seed injection** — cosine against the index picks the first-contact atoms;
   the seed energy is split among them proportionally to similarity.
3. **Propagate** — per hop: each active node forwards `damping ×` its energy,
   split across neighbours in proportion to edge weight; incoming energy is
   summed; nodes below the threshold are dropped.
4. **Dedup during propagation** — before forwarding, near-duplicates among the
   currently active set have their edge zeroed and contribute a vote instead.
5. **Rank** — return activated nodes ordered by accumulated energy, each carrying
   its vote count and the path(s) that activated it.

### 4.3 Developer UI

A read-only single-page inspector. Not a product surface; its only job is making
the parameters tunable by eye instead of by guesswork.

Shows: seed atoms and their energies · edge weights coloured by layer · severed
duplicate edges as dashed lines · vote counts · sliders for `damping`,
`threshold` and `max_hop` · the activated web side by side with the plain `top-k`
result list.

---

## 5. Roadmap

| Phase | Contents | Gate to leave it |
|---|---|---|
| **1. Research prototype** | Graph, propagation, dedup, eval harness, dev UI. One hardcoded store. | Beat **both** baselines (`top-k` and iterative retrieval) on MuSiQue by a meaningful margin; HippoRAG reported for reference (D28/D29) |
| **2. Library** | Public API, vector-store adapters, config, docs, packaging | Real external demand (issues, users, requests) — not a personal hunch |
| **3. Framework** | Ingestion, LLM calls, orchestration | — |

What carries over:

| Component | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Graph + propagation + votes | written | reused as-is | reused |
| Eval harness | written | becomes regression tests | reused |
| Vector store adapters | hardcoded, one file | added | reused |
| Public API, config, docs | — | added | reused |
| Ingestion, LLM, orchestration | — | — | **new problem domain** |

Phase 1 → Phase 2 is a **promotion**; Phase 2 → Phase 3 is a **pivot** into a
different problem domain (API design and community, not retrieval). Phase 3 is on
the roadmap by the owner's decision. The binding constraint is that it must not
leak backwards: no Phase 3 abstraction may appear in Phase 1 or 2 code on the
grounds that "we will need it later".

### Tentative revision

The owner has since restated the order as **simple working version → browser face
(UI) → framework / ecosystem**, with the project also distributable as a skill.
This was given explicitly as provisional and may change.

The one structural consequence worth recording: the library step does not
disappear, it is renamed. A browser face cannot be built without a stable API
behind it, so the "UI phase" contains the public-API and config work regardless
of what it is called — and the `core/` purity rule holds for exactly the same
reason under either ordering.

---

## 6. Prior art and differentiation

PPR-style graph retrieval is established: **HippoRAG** (Personalized PageRank
over an entity graph), **GraphRAG**, **LightRAG**, **RAPTOR**. The propagation
mechanism is therefore not the novel part, and the project should not be marketed
as if it were.

The differentiator is **D6 + D7**: converting redundancy into a vote while
severing its edge, dynamically, during propagation. MMR discards duplicates and
loses the signal; this design keeps the signal and discards only the tokens. No
published equivalent is known.

---

## 7. Success criteria

Phase 1 succeeds if, on **MuSiQue**, the web retriever beats **both** baselines
(`top-k` and iterative retrieval) at comparable context budget on the weighted
objective fixed in §8.1: **65% multi-hop accuracy + 35% Novelty@k**, with
**bridge-node recall** as the extra metric. **HippoRAG** results are reported
alongside for reference but are not a gate criterion (D29).

Phase 1 also succeeds if it *fails* clearly — a reliable negative number in two
weeks is a better outcome than a polished library built on an unmeasured
assumption. `top-k` is a surprisingly strong baseline.

---

## 8. Settings and open questions

### 8.1 Phase 1 settings (all decided)

| Item | Decision |
|---|---|
| Objective | **65% multi-hop accuracy + 35% serendipity**, weighted |
| Serendipity metric | **Novelty@k** — relevant nodes the web returns that `top-k` never does |
| Extra metric | **Bridge-node recall** — the only metric that measures the actual claim |
| Benchmark | **MuSiQue** |
| Baselines | **`top-k`** and **iterative retrieval** (LLM query rewriting) |
| Entity extraction | **spaCy + LLM hybrid** |
| Embedding | **multilingual-e5-large** (Turkish + English) |
| Vector store | **numpy + FAISS** |
| Layer weights | By hand (`semantic .5 / entity 1.0 / structural .3`), then a small grid search |
| Seed width | **5 atoms** |
| Environment | **Python 3.11 + uv** |
| Repo / license | **Public from day one, Apache-2.0** |

### 8.1b Decisions added 2026-08-12

| # | Decision | Value |
|---|---|---|
| D26 | Contradiction detection | **Index-time NLI**: a small multilingual NLI model runs inside `edges/` and emits negative edges; `core/` only consumes pre-marked data |
| D27 | Stop-threshold semantics | **Relative** — 15% of total injected energy (equals `1.5` at a `10.0` seed); consistent across thermal turns and profiles |
| D28 | Phase-1 gate | Beat **both** baselines (`top-k` **and** iterative retrieval) by a meaningful margin |
| D29 | Reference comparison | **HippoRAG** reported alongside; informative, not a gate |
| D30 | LLM provider | **Local-first (Ollama)**; free APIs (Gemini, OpenRouter, Groq) selectable via config; provider abstraction lives outside `core/`; secrets via environment |
| D31 | Packaging | **`src/spiyweb/` layout**; `eval/` renamed **`evaluation/`**; `ui/` as optional extra **`spiyweb[ui]`** |
| D32 | Thermal reset | **Hybrid** — caller-controlled `reset()` by default, optional topic-change auto-detection behind a config flag |
| D33 | Platforms | **macOS + Windows + Linux**; no OS-specific paths or calls; device order CUDA → MPS → CPU |
| D34 | Negative knowledge | Negated propositions become permanently **negative-polarity atoms** that absorb the energy of queries asserting the opposite and emit a "corpus disputes this" warning. Schema (`polarity`) and config flag designed now; implemented as an ablation **after** the first Phase 1 measurement |
| D35 | Explained abstention | On low confidence, `retrieve()` returns a **template-built, LLM-free structural refusal report**: activated entity clusters, the missing bridge, where energy died, what kind of source is missing. Phase 1 scope — a thin layer over confidence + gap machinery |
| D36 | Supersession vs contradiction | NLI contradiction + ordered timestamps on the same subject = **update, not conflict**: older atom damped, newer marked current, no user question. Requires NLI evidence, never mere recency. Implementation in Phase 2. (Honest note: temporal RAG is an active field — T-GRAG, Temporal Validity et al.; the differentiation is the integration into the energy web, not the problem itself) |
| D37 | Corpus lint | Offline, retrieval-independent **KB health inspection**: orphan clusters, overloaded hubs, contradiction map, duplicate density. Official Phase 2 product candidate and the project's plan B if multi-hop gains prove marginal |

### 8.2 Remaining open questions

Only items that cannot be settled before code exists:

1. **Normalising the weighted objective.** The two metrics have different scales;
   without a normalisation rule the 65/35 split is nominal rather than real.
2. **Proposition extraction cost.** Manageable on MuSiQue; must be measured
   before being pointed at a real corpus. May become an optional layer.
3. **Adaptive duplicate threshold formula.** Percentile-based or deviation-based
   is undecided. The computed value must surface in the UI either way.
4. **Safety cap values** for `max_nodes` and `max_hop`.
5. **Chunk size** (300–500 tokens mentioned in passing, never decided).
6. **Per-profile parameter values** for `precise` / `explore` / `compare`.
7. **Node mass formula** beyond "proportional to length".
8. **Novelty@k relevance judgement** — how a node `top-k` never returns is
   scored as relevant.
9. **Learned-layer forgetting coefficient** value.
10. **NLI model choice and candidate-pair threshold** for D26.
11. **Negated-proposition (polarity) detection method** for D34 — inside the
    NLI pipeline or a separate polarity classifier? Must be settled after the
    first Phase 1 measurement, before the ablation is implemented.

Closed since the first draft: thermal-memory reset on topic change → **D32**;
node data model → `Node` schema in `core/graph.py` (id, chunk/proposition
layer, source ID, length, UTC-epoch timestamp, cluster ID, D34 `polarity`),
implemented in step 2 together with `LayerWeights` and multi-layer merging.

---

## 9. Naming

`spiyweb`, from the spider-web metaphor. `spiweb` was unavailable (taken by a
company).

Two known and accepted costs: the "spiy" syllable has no single obvious English
pronunciation, and "Spidey" is a registered Marvel trademark, making the name
adjacent to protected IP. Both were raised and the name was chosen anyway.
Fallback candidates if it is ever revisited: `Arachne`, `Gossamer`, `Dragline`.

Availability checked 2026-08-11 — all three free:

| Registry | Result |
|---|---|
| PyPI (`pypi.org/pypi/spiyweb/json`) | 404 — available |
| npm (`registry.npmjs.org/spiyweb`) | 404 — available |
| GitHub repository search (`q=spiyweb`) | 0 results — available |

Since the repository is public from day one, the first commit effectively
reserves the name.
