# Spiyweb

**Graph-based retrieval for RAG, built on spreading activation — like a spider
web.**

Instead of cutting retrieval off at `top-k`, the query is injected into a
vector graph as an **energy seed** and spreads outward with decay. Strongly
related nodes light up first; weakly related but genuinely connected nodes
light up later, through multiple hops. The answer is built from the whole web,
not from one cluster of near-duplicates.

> **Status: Phase 1 — design complete, implementation pending.**
> This repository currently contains the full design specification and the
> decision log. Code lands next. The design is public from day one so the idea
> can be judged — and challenged — early.

---

## Why

Classic RAG retrieval has two structural weaknesses:

1. **A hard cutoff has no notion of indirect relevance.** When an answer is
   distributed across several documents, no single one of them may be among
   the most similar chunks — so `top-k` never sees it.
2. **Repetition is treated as content.** Ten near-identical chunks fill ten
   context slots and tell the model one thing, badly.

## How it works

Formally: **spreading activation** over a sparse multi-layer graph,
mathematically equivalent to Personalized PageRank with a damping factor. It
reduces to repeated sparse matrix–vector products — cheap and numerically
stable.

```
HOP 0   Q = 10.0  ->  first contact (cosine):   A = 5.60    C = 4.40
HOP 1   A -> B 2.24, D 1.12      C -> D 1.76, E 0.88 (dies)
HOP 2   D = 1.12 + 1.76 = 2.88   <- converging evidence
HOP 3   energy below threshold, the web stops on its own

RESULT  A 5.60 | C 4.40 | D 2.88 | B 2.24 | F 1.73
```

`D` is never the single most similar node to the query, yet it ranks third —
because two independent weak paths converged on it. That promotion is the
entire value proposition.

## What is different

Spreading activation over graphs is not new (HippoRAG, GraphRAG, LightRAG,
RAPTOR). Spiyweb's claimed differentiators are elsewhere:

- **Redundancy becomes a vote, not noise.** When a near-duplicate is found
  during propagation, its edge is severed and its energy share redistributed —
  and the surviving idea's **vote count** goes up. Repetition turns into
  corpus-support evidence instead of burning context slots. To our knowledge
  this dynamic dedup-to-vote conversion has no published equivalent.
- **Honesty outputs.** Every retrieval returns a **confidence score** (total
  energy, node count, hop depth), **corpus-gap warnings** (two dense clusters
  with no bridge), **contradiction records** with a ready-made, LLM-free
  question for the user, and **activation paths** as explanations the LLM can
  cite. The retriever can say "I don't know, and here is why."
- **Coloured multi-seed bridging.** A decomposed query injects differently
  coloured seeds; a node where two colours meet is a **bridge** — exactly
  where a multi-hop answer lives.
- **The web stops itself.** Termination is a relative energy threshold, not a
  "return N results" parameter.

## Phase 1 plan

| Item | Decision |
|---|---|
| Benchmark | MuSiQue (multi-hop) |
| Gate | Beat **both** baselines — plain `top-k` and iterative retrieval — by a meaningful margin |
| Reference | HippoRAG results reported alongside |
| Metrics | 65% multi-hop accuracy + 35% Novelty@k, plus bridge-node recall |
| Embedding | multilingual-e5-large |
| Store | numpy + FAISS, single file |
| LLM (index-time only) | Local-first (Ollama); free APIs optional |
| Environment | Python 3.11 + uv · macOS / Windows / Linux |

Phase 1 also succeeds if it *fails* clearly: a reliable negative number is a
better outcome than a polished library built on an unmeasured assumption.

## Documentation

- [`docs/specs/2026-08-10-spiyweb-design.md`](docs/specs/2026-08-10-spiyweb-design.md) — full design specification
- [`memory/`](memory/) — decision log: every design choice with its rationale and rejected alternatives (in Turkish)
- [`CLAUDE.md`](CLAUDE.md) — condensed engineering ground truth

## Contributing

Design feedback is welcome right now — especially prior art for the
dedup-to-vote mechanism. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE) © 2026 Yigit Erdogan
