# Spiyweb

[![ci](https://github.com/Yigtwxx/spiyweb/actions/workflows/ci.yml/badge.svg)](https://github.com/Yigtwxx/spiyweb/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Graph-based retrieval for RAG, built on spreading activation — like a spider
web.**

Instead of cutting retrieval off at `top-k`, the query is injected into a
vector graph as an **energy seed** and spreads outward with decay. Strongly
related nodes light up first; weakly related but genuinely connected nodes
light up later, through multiple hops. The answer is built from the whole web,
not from one cluster of near-duplicates.

> **Status: Phase 1 — the propagation core runs; everything around it is next.**
> `spiyweb.propagate` reproduces the worked example below, with the documented
> trace pinned as a test. Edge builders, the vector store, the evaluation
> harness and the developer UI are not written yet. The design has been public
> from day one so the idea can be judged — and challenged — early.

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

## Try it

The core is pure Python with no dependencies. Similarities come from the
caller; `core/` never computes them.

```bash
git clone https://github.com/Yigtwxx/spiyweb && cd spiyweb
uv sync --group dev && uv run pytest
```

```python
from spiyweb import Graph, propagate

graph = Graph.from_edges(
    [
        ("A", "A_dup", 0.0),  # near duplicate, edge suppressed by dedup
        ("A", "B", 0.8),
        ("A", "D", 0.4),
        ("C", "D", 0.6),
        ("C", "E", 0.3),
        ("D", "F", 0.5),
    ]
)
result = propagate(graph, seeds={"A": 0.9, "C": 0.7})

[(node, round(energy, 3)) for node, energy in result.ranked()]
# [('A', 5.625), ('C', 4.375), ('D', 2.875), ('B', 2.25), ('F', 1.725)]
result.activations["D"].contributors  # ('A', 'C') — converging evidence
result.stop_reason  # 'threshold' — the web stopped itself
```

`E` is missing because 0.875 of energy reached it against a floor of 1.5, and
`A_dup` is missing because its edge was suppressed and its share redistributed.
Neither outcome came from a result-count parameter.

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
