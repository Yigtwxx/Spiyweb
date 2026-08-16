# Spiyweb - MuSiQue report

1000 questions; the primary number is S@5 (0.65 * support recall + 0.35 * novelty).

## Systems

| system | k | support recall | novelty | S | bridge recall |
|---|---|---|---|---|---|
| topk | 2 | 0.732 | 0.000 | 0.475 | 0.828 |
| topk | 5 | 0.865 | 0.000 | 0.562 | 0.926 |
| topk | 10 | 0.920 | 0.000 | 0.598 | 0.970 |
| web | 2 | 0.748 | 0.106 | 0.523 | 0.810 |
| web | 5 | 0.910 | 0.088 | 0.623 | 0.944 |
| web | 10 | 0.944 | 0.055 | 0.633 | 0.970 |
| iterative | 2 | 0.820 | 0.142 | 0.583 | 0.835 |
| iterative | 5 | 0.944 | 0.084 | 0.643 | 0.959 |
| iterative | 10 | 0.962 | 0.057 | 0.645 | 0.978 |

Novelty is measured against plain `top-k` at the same cutoff, so `top-k`'s own novelty is 0 by construction.

## Reference (reported, not reproduced)

HippoRAG (ColBERTv2), arXiv:2405.14831 Table 2, MuSiQue: R@2 0.409, R@5 0.519. Informative only - different sample, never a gate (D29).

## Objective by hop count (S@5)

| hops | questions | topk | web | iterative |
|---|---|---|---|---|
| 2 | 1000 | 0.562 | 0.623 | 0.643 |

## Web stop reasons

| reason | count |
|---|---|
| threshold | 1000 |

