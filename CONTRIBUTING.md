# Contributing to Spiyweb

Thanks for your interest. A few things to know before contributing.

## Project status

Spiyweb is in **Phase 1 — design complete, implementation pending**. Right now
the most valuable contributions are not pull requests but:

- **Prior art.** If you know a published equivalent of the dynamic
  redundancy-to-vote mechanism (edge severed, vote incremented, at query
  time), please open an issue. Finding out early is a gift.
- **Design critique.** Holes in the propagation rules, the contradiction
  handling, or the evaluation plan. The settled invariants and their reasons
  are in [`CLAUDE.md`](CLAUDE.md); the full specification and the decision log
  are kept privately, so quote the rule you are challenging.
- **Benchmark suggestions.** Multi-hop datasets with little "answer already in
  one chunk" leakage, beyond MuSiQue.

## Ground rules

- Open an **issue first** for anything non-trivial. The design has settled
  invariants (see `CLAUDE.md` §2) — changes to those need discussion, not a
  surprise PR.
- Be honest about trade-offs. This project prefers a reliable negative result
  over a polished illusion.

## When code lands (Phase 1 implementation)

- **Python 3.11 + uv**, `src/` layout.
- Type annotations on every function signature.
- `ruff check .`, `ruff format --check .` and `pytest` must all be clean.
- All identifiers, comments and docstrings in English.
- Device selection order is fixed: **CUDA → MPS → CPU**.
- No magic numbers — every tunable is a documented `config.py` dataclass field.
- Every mechanism must be individually disableable from config; ablation is
  how this project proves itself.
- `core/` stays pure: no I/O, no LLM, no network, no filesystem.
- Public API changes go through `CHANGELOG.md`. `spiyweb.__all__` is
  snapshot-tested in `tests/test_public_api.py`, so an undeclared export
  fails CI before it reaches anyone.
- No secrets in source; read from environment variables.

## Commit messages

Conventional Commits:

```
<type>(<scope>): <short imperative description>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`. Keep the subject
under 72 characters; explain *why* in the body.

## License

By contributing you agree that your contributions are licensed under the
[Apache-2.0](LICENSE) license.
