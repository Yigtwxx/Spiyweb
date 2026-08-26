"""The three rules that keep a debugging aid from becoming an open door.

The viewer exists to be started from inside somebody's application, often on
a laptop on a shared network, sometimes on a box that also serves real
traffic. What it hands out is passage text and query history, which is the
corpus and the users' questions. So the bar is not "hard to stumble into", it
is "not reachable from another machine at all".

1. **Loopback only.** `127.0.0.1`, never `0.0.0.0`. Binding the wildcard
   would publish the corpus to the network, and there is no configuration
   knob for it here - a caller who genuinely wants that can put their own
   proxy in front and own the decision.
2. **A port nobody chose.** Port `0` lets the OS pick a free one, because the
   caller's own service may well be on 8000 and a viewer that steals the port
   of the thing it is debugging is worse than no viewer.
3. **An unguessable token.** Minted per server process and required on every
   API request. It rides in the URL the caller is handed, and the page sends
   it back on each call.

The token is per PROCESS, not per request: a page that loads its own assets
and then polls would burn a strictly single-use token on the first fetch. What
"single use" buys - a link that cannot be replayed later - comes from the
process instead, since the token dies with the server that minted it.
"""

from __future__ import annotations

import secrets

__all__ = ["LOOPBACK", "TOKEN_HEADER", "TOKEN_PARAM", "TokenGuard", "new_token"]

LOOPBACK = "127.0.0.1"
"""The only host this server ever binds. Not a default - the only value."""

TOKEN_PARAM = "token"
"""Query parameter carrying the token, as in the URL the caller is handed."""

TOKEN_HEADER = "x-spiyweb-token"
"""Header the page may use instead, so the token stays out of request logs."""

_TOKEN_BYTES = 32
"""256 bits. Long enough that guessing is not a threat model, short enough
that the URL is still something a person can paste."""


def new_token() -> str:
    """A fresh, unguessable token for one viewer process."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


class TokenGuard:
    """Holds one process's token and answers whether a request carries it."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token if token else new_token()

    @property
    def token(self) -> str:
        return self._token

    def accepts(self, candidate: str | None) -> bool:
        """Constant-time comparison - a timing oracle is still an oracle."""
        if not candidate:
            return False
        return secrets.compare_digest(candidate, self._token)


def ensure_loopback(host: str) -> str:
    """Refuse any host but the loopback, loudly rather than quietly.

    A silent rewrite to `127.0.0.1` would leave the caller believing they had
    published the viewer; a refusal makes them decide.
    """
    if host != LOOPBACK:
        raise ValueError(
            f"the viewer binds {LOOPBACK} only, never {host!r}: it serves "
            "passage text and query history, so publishing it is a decision "
            "for a proxy you control, not a keyword argument here"
        )
    return host
