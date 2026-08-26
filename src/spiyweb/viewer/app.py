"""The viewer's HTTP surface: recorded calls first, a live query on top.

Deliberately small. The repository's `server/` is a measurement rig - it
scans `data/`, supervises six-hour benchmark runs, polls the GPU - and none
of that belongs in a wheel somebody installed to debug their own retrieval.
What ships is the part that answers "what did MY application retrieve, and
why": a list of recorded calls, one record in full, and - only when the
caller handed us a live index - a box to ask a new question with.

Every `/api/*` route is behind the process token. The bundle's own assets are
not: they are the same JavaScript for everybody and carry no corpus, while
gating them would mean threading the token through every `<script src>`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from spiyweb import __version__
from spiyweb.viewer.bundle import ENTRY, MissingBundle, find_bundle
from spiyweb.viewer.security import TOKEN_HEADER, TOKEN_PARAM, TokenGuard

if TYPE_CHECKING:
    from pathlib import Path

    from spiyweb.session import SpiywebIndex
    from spiyweb.trace import TraceRecord
    from spiyweb.viewer.sources import TraceSource

__all__ = ["build_app"]

DEFAULT_PAGE_SIZE = 50
"""Records per page of the list view. Summaries only, so this is cheap."""

DEFAULT_PROFILE = "explore"
"""What the ASK box runs when the caller names no profile.

The library default splits 10.0 energy among 5 seeds and stops at 15% of it,
so nothing clears the threshold and the page draws five dots at hop 0 - the
mechanism this viewer exists to show, not showing. Same reasoning and same
value as the terminal's default; the answer carries the profile it used so
the picture is never unexplained."""

MAX_PAGE_SIZE = 500
"""Ceiling on `limit`, so one request cannot ask for a whole ring buffer of
full records by accident."""


def _summary(record: TraceRecord) -> dict[str, Any]:
    """A row of the list - never the nodes, edges or texts.

    A hundred summaries are a few kilobytes; a hundred full records are
    megabytes of passage text, and the list view draws none of it.
    """
    return {
        "trace_id": record.trace_id,
        "sequence": record.sequence,
        "recorded_at": record.recorded_at,
        "kind": record.kind,
        "query": record.query,
        "node_count": record.node_count,
        "total_energy": record.total_energy,
        "hops_used": record.hops_used,
        "stop_reason": record.stop_reason,
        "elapsed_ms": record.elapsed_ms,
        "dedup_mode": record.dedup_mode,
        "edge_count": len(record.edges),
        "bridge_count": len(record.bridges),
        "balanced": None if record.ledger is None else record.ledger.balanced,
    }


def build_app(
    source: TraceSource,
    *,
    guard: TokenGuard | None = None,
    index: SpiywebIndex | None = None,
    bundle: Path | str | None = None,
) -> FastAPI:
    """Assemble the viewer around one source of records.

    `index` is optional and is what separates the two products: without it
    this is a pure trace viewer that could be reading a file from another
    machine; with it the same page also offers a live query, because the
    caller already had the index open and a second copy of it is the thing
    D38 refused to build.
    """
    token = guard if guard is not None else TokenGuard()
    app = FastAPI(
        title="Spiyweb viewer",
        description="What this application's retrieval actually did.",
        version=__version__,
    )

    def _authorize(
        request: Request,
        token_param: str | None = Query(default=None, alias=TOKEN_PARAM),
    ) -> None:
        supplied = token_param or request.headers.get(TOKEN_HEADER)
        if not token.accepts(supplied):
            raise HTTPException(
                status_code=401,
                detail=(
                    "this viewer needs the token from the URL it printed; "
                    "open that link rather than the bare address"
                ),
            )

    guarded = [Depends(_authorize)]

    @app.get("/api/capabilities", dependencies=guarded)
    def capabilities() -> dict[str, Any]:
        """What this viewer can do, so one front end serves both modes."""
        records = source.records()
        # `live` means a query can actually RUN, not merely that an index is
        # attached. Offering a search box on an install with no embedder is
        # a button that returns 500, which is worse than no button.
        live = index is not None and index.can_query
        return {
            "version": __version__,
            "mode": source.kind,
            "origin": source.origin,
            "live": live,
            "colored": live,
            "count": len(records),
            "runs": False,
            "bundle": _bundle_ok,
        }

    @app.get("/api/traces", dependencies=guarded)
    def traces(
        limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        """Newest first - the call somebody is debugging is the last one."""
        records = source.records()
        newest = list(reversed(records))
        page = newest[offset : offset + limit]
        return {
            "total": len(records),
            "offset": offset,
            "limit": limit,
            "traces": [_summary(record) for record in page],
        }

    @app.get("/api/traces/{trace_id}", dependencies=guarded)
    def trace(trace_id: str) -> dict[str, Any]:
        record = source.get(trace_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no trace {trace_id!r}")
        return record.to_dict()

    @app.get("/api/traces/{trace_id}/scene", dependencies=guarded)
    def scene(
        trace_id: str,
        max_nodes: int = Query(default=300, ge=1, le=2000),
        max_edges: int = Query(default=1500, ge=0, le=20000),
        edge_mode: str = Query(default="induced", pattern="^(induced|contributors)$"),
        label_top_n: int = Query(default=12, ge=0, le=200),
    ) -> dict[str, Any]:
        """The recorded call, laid out by the SAME builder the live rig uses.

        Laid out here rather than in the browser because `spiyweb.scene` is
        already the one picture both front ends draw (Faz 2.2), and a second
        layout in TypeScript would drift from it within a month.
        """
        record = source.get(trace_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no trace {trace_id!r}")
        from spiyweb.scene import ViewConfig
        from spiyweb.viewer.scenes import scene_payload

        return scene_payload(
            record,
            ViewConfig(
                max_nodes=max_nodes,
                max_edges=max_edges,
                edge_mode=edge_mode,
                label_top_n=label_top_n,
            ),
        )

    @app.post("/api/query", dependencies=guarded)
    def query(payload: dict[str, Any]) -> dict[str, Any]:
        """Ask a new question - only where a live index was handed over."""
        if index is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "this viewer is reading recorded traces and holds no "
                    "index, so it cannot run a new query. Start it from an "
                    "open SpiywebIndex if you want one"
                ),
            )
        parts = payload.get("parts") or None
        profile = payload.get("profile") or DEFAULT_PROFILE
        text = str(payload.get("query") or "").strip()
        if not parts and not text:
            raise HTTPException(status_code=400, detail="ask a question first")
        try:
            if parts:
                answer = index.retrieve_colored(
                    {str(k): str(v) for k, v in parts.items()}, profile=profile
                )
            else:
                answer = index.retrieve(text, profile=profile)
        except ImportError as missing:
            # The embedder is imported on the first query, so a missing
            # extra surfaces HERE rather than at startup. Reported as the
            # pip line that fixes it; a 500 tells the reader nothing.
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{missing} - embedding a question needs the model: "
                    'pip install "spiyweb[embed]"'
                ),
            ) from missing
        if answer.trace is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "the query ran but tracing is switched off on this index, "
                    "so there is nothing to show; open it without "
                    "TraceConfig(enabled=False)"
                ),
            )
        return answer.trace.to_dict()

    _bundle_ok = _mount_bundle(app, bundle)
    return app


def _mount_bundle(app: FastAPI, bundle: Path | str | None) -> bool:
    """Serve the built page, or leave a route that explains its absence.

    A missing bundle is not fatal: the JSON API is the load-bearing half and
    a caller may well be driving it themselves. So the failure surfaces at
    `/`, where a person is looking, with the message that says which case
    this is.
    """
    try:
        root = find_bundle(bundle)
    except MissingBundle as absent:
        reason = str(absent)

        @app.get("/")
        def no_page() -> JSONResponse:
            return JSONResponse(status_code=503, content={"detail": reason})

        return False

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(root / ENTRY)

    # `html=True` so a client-side route reloads into the page rather than a
    # 404; mounted last so nothing shadows the API.
    app.mount("/", StaticFiles(directory=root, html=True), name="bundle")
    return True
