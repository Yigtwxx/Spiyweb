"""The HTTP surface. Read-only endpoints first; run control lives in `runs`.

Heavy endpoints are plain `def`, not `async def`, on purpose: `retrieve()` is
seconds of CPU work, and inside a coroutine it would block the event loop and
freeze the event stream that is watching a six-hour measurement. Starlette
runs plain `def` handlers in a threadpool, which is exactly right here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from server._paths import WEB_DIST
from server.inspect_api import QueryProblem
from server.inspect_api import run as run_inspect
from server.resources import CACHE, EDGE_LAYERS, MissingExtra
from server.runner import SUPERVISOR, RunRefused
from server.runner import plan as supervisor_plan
from server.schemas import (
    ApiError,
    ArtifactInfo,
    AtomHit,
    BootstrapReport,
    CacheEntryDto,
    CacheStatus,
    GpuDto,
    HopScore,
    IndexDetail,
    IndexSummary,
    InspectRequest,
    InspectResponse,
    LayerCount,
    LogTail,
    OllamaDto,
    PairedDiff,
    RunPlan,
    RunRequest,
    RunStartRequest,
    RunStatus,
    RunStopRequest,
    SystemStatus,
)
from server.stream import event_stream
from server.system import read_gpu, read_ollama

app = FastAPI(
    title="Spiyweb",
    description="Browser face of the spreading-activation rig.",
    version="0.1.0",
)

_ARTIFACTS = (
    "nodes.json",
    "vectors.npz",
    "entities.json",
    "propositions.json",
    "edges_nli.json",
    "meta.json",
    "results.json",
    "per_query.jsonl",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _stamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(
        timespec="seconds"
    )


def _summary(root: Path) -> IndexSummary:
    meta = CACHE.meta(root)
    return IndexSummary(
        name=root.name,
        dataset=CACHE.dataset_kind(root),
        corpus_chunks=meta.get("corpus_chunks"),  # type: ignore[arg-type]
        propositions=meta.get("propositions"),  # type: ignore[arg-type]
        questions=meta.get("questions"),  # type: ignore[arg-type]
        llm_model=meta.get("llm_model"),  # type: ignore[arg-type]
        nli_edges=meta.get("nli_edges"),  # type: ignore[arg-type]
        has_results=(root / "results.json").exists(),
        has_per_query=(root / "per_query.jsonl").exists(),
        modified_at=_stamp(root / "nodes.json"),
    )


@app.get("/api/indexes", response_model=list[IndexSummary])
def list_indexes() -> list[IndexSummary]:
    """Every index directory under `data/` that carries a node registry."""
    return [_summary(root) for root in CACHE.available()]


@app.get("/api/indexes/{name}", response_model=IndexDetail)
def index_detail(name: str) -> IndexDetail:
    """Artifacts and per-layer edge counts, so the UI never offers a dead knob."""
    try:
        root = CACHE.index_root(name)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    counts = CACHE.layer_counts(root)
    records = CACHE.node_records(root)
    return IndexDetail(
        **_summary(root).model_dump(),
        meta=CACHE.meta(root),
        layers=[
            LayerCount(layer=layer, edges=counts[layer], present=counts[layer] > 0)
            for layer in EDGE_LAYERS
        ],
        artifacts=[
            ArtifactInfo(
                name=filename,
                exists=(root / filename).exists(),
                bytes=(root / filename).stat().st_size
                if (root / filename).exists()
                else 0,
            )
            for filename in _ARTIFACTS
        ],
        nodes=len(records),
    )


@app.get("/api/atoms", response_model=list[AtomHit])
def list_atoms(
    index: str,
    q: str = "",
    limit: int = Query(default=40, ge=1, le=200),
    sample_size: int = 1000,
    sample_seed: int = 42,
) -> list[AtomHit]:
    """Search corpus atoms by id or title - no embedder, so no torch."""
    try:
        root = CACHE.index_root(index)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    corpus = CACHE.corpus(root, sample_size, sample_seed)
    titles, texts = corpus["titles"], corpus["texts"]
    needle = q.strip().casefold()
    hits: list[AtomHit] = []
    for record in CACHE.node_records(root):
        node_id = str(record["id"])
        title = titles.get(node_id, "")
        if (
            needle
            and needle not in node_id.casefold()
            and needle not in title.casefold()
        ):
            continue
        body = texts.get(node_id, "")
        hits.append(
            AtomHit(
                id=node_id,
                title=title or node_id,
                snippet=" ".join(body.split())[:160],
            )
        )
        if len(hits) >= limit:
            break
    return hits


@app.post("/api/inspect", response_model=InspectResponse)
def inspect(request: InspectRequest) -> InspectResponse:
    """One query: web, ledger, side-by-side comparison and honesty outputs."""
    from server.runner import SUPERVISOR

    status = SUPERVISOR.status()
    # "external" counts too: the guard protects the MACHINE, and a run
    # started from a terminal takes exactly as much RAM and CPU as one this
    # server started. Checking only "running" left the guard open in the one
    # case the header was already reporting on screen.
    if status.state in ("running", "stopping", "external"):
        started_here = status.state != "external"
        raise HTTPException(
            status_code=423,
            detail=(
                "a measurement run "
                + ("" if started_here else "started outside this server ")
                + "is using this machine; loading an index here would take "
                "RAM and CPU from it"
            ),
        )
    try:
        return run_inspect(request)
    except MissingExtra as error:
        return JSONResponse(  # type: ignore[return-value]
            status_code=503,
            content=ApiError(
                code="missing_extra", message=str(error), hint=error.hint
            ).model_dump(),
        )
    except (QueryProblem, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/system", response_model=SystemStatus)
def system_status() -> SystemStatus:
    """GPU headroom against the 88% rule, and whether Ollama is up."""
    gpu = read_gpu()
    ollama = read_ollama()
    return SystemStatus(
        gpu=GpuDto(**gpu.__dict__),
        ollama=OllamaDto(
            reachable=ollama.reachable,
            models=list(ollama.models),
            error=ollama.error,
        ),
        cache=_cache_status(),
        checked_at=_now(),
    )


def _cache_status() -> CacheStatus:
    entries = CACHE.entries()
    return CacheStatus(
        entries=[CacheEntryDto(**entry.__dict__) for entry in entries],
        total_bytes=sum(entry.approx_bytes for entry in entries),
        embedder_loaded=CACHE.embedder_loaded,
    )


@app.get("/api/cache", response_model=CacheStatus)
def cache_status() -> CacheStatus:
    return _cache_status()


@app.delete("/api/cache", response_model=CacheStatus)
def clear_cache() -> CacheStatus:
    """Drop every loaded artifact - the operator may want the RAM back."""
    CACHE.clear()
    return _cache_status()


@app.get("/api/results/{name}")
def results(name: str) -> dict[str, object]:
    """The run's `results.json` as written, plus its receipt."""
    try:
        root = CACHE.index_root(name)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    path = root / "results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} has no results.json yet")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["index"] = name
    payload["meta"] = CACHE.meta(root)
    payload["modified_at"] = _stamp(path)
    return payload


@app.get("/api/results/{name}/ci", response_model=BootstrapReport)
def results_ci(
    name: str,
    k: int = Query(default=5, ge=1, le=20),
    iterations: int = Query(default=10000, ge=200, le=50000),
) -> BootstrapReport:
    """Paired bootstrap CI over `per_query.jsonl`.

    The measurement protocol requires an interval, never a point estimate,
    and the harness only writes point estimates - so the interval is computed
    here, from the same per-question records.
    """
    from spiyweb.evaluation.stats import bootstrap_report

    try:
        root = CACHE.index_root(name)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    path = root / "per_query.jsonl"
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"{name} has no per_query.jsonl yet"
        )
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = bootstrap_report(records, k=k, iterations=iterations)
    return BootstrapReport(
        index=name,
        k=report.k,
        iterations=report.iterations,
        seed=report.seed,
        questions=report.questions,
        means=dict(report.means),
        bridge=dict(report.bridge),
        diffs=[
            PairedDiff(
                rival=diff.rival,
                mean=diff.mean,
                ci_low=diff.ci_low,
                ci_high=diff.ci_high,
                p=diff.p,
                significant=diff.significant,
            )
            for diff in report.diffs
        ],
        by_hop=[
            HopScore(hops=row.hops, questions=row.questions, scores=dict(row.scores))
            for row in report.by_hop
        ],
    )


# --- run control -----------------------------------------------------------


@app.get("/api/runs/current", response_model=RunStatus)
def run_current() -> RunStatus:
    return SUPERVISOR.status()


@app.post("/api/runs/plan", response_model=RunPlan)
def run_plan(request: RunRequest) -> RunPlan:
    """What would run. Nothing is launched here."""
    try:
        return supervisor_plan(request)
    except RunRefused as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/runs/start", response_model=RunStatus)
def run_start(body: RunStartRequest) -> RunStatus:
    """Launch, but only against a matching plan token and a typed word."""
    try:
        return SUPERVISOR.start(body.request, body.token, body.typed)
    except RunRefused as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/runs/stop", response_model=RunStatus)
def run_stop(body: RunStopRequest) -> RunStatus:
    try:
        return SUPERVISOR.stop(body.typed)
    except RunRefused as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/runs/release", response_model=RunStatus)
def run_release() -> RunStatus:
    """Clear a lock whose process is gone - never done automatically."""
    try:
        SUPERVISOR.release_lock()
    except RunRefused as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return SUPERVISOR.status()


@app.get("/api/runs/logs", response_model=LogTail)
def run_logs(tail: int = Query(default=200, ge=1, le=2000)) -> LogTail:
    out, err = SUPERVISOR.tail(tail)
    return LogTail(out=out, err=err)


@app.get("/api/runs/stream")
def run_stream() -> StreamingResponse:
    """Server-sent events: the server sets the pace, not the browser.

    `nvidia-smi` costs a process launch each time, so polling it from the
    client every two seconds for six hours would take real time away from the
    run being watched. Here each source ticks at its own natural frequency,
    and when nothing is running the stream sends only heartbeats.
    """
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# The built front end is served by this same process, so production has one
# origin and the dev proxy is the only place a second one exists.
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        candidate = WEB_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
