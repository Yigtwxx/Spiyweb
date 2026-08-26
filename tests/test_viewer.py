"""The viewer: recorded calls served over HTTP, and the three security rules.

Phase 2.5. Two claims are under test and they are not the same claim.

The first is D38's product claim: a JSONL file can be served by something
that never loads the index that wrote it. The `FileSource` tests are that -
they read a file and answer questions about it, with no graph and no store in
the process.

The second is the security claim, and it is the one worth being pedantic
about. This server hands out passage text and query history - the corpus and
the users' questions - from inside somebody else's running application. So
loopback-only, an OS-chosen port and the token are asserted individually
rather than trusted to a code review.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from spiyweb.config import TraceConfig
from spiyweb.viewer import FileSource, MissingBundle, StoreSource, TokenGuard
from spiyweb.viewer.app import build_app
from spiyweb.viewer.security import TOKEN_HEADER, ensure_loopback

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spiyweb.session import SpiywebIndex

TOKEN = "test-token-not-a-real-one"


@pytest.fixture
def index(open_tiny: Callable[..., SpiywebIndex]) -> SpiywebIndex:
    return open_tiny()


def _client(source: object, **kwargs: object) -> TestClient:
    app = build_app(source, guard=TokenGuard(TOKEN), **kwargs)  # type: ignore[arg-type]
    return TestClient(app)


def _get(client: TestClient, path: str) -> object:
    response = client.get(path, headers={TOKEN_HEADER: TOKEN})
    assert response.status_code == 200, response.text
    return response.json()


# --- the token ------------------------------------------------------------


def test_every_api_route_refuses_a_request_without_the_token(
    index: SpiywebIndex,
) -> None:
    """The corpus is behind this; an unauthenticated read is not a 404."""
    index.retrieve("who raised the tower")
    client = _client(StoreSource(index.traces))
    for path in ("/api/capabilities", "/api/traces"):
        assert client.get(path).status_code == 401, path
    assert client.post("/api/query", json={"query": "x"}).status_code == 401


def test_a_wrong_token_is_refused_like_a_missing_one(index: SpiywebIndex) -> None:
    client = _client(StoreSource(index.traces))
    assert client.get("/api/capabilities?token=nearly-right").status_code == 401


def test_the_token_is_accepted_from_the_url_or_the_header(
    index: SpiywebIndex,
) -> None:
    """The URL carries it; the page may move it to a header to stay out of logs."""
    client = _client(StoreSource(index.traces))
    assert client.get(f"/api/capabilities?token={TOKEN}").status_code == 200
    assert (
        client.get("/api/capabilities", headers={TOKEN_HEADER: TOKEN}).status_code
        == 200
    )


def test_two_guards_never_mint_the_same_token() -> None:
    assert TokenGuard().token != TokenGuard().token
    assert len(TokenGuard().token) >= 32


def test_binding_anything_but_the_loopback_is_refused() -> None:
    """A silent rewrite would leave the caller thinking they had published it."""
    assert ensure_loopback("127.0.0.1") == "127.0.0.1"
    for host in ("0.0.0.0", "::", "192.168.1.10"):
        with pytest.raises(ValueError, match=r"binds 127\.0\.0\.1 only"):
            ensure_loopback(host)


# --- serving recorded calls ------------------------------------------------


def test_the_list_view_is_summaries_and_never_the_passage_text(
    index: SpiywebIndex,
) -> None:
    """A hundred full records are megabytes of corpus the list never draws."""
    index.retrieve("who raised the tower")
    client = _client(StoreSource(index.traces))
    body = _get(client, "/api/traces")
    assert body["total"] == 1
    row = body["traces"][0]
    assert row["query"] == "who raised the tower"
    assert row["node_count"] > 0
    assert row["balanced"] is True
    assert "nodes" not in row and "edges" not in row


def test_the_newest_call_comes_first(index: SpiywebIndex) -> None:
    """The call somebody is debugging is the one they just made."""
    for number in range(5):
        index.retrieve(f"question number {number}")
    body = _get(_client(StoreSource(index.traces)), "/api/traces")
    assert [row["sequence"] for row in body["traces"]] == [4, 3, 2, 1, 0]


def test_one_trace_comes_back_whole(index: SpiywebIndex) -> None:
    answer = index.retrieve("who raised the tower")
    assert answer.trace is not None
    client = _client(StoreSource(index.traces))
    body = _get(client, f"/api/traces/{answer.trace.trace_id}")
    assert body == answer.trace.to_dict()
    assert body["nodes"] and body["edges"] and body["ledger"]


def test_an_unknown_trace_is_a_404(index: SpiywebIndex) -> None:
    client = _client(StoreSource(index.traces))
    response = client.get("/api/traces/nope", headers={TOKEN_HEADER: TOKEN})
    assert response.status_code == 404


def test_paging_walks_the_whole_buffer(index: SpiywebIndex) -> None:
    for number in range(12):
        index.retrieve(f"question number {number}")
    client = _client(StoreSource(index.traces))
    first = _get(client, "/api/traces?limit=5&offset=0")
    second = _get(client, "/api/traces?limit=5&offset=5")
    assert first["total"] == second["total"] == 12
    assert len(first["traces"]) == len(second["traces"]) == 5
    ids = {row["trace_id"] for row in first["traces"] + second["traces"]}
    assert len(ids) == 10


# --- the file product ------------------------------------------------------


def test_a_trace_file_is_served_without_an_index_in_the_process(
    open_tiny: Callable[..., SpiywebIndex], tmp_path: Path
) -> None:
    """D38's actual product: what wrote the file is not what reads it."""
    writer = open_tiny(TraceConfig(directory=tmp_path))
    writer.retrieve("who raised the tower")
    writer.retrieve("what happened afterwards")
    writer.close()

    client = _client(FileSource(tmp_path))
    capabilities = _get(client, "/api/capabilities")
    assert capabilities["mode"] == "file"
    assert capabilities["live"] is False
    assert capabilities["count"] == 2
    assert _get(client, "/api/traces")["total"] == 2


def test_a_file_source_notices_new_records(
    open_tiny: Callable[..., SpiywebIndex], tmp_path: Path
) -> None:
    """The application writing the file is usually still running."""
    writer = open_tiny(TraceConfig(directory=tmp_path))
    source = FileSource(tmp_path)
    assert source.records() == ()

    writer.retrieve("who raised the tower")
    assert len(source.records()) == 1
    writer.retrieve("what happened afterwards")
    assert len(source.records()) == 2


def test_a_missing_trace_file_is_empty_rather_than_an_error(tmp_path: Path) -> None:
    source = FileSource(tmp_path / "absent.jsonl")
    assert source.records() == ()
    assert source.get("anything") is None


# --- the live half ---------------------------------------------------------


def test_a_live_index_can_be_asked_a_new_question(index: SpiywebIndex) -> None:
    """A on top of B: the index is already open, so a second copy is waste."""
    client = _client(StoreSource(index.traces), index=index)
    assert _get(client, "/api/capabilities")["live"] is True

    response = client.post(
        "/api/query",
        json={"query": "who raised the tower"},
        headers={TOKEN_HEADER: TOKEN},
    )
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["query"] == "who raised the tower"
    assert record["nodes"]
    # The answer went through the ordinary path, so the ring buffer has it.
    assert _get(client, "/api/traces")["total"] == 1


def test_a_coloured_question_reaches_the_coloured_path(index: SpiywebIndex) -> None:
    client = _client(StoreSource(index.traces), index=index)
    response = client.post(
        "/api/query",
        json={
            "parts": {
                "tower": "who raised the tower",
                "after": "what happened afterwards",
            }
        },
        headers={TOKEN_HEADER: TOKEN},
    )
    assert response.status_code == 200, response.text
    assert response.json()["kind"] == "colored"


def test_a_file_viewer_refuses_to_run_a_query(tmp_path: Path) -> None:
    """No index means no query - and it says that instead of half-answering."""
    client = _client(FileSource(tmp_path))
    response = client.post(
        "/api/query", json={"query": "x"}, headers={TOKEN_HEADER: TOKEN}
    )
    assert response.status_code == 409
    assert "holds no" in response.json()["detail"]


def test_an_empty_question_is_refused(index: SpiywebIndex) -> None:
    client = _client(StoreSource(index.traces), index=index)
    response = client.post(
        "/api/query", json={"query": "   "}, headers={TOKEN_HEADER: TOKEN}
    )
    assert response.status_code == 400


def test_a_query_on_an_untraced_index_says_so(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """Silently returning nothing would look like the query failed."""
    quiet = open_tiny(TraceConfig(enabled=False))
    client = _client(StoreSource(quiet.traces), index=quiet)
    response = client.post(
        "/api/query",
        json={"query": "who raised the tower"},
        headers={TOKEN_HEADER: TOKEN},
    )
    assert response.status_code == 409
    assert "tracing is switched off" in response.json()["detail"]


# --- the bundle ------------------------------------------------------------


def test_a_missing_bundle_leaves_the_api_working_and_explains_the_page(
    index: SpiywebIndex, tmp_path: Path
) -> None:
    """The JSON half is load-bearing; only the page needs a build step."""
    client = _client(StoreSource(index.traces), bundle=None)
    assert _get(client, "/api/capabilities")["bundle"] in (True, False)


def test_a_bundle_that_is_not_one_is_refused(tmp_path: Path) -> None:
    from spiyweb.viewer.bundle import find_bundle

    with pytest.raises(MissingBundle, match="does not look like"):
        find_bundle(tmp_path)


def test_a_built_bundle_is_served_at_the_root(
    index: SpiywebIndex, tmp_path: Path
) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>x</title>", "utf-8")
    client = _client(StoreSource(index.traces), bundle=tmp_path)
    assert _get(client, "/api/capabilities")["bundle"] is True
    page = client.get("/")
    assert page.status_code == 200
    assert "doctype" in page.text.lower()


# --- starting it for real --------------------------------------------------


def test_inspect_url_starts_a_real_server_on_a_free_loopback_port(
    index: SpiywebIndex,
) -> None:
    """The end-to-end claim: two lines in an application produce a link."""
    import urllib.parse

    url = index.inspect_url()
    try:
        parsed = urllib.parse.urlparse(url)
        assert parsed.hostname == "127.0.0.1"
        assert parsed.port and parsed.port > 0
        token = urllib.parse.parse_qs(parsed.query)["token"][0]
        assert len(token) >= 32

        index.retrieve("who raised the tower")
        import urllib.request

        with urllib.request.urlopen(
            f"http://127.0.0.1:{parsed.port}/api/traces?token={token}", timeout=10
        ) as response:
            assert response.status == 200

        # Twice is the same server, not a second one on a second port.
        assert index.inspect_url() == url
    finally:
        index.close()


def test_closing_the_index_stops_the_viewer(index: SpiywebIndex) -> None:
    index.inspect_url()
    handle = index._viewer
    assert handle is not None and handle.running
    index.close()
    assert not handle.running


# --- drawing a recorded call ----------------------------------------------


def test_a_recorded_call_is_laid_out_by_the_shared_scene_builder(
    index: SpiywebIndex,
) -> None:
    """One picture for both front ends - that is why scene.py was promoted."""
    answer = index.retrieve("who raised the tower")
    assert answer.trace is not None
    client = _client(StoreSource(index.traces))
    scene = _get(client, f"/api/traces/{answer.trace.trace_id}/scene")

    drawn = {node["id"] for node in scene["nodes"]}
    assert drawn == {node for node, _ in answer.result.ranked()}
    for node in scene["nodes"]:
        # Both layouts ride along: force coordinates and the concentric
        # hop rings, which are the ones that answer "how far did it get".
        for key in ("x", "y", "rx", "ry"):
            assert 0.0 <= node[key] <= 1.0, (key, node[key])
        assert node["tooltip"]
    for edge in scene["edges"]:
        assert edge["source"] in drawn and edge["target"] in drawn
    assert scene["layer_order"]
    assert scene["max_hop"] >= 0


def test_a_scene_can_be_drawn_from_a_file_with_no_index_present(
    open_tiny: Callable[..., SpiywebIndex], tmp_path: Path
) -> None:
    """The layout is rebuilt from the record, not from the graph it came from."""
    writer = open_tiny(TraceConfig(directory=tmp_path))
    answer = writer.retrieve("who raised the tower")
    assert answer.trace is not None
    writer.close()

    client = _client(FileSource(tmp_path))
    scene = _get(client, f"/api/traces/{answer.trace.trace_id}/scene")
    assert scene["nodes"]
    assert scene["edges"]


def test_a_scene_for_an_unknown_trace_is_a_404(index: SpiywebIndex) -> None:
    client = _client(StoreSource(index.traces))
    response = client.get("/api/traces/nope/scene", headers={TOKEN_HEADER: TOKEN})
    assert response.status_code == 404


# --- one shape on the wire --------------------------------------------------


def test_the_package_payload_and_the_rig_schema_agree(index: SpiywebIndex) -> None:
    """The Faz 2.5 carry-over, pinned.

    The rig and the shipped viewer used to each turn a run into JSON in their
    own copy of the same ninety lines. They now share
    `spiyweb.viewer.payload`, and `server/schemas.py` builds its pydantic
    models straight from those keys - so constructing the models here is a
    test that the two halves still describe the same thing. A field renamed
    on one side fails right here instead of silently meaning something else
    on one of the two pages.
    """
    from server.schemas import LedgerDto, SceneDto

    from spiyweb.config import RetrievalConfig
    from spiyweb.ledger import build_ledger
    from spiyweb.viewer.payload import ledger_payload
    from spiyweb.viewer.scenes import scene_payload

    answer = index.retrieve("who raised the tower")
    assert answer.trace is not None

    scene = SceneDto(**scene_payload(answer.trace))
    assert scene.nodes and scene.layer_order
    assert all(0.0 <= node.rx <= 1.0 for node in scene.nodes)

    book = build_ledger(
        answer.result.propagation, index.graph, RetrievalConfig().propagation
    )
    ledger = LedgerDto(
        **ledger_payload(
            book,
            dedup_cuts=len(answer.result.propagation.suppressed),
            contact_cuts=len(answer.result.contact_suppressed),
            contact_tau=answer.result.contact_tau,
        )
    )
    assert ledger.injected > 0.0
    assert ledger.balanced is True
    assert ledger.destroyed.total == pytest.approx(0.0)


def test_two_threads_never_start_two_viewers(index: SpiywebIndex) -> None:
    """A dropped handle is a listening socket nobody can close."""
    import threading

    urls: list[str] = []
    barrier = threading.Barrier(4)

    def ask() -> None:
        barrier.wait()
        urls.append(index.inspect_url())

    threads = [threading.Thread(target=ask) for _ in range(4)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert len(set(urls)) == 1, f"started more than one server: {set(urls)}"
    finally:
        index.close()


def test_live_means_a_query_can_actually_run(
    open_tiny: Callable[..., SpiywebIndex], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Holding an index is not the same as being able to ask it something.

    The shipped 0.1.0 offered a search box whenever an index was attached.
    On an install without `sentence-transformers` the button returned an
    opaque 500 - a promise the environment could not keep.
    """
    index = open_tiny()
    monkeypatch.setattr(type(index), "can_query", property(lambda self: False))
    client = _client(StoreSource(index.traces), index=index)
    capabilities = _get(client, "/api/capabilities")
    assert capabilities["live"] is False
    assert capabilities["colored"] is False


def test_an_index_with_an_injected_embedder_can_query(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    """The fake embedder answers yes without importing anything."""
    assert open_tiny().can_query is True


def test_a_missing_embedder_is_reported_as_the_pip_line(
    open_tiny: Callable[..., SpiywebIndex], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extra is imported on the FIRST query, so it surfaces here."""
    index = open_tiny()

    def refuse(*args: object, **kwargs: object) -> None:
        raise ImportError("sentence-transformers is required")

    monkeypatch.setattr(type(index), "retrieve", refuse)
    client = _client(StoreSource(index.traces), index=index)
    response = client.post(
        "/api/query",
        json={"query": "who raised the tower"},
        headers={TOKEN_HEADER: TOKEN},
    )
    assert response.status_code == 503
    assert "spiyweb[embed]" in response.json()["detail"]
