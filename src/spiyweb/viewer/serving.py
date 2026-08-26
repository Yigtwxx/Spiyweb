"""Starting the viewer from inside a running application, and stopping it.

The shape this has to fit is a person adding two lines to their own program:

    index = spiyweb.open_index("my-index")
    print(index.inspect_url())

Which means it cannot block, cannot take over the process, and cannot fight
the caller's own web server for a port. So: uvicorn on a daemon thread, port
`0` so the OS picks, `127.0.0.1` so nothing off the machine can reach it, and
a token in the URL. The handle it returns can stop the server again, and the
thread is a daemon so a caller who forgets does not hang their own exit.

Reading the port back is the fiddly part and it is done properly here rather
than by guessing: the thread waits until uvicorn reports it has started, then
asks the bound socket what port it got. Guessing would mean printing a URL
before anything listens on it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from spiyweb.viewer.app import build_app
from spiyweb.viewer.security import LOOPBACK, TOKEN_PARAM, TokenGuard, ensure_loopback
from spiyweb.viewer.sources import FileSource, StoreSource

if TYPE_CHECKING:
    from pathlib import Path

    from spiyweb.session import SpiywebIndex
    from spiyweb.viewer.sources import TraceSource

__all__ = ["ViewerHandle", "serve", "serve_file"]

STARTUP_TIMEOUT = 20.0
"""Seconds to wait for uvicorn to bind. Generous: the first import of
FastAPI on a cold filesystem is not instant, and a spurious timeout here
would look like a bug in the caller's application."""

SHUTDOWN_TIMEOUT = 5.0
"""Seconds a stopped server gets to finish in-flight requests."""


@dataclass
class ViewerHandle:
    """A running viewer: where it is, and how to stop it.

    Usable as a context manager, which is what a test or a short script
    wants; a long-lived application just keeps the handle and lets the
    daemon thread live as long as the process does.
    """

    url: str
    port: int
    token: str
    _server: object = field(repr=False)
    _thread: threading.Thread = field(repr=False)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> None:
        """Ask the server to exit and wait briefly for it."""
        self._server.should_exit = True  # type: ignore[attr-defined]
        self._thread.join(timeout=SHUTDOWN_TIMEOUT)

    def __enter__(self) -> ViewerHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def __str__(self) -> str:
        return self.url


def serve(
    source: TraceSource,
    *,
    index: SpiywebIndex | None = None,
    host: str = LOOPBACK,
    port: int = 0,
    bundle: Path | str | None = None,
    token: str | None = None,
) -> ViewerHandle:
    """Start the viewer in a background thread and return where it lives."""
    import uvicorn

    ensure_loopback(host)
    guard = TokenGuard(token)
    app = build_app(source, guard=guard, index=index, bundle=bundle)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        # The caller's application owns the process; installing signal
        # handlers from a library thread would steal its Ctrl-C.
        lifespan="on",
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

    thread = threading.Thread(target=server.run, name="spiyweb-viewer", daemon=True)
    thread.start()
    bound = _wait_for_port(server, thread)
    return ViewerHandle(
        url=f"http://{host}:{bound}/?{TOKEN_PARAM}={guard.token}",
        port=bound,
        token=guard.token,
        _server=server,
        _thread=thread,
    )


def serve_file(
    path: Path | str,
    *,
    host: str = LOOPBACK,
    port: int = 0,
    bundle: Path | str | None = None,
    token: str | None = None,
) -> ViewerHandle:
    """Open a JSONL trace file - no index, no model, no vector store.

    This is the D38 product in one line: the traces an application wrote,
    read back by something that never loads what wrote them.
    """
    return serve(FileSource(path), host=host, port=port, bundle=bundle, token=token)


def serve_index(
    index: SpiywebIndex,
    *,
    host: str = LOOPBACK,
    port: int = 0,
    bundle: Path | str | None = None,
    token: str | None = None,
) -> ViewerHandle:
    """Show a live index's own ring buffer, and let it be queried."""
    return serve(
        StoreSource(index.traces, origin=str(index.layout.root)),
        index=index,
        host=host,
        port=port,
        bundle=bundle,
        token=token,
    )


def _wait_for_port(server: object, thread: threading.Thread) -> int:
    """Block until uvicorn is listening, then report the port it got.

    Polling `server.started` and not a fixed sleep: the point of the wait is
    that the URL handed back is one that already answers.
    """
    deadline = threading.Event()
    waited = 0.0
    step = 0.01
    while waited < STARTUP_TIMEOUT:
        if getattr(server, "started", False):
            for bound in getattr(server, "servers", ()):
                for socket in bound.sockets:
                    return int(socket.getsockname()[1])
            break
        if not thread.is_alive():
            raise RuntimeError(
                "the viewer's server thread exited before it started "
                "listening; the port may already be taken"
            )
        deadline.wait(step)
        waited += step
    raise TimeoutError(
        f"the viewer did not start listening within {STARTUP_TIMEOUT:.0f}s"
    )
