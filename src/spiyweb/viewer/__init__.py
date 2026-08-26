"""The browser face that ships inside the wheel (D38, Faz 2.5).

`server/` in the repository is the measurement rig: it scans `data/`,
supervises benchmark runs and polls the GPU. This package is the half that a
person who ran `pip install spiyweb[web]` should get - the recorded calls of
their own application, drawn without loading a second copy of their index.

Nothing here is imported by `import spiyweb`. FastAPI and uvicorn arrive with
the `web` extra, so every name below resolves lazily and a missing extra says
so in a sentence rather than a traceback:

    index = spiyweb.open_index("my-index")
    print(index.inspect_url())        # -> http://127.0.0.1:PORT/?token=...

    from spiyweb.viewer import serve_file
    with serve_file("traces/traces.jsonl") as viewer:
        print(viewer.url)             # a file, no index, no model

The record-side pieces - `TraceSource`, `StoreSource`, `FileSource` and the
bundle locator - cost nothing and import eagerly; only the serving half is
deferred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spiyweb.viewer.bundle import BUNDLE_ENV, MissingBundle, bundle_path, find_bundle
from spiyweb.viewer.security import LOOPBACK, TokenGuard, new_token
from spiyweb.viewer.sources import FileSource, StoreSource, TraceSource

if TYPE_CHECKING:
    from spiyweb.viewer.app import build_app
    from spiyweb.viewer.serving import ViewerHandle, serve, serve_file, serve_index

__all__ = [
    "BUNDLE_ENV",
    "LOOPBACK",
    "FileSource",
    "MissingBundle",
    "StoreSource",
    "TokenGuard",
    "TraceSource",
    "ViewerHandle",
    "build_app",
    "bundle_path",
    "find_bundle",
    "new_token",
    "serve",
    "serve_file",
    "serve_index",
]

_DEFERRED = {
    "build_app": "spiyweb.viewer.app",
    "ViewerHandle": "spiyweb.viewer.serving",
    "serve": "spiyweb.viewer.serving",
    "serve_file": "spiyweb.viewer.serving",
    "serve_index": "spiyweb.viewer.serving",
}


def __getattr__(name: str) -> object:
    """Resolve the FastAPI-bound names on demand, with a readable failure."""
    module_name = _DEFERRED.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as missing:  # pragma: no cover - depends on the env
        raise ImportError(
            f'{name} needs the browser face\'s dependencies: pip install "spiyweb[web]"'
        ) from missing
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
