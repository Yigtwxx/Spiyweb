"""Where the viewer's records come from - a live store, or a file on disk.

Two sources, one protocol, and the difference matters more than it looks. A
`StoreSource` wraps the ring buffer of a running `SpiywebIndex`: it grows as
the application answers questions, and it forgets the oldest. A `FileSource`
wraps a JSONL file somebody's application wrote, possibly on another machine,
possibly last week - it holds everything and needs no index, no model and no
numpy to be read.

The second is the one that makes the trace viewer a product rather than a
debug endpoint, so it is not an afterthought here: the file is re-read when
it changes, because the application that is writing it is usually still
running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from spiyweb.trace import TRACE_FILENAME, load_traces

if TYPE_CHECKING:
    from pathlib import Path

    from spiyweb.trace import TraceRecord, TraceStore

__all__ = ["FileSource", "StoreSource", "TraceSource"]


@runtime_checkable
class TraceSource(Protocol):
    """Everything the viewer needs from a pile of records."""

    @property
    def kind(self) -> str:
        """`"store"` for a live ring buffer, `"file"` for a JSONL file."""
        ...

    @property
    def origin(self) -> str:
        """Human-readable provenance - an index path or a file path."""
        ...

    def records(self) -> tuple[TraceRecord, ...]:
        """Everything available, oldest first."""
        ...

    def get(self, trace_id: str) -> TraceRecord | None:
        """One record by id, or `None`."""
        ...


class StoreSource:
    """The live ring buffer of an open index."""

    def __init__(self, store: TraceStore, *, origin: str = "") -> None:
        self._store = store
        self._origin = origin

    @property
    def kind(self) -> str:
        return "store"

    @property
    def origin(self) -> str:
        return self._origin

    def records(self) -> tuple[TraceRecord, ...]:
        return self._store.records()

    def get(self, trace_id: str) -> TraceRecord | None:
        return self._store.get(trace_id)


class FileSource:
    """A JSONL trace file, re-read whenever it has changed.

    Cached on `(mtime, size)` rather than read every time: a viewer polling a
    ten-thousand-line file would otherwise re-parse it once a second. Cached
    on both, and not mtime alone, because an append inside the same
    filesystem timestamp tick is exactly what a busy writer produces.
    """

    def __init__(self, path: Path | str) -> None:
        from pathlib import Path as _Path

        target = _Path(path)
        self._path = target / TRACE_FILENAME if target.is_dir() else target
        self._stamp: tuple[float, int] | None = None
        self._cache: tuple[TraceRecord, ...] = ()

    @property
    def kind(self) -> str:
        return "file"

    @property
    def origin(self) -> str:
        return str(self._path)

    @property
    def path(self) -> Path:
        return self._path

    def records(self) -> tuple[TraceRecord, ...]:
        if not self._path.exists():
            self._stamp, self._cache = None, ()
            return ()
        stat = self._path.stat()
        stamp = (stat.st_mtime, stat.st_size)
        if stamp != self._stamp:
            self._cache = load_traces(self._path)
            self._stamp = stamp
        return self._cache

    def get(self, trace_id: str) -> TraceRecord | None:
        for record in reversed(self.records()):
            if record.trace_id == trace_id:
                return record
        return None
