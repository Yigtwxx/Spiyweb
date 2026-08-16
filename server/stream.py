"""Server-sent events, paced by the server.

Each source ticks at the frequency it deserves: the progress counter is a
line count over an incrementally growing file and costs nothing, while
`nvidia-smi` spawns a process and must not be asked every two seconds for six
hours. Putting the pacing here rather than in the browser is the whole point
— a client loop would set the rate for a machine it knows nothing about.

When no run is active the stream sends one status and then only heartbeats;
it does not even look at the GPU.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from server.runner import SUPERVISOR
from server.settings import SETTINGS
from server.system import read_gpu, read_ollama


def _event(name: str, payload: object) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, default=str)}\n\n"


def event_stream() -> Iterator[str]:
    """Yield SSE frames until the client goes away."""
    last_gpu = 0.0
    last_ollama = 0.0
    last_beat = time.monotonic()
    last_out = 0
    last_err = 0

    status = SUPERVISOR.status()
    yield _event("status", status.model_dump())
    previous_state = status.state

    while True:
        now = time.monotonic()
        status = SUPERVISOR.status()
        if status.state != previous_state:
            yield _event("status", status.model_dump())
            if previous_state in ("running", "stopping") and status.state in (
                "finished",
                "failed",
            ):
                yield _event(
                    "done",
                    {
                        "exit_code": status.exit_code,
                        "finished_at": status.finished_at,
                    },
                )
            previous_state = status.state

        active = status.state in ("running", "stopping")
        if active:
            if status.progress is not None:
                yield _event(
                    "progress",
                    {
                        **status.progress.model_dump(),
                        "duty": status.duty.model_dump() if status.duty else None,
                    },
                )
            out, err = SUPERVISOR.tail(SETTINGS.log_tail_lines)
            if len(out) > last_out or len(err) > last_err:
                yield _event(
                    "log",
                    {"out": out[-40:], "err": err[-20:]},
                )
                last_out, last_err = len(out), len(err)

            if now - last_gpu >= SETTINGS.gpu_interval:
                yield _event("gpu", read_gpu().__dict__)
                last_gpu = now
            if now - last_ollama >= SETTINGS.ollama_interval:
                ollama = read_ollama()
                yield _event(
                    "ollama",
                    {
                        "reachable": ollama.reachable,
                        "models": list(ollama.models),
                        "error": ollama.error,
                    },
                )
                last_ollama = now

        if now - last_beat >= SETTINGS.heartbeat_interval:
            yield ": ping\n\n"
            last_beat = now

        time.sleep(SETTINGS.progress_interval if active else 1.0)
