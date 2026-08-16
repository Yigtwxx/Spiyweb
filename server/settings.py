"""Server tunables - no magic numbers in the modules, same rule as `config.py`.

Everything here can be overridden from the environment, because the one
machine this runs on is also the machine running six-hour measurements: the
operator has to be able to slow the server down without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw and raw.strip() else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw and raw.strip() else default


@dataclass(frozen=True)
class Settings:
    """One immutable settings object, read from the environment at import.

    Attributes:
        cache_graphs: Merged graphs held at once. Each is roughly a hundred
            megabytes on the MuSiQue index, so this is a memory decision, not
            a speed one.
        cache_stores / cache_vectors / cache_layers: Same idea per artifact.
        cache_texts / cache_entities: Cheap, so a little more headroom.
        progress_interval: Seconds between progress ticks on the event
            stream. The measure is a line count over an incrementally read
            file, so this is nearly free.
        gpu_interval: Seconds between `nvidia-smi` calls. Each one spawns a
            process (~150 ms); polling it every two seconds for six hours
            would steal real time from the run being watched.
        ollama_interval: Seconds between Ollama reachability checks - the
            answer rarely changes.
        heartbeat_interval: Seconds between SSE comments, so proxies and
            browsers keep an idle connection open.
        log_tail_lines: Lines of stdout/stderr kept for the log view.
        stop_grace_seconds: How long a stopped run gets to exit on its own
            before it is killed.
        max_scene_nodes: Hard ceiling on drawable atoms; the client asks for
            fewer, the server refuses more.
    """

    host: str = os.environ.get("SPIYWEB_HOST", "127.0.0.1")
    port: int = _int("SPIYWEB_PORT", 8000)
    cache_graphs: int = _int("SPIYWEB_CACHE_GRAPHS", 2)
    cache_stores: int = _int("SPIYWEB_CACHE_STORES", 2)
    cache_vectors: int = _int("SPIYWEB_CACHE_VECTORS", 2)
    cache_layers: int = _int("SPIYWEB_CACHE_LAYERS", 2)
    cache_texts: int = _int("SPIYWEB_CACHE_TEXTS", 2)
    cache_entities: int = _int("SPIYWEB_CACHE_ENTITIES", 3)
    progress_interval: float = _float("SPIYWEB_PROGRESS_INTERVAL", 2.0)
    gpu_interval: float = _float("SPIYWEB_GPU_INTERVAL", 5.0)
    ollama_interval: float = _float("SPIYWEB_OLLAMA_INTERVAL", 15.0)
    heartbeat_interval: float = _float("SPIYWEB_HEARTBEAT_INTERVAL", 30.0)
    log_tail_lines: int = _int("SPIYWEB_LOG_TAIL", 500)
    stop_grace_seconds: float = _float("SPIYWEB_STOP_GRACE", 10.0)
    max_scene_nodes: int = _int("SPIYWEB_MAX_SCENE_NODES", 600)
    vram_budget_share: float = _float("SPIYWEB_VRAM_BUDGET", 0.88)
    ollama_url: str = os.environ.get("SPIYWEB_OLLAMA_URL", "http://localhost:11434")


SETTINGS = Settings()
