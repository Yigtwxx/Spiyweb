"""What the machine is doing: GPU headroom and whether Ollama is up.

The %88 VRAM rule is not decoration here. On this laptop e5-large and an
Ollama model do not both fit: leaving the embedder on the GPU during an
evaluate stage pushed it to 7843/8188 MiB, and the fix was `--device cpu`.
Showing the number, with the budget line drawn on it, is how that mistake
stays visible instead of being rediscovered.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass

from server.settings import SETTINGS

_QUERY = "name,memory.used,memory.total,utilization.gpu,temperature.gpu"


@dataclass(frozen=True)
class GpuStatus:
    present: bool
    name: str | None
    vram_used_mb: int
    vram_total_mb: int
    vram_share: float
    utilization: int
    temperature: int | None
    over_budget: bool
    budget_share: float
    error: str | None = None


@dataclass(frozen=True)
class OllamaStatus:
    reachable: bool
    models: tuple[str, ...]
    error: str | None


def read_gpu() -> GpuStatus:
    """One `nvidia-smi` call. Absent tooling is a state, not an error."""
    empty = GpuStatus(
        present=False,
        name=None,
        vram_used_mb=0,
        vram_total_mb=0,
        vram_share=0.0,
        utilization=0,
        temperature=None,
        over_budget=False,
        budget_share=SETTINGS.vram_budget_share,
    )
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return GpuStatus(**{**empty.__dict__, "error": str(error)})
    line = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not line:
        return GpuStatus(
            **{**empty.__dict__, "error": completed.stderr.strip() or None}
        )
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) < 5:
        return GpuStatus(**{**empty.__dict__, "error": "unexpected nvidia-smi output"})
    used, total = int(float(parts[1])), int(float(parts[2]))
    share = used / total if total else 0.0
    return GpuStatus(
        present=True,
        name=parts[0],
        vram_used_mb=used,
        vram_total_mb=total,
        vram_share=share,
        utilization=int(float(parts[3])),
        temperature=int(float(parts[4])),
        over_budget=share > SETTINGS.vram_budget_share,
        budget_share=SETTINGS.vram_budget_share,
    )


def read_ollama() -> OllamaStatus:
    """Is the local model server up, and which models does it hold?"""
    url = f"{SETTINGS.ollama_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as error:
        return OllamaStatus(reachable=False, models=(), error=str(error))
    models = tuple(
        str(entry.get("name", ""))
        for entry in payload.get("models", [])
        if entry.get("name")
    )
    return OllamaStatus(reachable=True, models=models, error=None)
