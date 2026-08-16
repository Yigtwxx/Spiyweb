"""Starting and stopping measurement runs, with the guards that make it safe.

The owner asked for a UI that can launch runs. The hazard is obvious — a
stray click could kill six hours of GPU time — so the design answers it three
times over rather than with a confirmation dialog:

1. A run cannot be started from a form. The form produces a PLAN, which shows
   the exact argv that will execute and carries a token hashed from it. The
   start call must present that token, so "what you saw" and "what runs"
   cannot diverge.
2. Both starting and stopping require the operator to TYPE a word. The failure
   mode being defended against is a misplaced click, and a second button is
   defeated by a second misplaced click; typing is not.
3. One run at a time, enforced by an atomically created lock file rather than
   an in-process flag, so a server restart cannot lose track of a live child.

A dead PID leaves a `stale` lock, which is reported and never cleaned up
automatically: silent cleanup is the most likely path to two runs at once.

Progress is read from `llm_cache*.jsonl` line counts. That file gets one line
per completed LLM call the moment it returns, which makes it both the most
reliable progress signal available and the reason a killed run resumes almost
for free.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from server._paths import DATA_ROOT, LOG_DIR, REPO_ROOT, RUN_LOCK
from server.resources import CACHE
from server.schemas import (
    DutyState,
    RunPlan,
    RunProgress,
    RunRequest,
    RunStatus,
)
from server.settings import SETTINGS

STAGES = ("download", "index", "evaluate", "report", "all")
DATASETS = ("musique", "2wiki", "hotpotqa")
WEB_MODES = ("colored", "plain")
CHAIN_MODES = ("none", "single", "sequential")
DEVICES = ("cpu", "cuda", "mps")

EXTERNAL_WINDOW_SECONDS = 90.0
"""How recently an LLM cache must have grown to count as a live outside run."""


class RunRefused(RuntimeError):
    """The request was rejected before anything was launched."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def process_alive(pid: int) -> bool:
    """Is this PID still running? Windows and POSIX, no new dependency."""
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes

    still_active = 259
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # type: ignore[attr-defined]
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
            handle, ctypes.byref(code)
        )
        return bool(ok) and code.value == still_active
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]


def build_argv(request: RunRequest) -> list[str]:
    """Turn a request into an argv from an allowlist - never free text.

    Every value is checked against the CLI's own documented choices, and the
    data directory must be one the server itself listed. A browser cannot
    smuggle an argument in here.
    """
    if request.stage not in STAGES:
        raise RunRefused(f"stage {request.stage!r} is not one of {STAGES}")
    if request.dataset not in DATASETS:
        raise RunRefused(f"dataset {request.dataset!r} is not one of {DATASETS}")
    if request.web not in WEB_MODES:
        raise RunRefused(f"web {request.web!r} is not one of {WEB_MODES}")
    if request.chain_mode is not None and request.chain_mode not in CHAIN_MODES:
        raise RunRefused(f"chain_mode {request.chain_mode!r} is not valid")
    if request.device is not None and request.device not in DEVICES:
        raise RunRefused(f"device {request.device!r} is not valid")

    known = {path.name for path in CACHE.available()}
    if request.index not in known and request.stage not in ("download", "index", "all"):
        raise RunRefused(f"index {request.index!r} does not exist yet")
    if "/" in request.index or "\\" in request.index or not request.index:
        raise RunRefused("index name must be a plain directory name")

    argv = [
        sys.executable,
        "-u",  # unbuffered: the log tail is useless otherwise
        "-m",
        "spiyweb.evaluation.run",
        request.stage,
        "--data-dir",
        str(Path("data") / request.index),
        "--dataset",
        request.dataset,
    ]
    if request.sample_size is not None:
        argv += ["--sample-size", str(request.sample_size)]
    if request.sample_seed is not None:
        argv += ["--sample-seed", str(request.sample_seed)]
    if request.force:
        argv.append("--force")
    if request.no_entity_llm:
        argv.append("--no-entity-llm")
    if request.propositions:
        argv.append("--propositions")
    if request.nli:
        argv.append("--nli")
    if request.skip_iterative:
        argv.append("--skip-iterative")
    argv += ["--web", request.web]
    if request.device:
        argv += ["--device", request.device]
    if request.llm_model:
        argv += ["--llm-model", request.llm_model]
    if request.decomp_model:
        argv += ["--decomp-model", request.decomp_model]
    if request.max_colors is not None:
        argv += ["--max-colors", str(request.max_colors)]
    if request.chain_mode:
        argv += ["--chain-mode", request.chain_mode]
    return argv


def _llm_calls(root: Path) -> int:
    """Total completed LLM calls across every model's cache file."""
    total = 0
    for path in root.glob("llm_cache*.jsonl"):
        try:
            with path.open("rb") as handle:
                total += sum(1 for _ in handle)
        except OSError:
            continue
    return total


def plan(request: RunRequest) -> RunPlan:
    """What would run, what it would skip, and what to worry about."""
    argv = build_argv(request)
    root = DATA_ROOT / request.index
    will_skip: list[str] = []
    warnings: list[str] = []

    if request.stage in ("index", "all") and not request.force:
        for artifact, stage in (
            ("vectors.npz", "embed"),
            ("entities.json", "entity extraction"),
            ("edges_entity.json", "entity edges"),
            ("edges_semantic.json", "semantic edges"),
            ("propositions.json", "proposition extraction"),
        ):
            if (root / artifact).exists():
                will_skip.append(f"{stage} ({artifact} exists)")
    if request.force:
        warnings.append(
            "--force rebuilds every index artifact; existing ones are overwritten"
        )
    if request.stage in ("evaluate", "all") and (root / "results.json").exists():
        warnings.append("results.json exists and will be overwritten")
    if request.device != "cpu" and request.stage in ("evaluate", "all"):
        warnings.append(
            "the embedder is not pinned to the CPU; on this GPU that pushed "
            "VRAM to 96% and squeezed Ollama"
        )

    estimated: int | None = None
    results_path = root / "results.json"
    if request.stage in ("evaluate", "all") and results_path.exists():
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            per_question = float(payload["combo"]["llm_calls_per_question"])
            questions = int(payload["question_count"])
            estimated = int(per_question * questions)
        except (KeyError, ValueError, TypeError, OSError):
            estimated = None

    token = hashlib.sha256(" ".join(argv).encode("utf-8")).hexdigest()[:16]
    return RunPlan(
        argv=argv,
        cwd=str(REPO_ROOT),
        will_skip=will_skip,
        estimated_llm_calls=estimated,
        warnings=warnings,
        token=token,
        confirm_word=request.index,
    )


@dataclass
class _Live:
    """State of the child the server is currently supervising."""

    request: RunRequest
    argv: list[str]
    process: subprocess.Popen[bytes]
    started: float
    calls_at_start: int
    out_path: Path
    err_path: Path
    cycle: int = 1
    phase: str = "window"
    phase_started: float = field(default_factory=time.monotonic)
    stopping: bool = False


class RunSupervisor:
    """Owns at most one child process, and the lock that proves it."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._live: _Live | None = None
        self._finished: RunStatus | None = None
        self._timer: threading.Thread | None = None

    # -- lock -----------------------------------------------------------------

    def _read_lock(self) -> dict[str, object] | None:
        if not RUN_LOCK.exists():
            return None
        try:
            payload = json.loads(RUN_LOCK.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_lock(self, payload: dict[str, object]) -> None:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            with RUN_LOCK.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1)
        except FileExistsError as error:
            raise RunRefused("a run lock already exists") from error

    def external(self) -> tuple[str, int] | None:
        """An index being written to right now by something this server did not start.

        A measurement launched from a terminal is invisible to the lock, and a
        page that says "nothing running" while the GPU is at 98% is lying
        about the machine. The evidence is the same file progress uses: an
        `llm_cache*.jsonl` that gained bytes in the last minute means some
        process is completing LLM calls into that index.
        """
        if not DATA_ROOT.exists():
            return None
        fresh = time.time() - EXTERNAL_WINDOW_SECONDS
        for root in sorted(DATA_ROOT.iterdir()):
            if not root.is_dir():
                continue
            for path in root.glob("llm_cache*.jsonl"):
                try:
                    if path.stat().st_mtime >= fresh:
                        return root.name, _llm_calls(root)
                except OSError:
                    continue
        return None

    def release_lock(self) -> None:
        """Remove a lock the operator has confirmed is stale."""
        with self._lock:
            existing = self._read_lock()
            if existing is None:
                return
            pid = int(existing.get("pid", 0))
            if process_alive(pid):
                raise RunRefused(f"pid {pid} is still alive; stop it instead")
            RUN_LOCK.unlink(missing_ok=True)

    # -- status ---------------------------------------------------------------

    def status(self) -> RunStatus:
        with self._lock:
            live = self._live
            if live is not None:
                code = live.process.poll()
                if code is None:
                    return self._running_status(live)
                self._finish(live, code)
            existing = self._read_lock()
            if existing is not None:
                pid = int(existing.get("pid", 0))
                if not process_alive(pid):
                    return RunStatus(
                        state="stale",
                        pid=pid,
                        argv=list(existing.get("argv", [])),  # type: ignore[arg-type]
                        cwd=str(existing.get("cwd") or ""),
                        index=str(existing.get("index") or ""),
                        started_at=str(existing.get("started_at") or ""),
                        finished_at=None,
                        exit_code=None,
                        duty=None,
                        progress=None,
                        log_out=str(existing.get("log_out") or ""),
                        log_err=str(existing.get("log_err") or ""),
                    )
            if self._finished is not None:
                return self._finished
            outside = self.external()
            if outside is not None:
                name, calls = outside
                return RunStatus(
                    state="external",
                    pid=None,
                    argv=[],
                    cwd=str(REPO_ROOT),
                    index=name,
                    started_at=None,
                    finished_at=None,
                    exit_code=None,
                    duty=None,
                    progress=RunProgress(
                        stage="unknown",
                        llm_calls=calls,
                        llm_calls_at_start=0,
                        llm_calls_per_min=0.0,
                        questions_total=None,
                        elapsed_s=0.0,
                        eta_s=None,
                        eta_basis=None,
                    ),
                    log_out=None,
                    log_err=None,
                )
            return RunStatus(
                state="idle",
                pid=None,
                argv=[],
                cwd=None,
                index=None,
                started_at=None,
                finished_at=None,
                exit_code=None,
                duty=None,
                progress=None,
                log_out=None,
                log_err=None,
            )

    def _progress(self, live: _Live) -> RunProgress:
        root = DATA_ROOT / live.request.index
        calls = _llm_calls(root)
        elapsed = max(1e-6, time.monotonic() - live.started)
        made = max(0, calls - live.calls_at_start)
        rate = made / (elapsed / 60.0)
        questions: int | None = None
        meta = CACHE.meta(root) if root.exists() else {}
        raw = meta.get("questions")
        if isinstance(raw, int):
            questions = raw
        return RunProgress(
            stage=live.request.stage,
            llm_calls=calls,
            llm_calls_at_start=live.calls_at_start,
            llm_calls_per_min=rate,
            questions_total=questions,
            elapsed_s=elapsed,
            eta_s=None,
            eta_basis=None,
        )

    def _duty(self, live: _Live) -> DutyState:
        duty = live.request.duty
        span = duty.window_s if live.phase == "window" else duty.cool_s
        left = max(0.0, span - (time.monotonic() - live.phase_started))
        return DutyState(
            enabled=duty.enabled,
            phase=live.phase,
            cycle=live.cycle,
            window_s=duty.window_s,
            cool_s=duty.cool_s,
            seconds_left=left,
        )

    def _running_status(self, live: _Live) -> RunStatus:
        return RunStatus(
            state="stopping" if live.stopping else "running",
            pid=live.process.pid,
            argv=list(live.argv),
            cwd=str(REPO_ROOT),
            index=live.request.index,
            started_at=datetime.fromtimestamp(
                time.time() - (time.monotonic() - live.started), UTC
            ).isoformat(timespec="seconds"),
            finished_at=None,
            exit_code=None,
            duty=self._duty(live) if live.request.duty.enabled else None,
            progress=self._progress(live),
            log_out=str(live.out_path),
            log_err=str(live.err_path),
        )

    def _finish(self, live: _Live, code: int) -> None:
        self._finished = RunStatus(
            state="failed" if code not in (0, None) else "finished",
            pid=live.process.pid,
            argv=list(live.argv),
            cwd=str(REPO_ROOT),
            index=live.request.index,
            started_at=None,
            finished_at=_now(),
            exit_code=code,
            duty=None,
            progress=self._progress(live),
            log_out=str(live.out_path),
            log_err=str(live.err_path),
        )
        self._live = None
        RUN_LOCK.unlink(missing_ok=True)

    # -- control --------------------------------------------------------------

    def start(self, request: RunRequest, token: str, typed: str) -> RunStatus:
        proposed = plan(request)
        if token != proposed.token:
            raise RunRefused(
                "the plan changed since it was shown; review the new command"
            )
        if typed.strip() != proposed.confirm_word:
            raise RunRefused(f"type {proposed.confirm_word!r} to confirm this run")
        with self._lock:
            current = self.status()
            if current.state in ("running", "stopping"):
                raise RunRefused(f"a run is already active (pid {current.pid})")
            if current.state == "stale":
                raise RunRefused(
                    "a stale lock is in the way; clear it once you have checked "
                    "that the old process is really gone"
                )
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            tag = f"{request.index}_{request.stage}_{int(time.time())}"
            out_path = LOG_DIR / f"{tag}.out"
            err_path = LOG_DIR / f"{tag}.err"
            argv = proposed.argv
            flags = 0
            if os.name == "nt":
                # Without its own process group a Ctrl+C aimed at the server
                # would take the child down with it.
                flags = subprocess.CREATE_NEW_PROCESS_GROUP
            process = subprocess.Popen(
                argv,
                cwd=REPO_ROOT,
                stdout=out_path.open("wb"),
                stderr=err_path.open("wb"),
                creationflags=flags,
            )
            self._write_lock(
                {
                    "pid": process.pid,
                    "argv": argv,
                    "cwd": str(REPO_ROOT),
                    "index": request.index,
                    "started_at": _now(),
                    "log_out": str(out_path),
                    "log_err": str(err_path),
                }
            )
            self._finished = None
            self._live = _Live(
                request=request,
                argv=argv,
                process=process,
                started=time.monotonic(),
                calls_at_start=_llm_calls(DATA_ROOT / request.index),
                out_path=out_path,
                err_path=err_path,
            )
            if request.duty.enabled:
                self._start_duty_timer()
            return self._running_status(self._live)

    def stop(self, typed: str) -> RunStatus:
        if typed.strip().upper() != "STOP":
            raise RunRefused("type STOP to confirm")
        with self._lock:
            live = self._live
            if live is None:
                raise RunRefused("no run is active")
            live.stopping = True
            live.request.duty.enabled = False
            process = live.process
        process.terminate()
        deadline = time.monotonic() + SETTINGS.stop_grace_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if process.poll() is None:
            process.kill()
        return self.status()

    def tail(self, lines: int) -> tuple[list[str], list[str]]:
        with self._lock:
            live = self._live
            paths = (
                (live.out_path, live.err_path)
                if live is not None
                else (
                    Path(self._finished.log_out or "") if self._finished else Path(),
                    Path(self._finished.log_err or "") if self._finished else Path(),
                )
            )

        def read(path: Path) -> list[str]:
            if not path or not path.exists():
                return []
            text = path.read_text(encoding="utf-8", errors="replace")
            return text.splitlines()[-lines:]

        return read(paths[0]), read(paths[1])

    # -- duty cycle -----------------------------------------------------------

    def _start_duty_timer(self) -> None:
        """Run the 40/5 rhythm here rather than in a wrapper script.

        The server already owns the PID, the lock and the stop path; giving a
        second owner to the same child is how a process ends up orphaned.
        Killing mid-run is safe because every completed LLM call is already on
        disk and every index stage skips itself when its artifact exists.
        """

        def loop() -> None:
            while True:
                time.sleep(1.0)
                with self._lock:
                    live = self._live
                    if live is None or live.stopping or not live.request.duty.enabled:
                        return
                    elapsed = time.monotonic() - live.phase_started
                    if live.phase == "window" and elapsed >= live.request.duty.window_s:
                        live.phase = "cooling"
                        live.phase_started = time.monotonic()
                        process = live.process
                    else:
                        continue
                process.terminate()
                time.sleep(2.0)
                if process.poll() is None:
                    process.kill()
                time.sleep(max(0.0, self._cool_remaining()))
                if not self._relaunch():
                    return

        self._timer = threading.Thread(target=loop, daemon=True, name="duty-cycle")
        self._timer.start()

    def _cool_remaining(self) -> float:
        with self._lock:
            live = self._live
            if live is None:
                return 0.0
            return max(
                0.0, live.request.duty.cool_s - (time.monotonic() - live.phase_started)
            )

    def _relaunch(self) -> bool:
        with self._lock:
            live = self._live
            if live is None or live.stopping:
                return False
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            live.process = subprocess.Popen(
                live.argv,
                cwd=REPO_ROOT,
                stdout=live.out_path.open("ab"),
                stderr=live.err_path.open("ab"),
                creationflags=flags,
            )
            live.cycle += 1
            live.phase = "window"
            live.phase_started = time.monotonic()
            return True


SUPERVISOR = RunSupervisor()
