"""The guards that stand between a stray click and six hours of GPU time."""

from __future__ import annotations

import pytest
from server import runner
from server.schemas import RunRequest


def _request(**overrides: object) -> RunRequest:
    base = {
        "index": "musique",
        "stage": "report",
        "dataset": "musique",
        "device": "cpu",
    }
    base.update(overrides)
    return RunRequest(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("state", "phrase"),
    [
        ("running", "is using this machine"),
        ("stopping", "is using this machine"),
        # The case that mattered in practice: a run launched from a terminal
        # takes the same RAM and CPU, and the header already reports it.
        ("external", "started outside this server"),
    ],
)
def test_inspect_is_locked_while_any_run_holds_the_machine(
    state: str, phrase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from server import app as server_app

    held = runner.RunStatus(
        state=state,
        pid=None,
        argv=[],
        cwd=None,
        index="musique_prop200",
        started_at=None,
        finished_at=None,
        exit_code=None,
        duty=None,
        progress=None,
        log_out=None,
        log_err=None,
    )
    monkeypatch.setattr(runner.SUPERVISOR, "status", lambda: held)
    client = fastapi_testclient.TestClient(server_app.app)
    response = client.post(
        "/api/inspect",
        json={"index": "musique", "query": {"mode": "atom", "node": "d00000:0"}},
    )
    assert response.status_code == 423, response.text
    assert phrase in response.json()["detail"]


def test_argv_is_built_from_the_allowlist() -> None:
    argv = runner.build_argv(_request())
    assert argv[1:5] == ["-u", "-m", "spiyweb.evaluation.run", "report"]
    assert "--data-dir" in argv
    assert "--dataset" in argv


def test_unknown_stage_is_refused() -> None:
    with pytest.raises(runner.RunRefused):
        runner.build_argv(_request(stage="rm -rf"))


def test_unknown_dataset_is_refused() -> None:
    with pytest.raises(runner.RunRefused):
        runner.build_argv(_request(dataset="whatever"))


def test_unknown_device_is_refused() -> None:
    with pytest.raises(runner.RunRefused):
        runner.build_argv(_request(device="tpu"))


def test_a_path_cannot_be_smuggled_through_the_index_name() -> None:
    """The index name reaches the server from a browser; it is untrusted."""
    with pytest.raises(runner.RunRefused):
        runner.build_argv(_request(index="../../etc", stage="index"))


def test_flags_only_appear_when_asked_for() -> None:
    plain = runner.build_argv(_request(stage="index"))
    assert "--propositions" not in plain
    assert "--nli" not in plain
    rich = runner.build_argv(_request(stage="index", propositions=True, nli=True))
    assert "--propositions" in rich
    assert "--nli" in rich


def test_the_plan_token_is_the_hash_of_the_command() -> None:
    """Two different commands cannot share a token, so a changed form is caught."""
    first = runner.plan(_request(stage="report"))
    same = runner.plan(_request(stage="report"))
    other = runner.plan(_request(stage="report", sample_size=200))
    assert first.token == same.token
    assert first.token != other.token


def test_starting_with_a_stale_token_is_refused() -> None:
    request = _request(stage="report")
    with pytest.raises(runner.RunRefused, match="plan changed"):
        runner.SUPERVISOR.start(request, "not-the-token", request.index)


def test_starting_without_typing_the_word_is_refused() -> None:
    request = _request(stage="report")
    token = runner.plan(request).token
    with pytest.raises(runner.RunRefused, match="to confirm"):
        runner.SUPERVISOR.start(request, token, "yes")


def test_the_confirm_word_is_the_index_name() -> None:
    assert runner.plan(_request()).confirm_word == "musique"


def test_stopping_requires_the_literal_word() -> None:
    with pytest.raises(runner.RunRefused, match="type STOP"):
        runner.SUPERVISOR.stop("please stop")


def test_force_is_called_out_as_a_warning() -> None:
    warnings = runner.plan(_request(stage="index", force=True)).warnings
    assert any("--force" in warning for warning in warnings)


def test_a_gpu_embedder_on_evaluate_is_called_out() -> None:
    """The 96% VRAM mistake is worth naming before it is repeated."""
    warnings = runner.plan(_request(stage="evaluate", device="cuda")).warnings
    assert any("VRAM" in warning for warning in warnings)


def test_a_dead_pid_is_not_alive() -> None:
    assert runner.process_alive(0) is False
    assert runner.process_alive(-1) is False
