import json
import sys
import time
from pathlib import Path

from infer_python import should_continue
from vla_execution import VlaProcessManager, build_infer_command


def selectable_model(local_path: Path, cameras: list[str]) -> dict:
    return {
        "id": local_path.name,
        "policy_type": "smolvla",
        "downloaded": True,
        "selectable": True,
        "local_path": str(local_path),
        "camera_features": cameras,
        "tasks": ["pick up the bow"],
    }


def wait_until(predicate, timeout: float = 3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


def test_build_infer_command_uses_registry_path_exact_task_and_required_camera(tmp_path):
    model_path = tmp_path / "models" / "model_a"
    command = build_infer_command(
        python="/env/bin/python",
        script=Path("infer_python.py"),
        model=selectable_model(model_path, ["observation.images.top"]),
        task="pick up the bow",
        robot={"port": "/dev/follower", "id": "arm"},
        cameras={
            "top": {"index": 1, "w": 320, "h": 240},
            "wrist": {"index": 0, "w": 320, "h": 240},
        },
        max_steps=100,
    )

    assert "--model-path=" + str(model_path) in command
    assert "--task=pick up the bow" in command
    assert f"--cameras={json.dumps({'top': 1}, separators=(',', ':'))}" in command
    assert "--max-steps=100" in command
    assert not any("wrist" in item for item in command)


def test_build_infer_command_rejects_missing_required_camera(tmp_path):
    try:
        build_infer_command(
            python=sys.executable,
            script=Path("infer_python.py"),
            model=selectable_model(tmp_path / "model_a", ["observation.images.wrist"]),
            task="pick up the bow",
            robot={"port": "/dev/follower", "id": "arm"},
            cameras={"top": {"index": 1}},
            max_steps=100,
        )
    except ValueError as exc:
        assert str(exc) == "Missing configured cameras: wrist"
    else:
        raise AssertionError("missing camera must fail")


def test_process_manager_tracks_real_subprocess_completion():
    manager = VlaProcessManager(max_log_lines=20)

    result = manager.start(
        [sys.executable, "-c", "print('ready')"], "model_a", "pick up the bow"
    )
    wait_until(lambda: not manager.status()["running"])
    status = manager.status()

    assert result["ok"] is True
    assert status["state"] == "completed"
    assert status["exit_code"] == 0
    assert status["lines"] == ["ready"]
    assert status["model_id"] == "model_a"


def test_process_manager_reports_failed_exit():
    manager = VlaProcessManager(max_log_lines=20)
    manager.start(
        [sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"],
        "model_a",
        "pick up the bow",
    )

    wait_until(lambda: not manager.status()["running"])

    assert manager.status()["state"] == "failed"
    assert manager.status()["exit_code"] == 3


def test_process_manager_stops_process_group():
    manager = VlaProcessManager(max_log_lines=20)
    manager.start(
        [sys.executable, "-c", "import time; print('running', flush=True); time.sleep(30)"],
        "model_a",
        "pick up the bow",
    )
    wait_until(lambda: manager.status()["state"] == "running")

    result = manager.stop(timeout=1.0)

    assert result["ok"] is True
    assert manager.status()["state"] == "stopped"
    assert manager.status()["running"] is False


def test_should_continue_stops_at_bounded_step_count():
    assert should_continue(True, 99, 100) is True
    assert should_continue(True, 100, 100) is False
    assert should_continue(True, 100, 0) is True
    assert should_continue(False, 0, 0) is False
