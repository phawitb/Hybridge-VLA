import json
import subprocess
import sys
import time
from pathlib import Path

from infer_python import (
    cycle_boundary_reached,
    normalize_cycle_command,
    normalize_worker_command,
    reset_policy_task_state,
    should_continue,
)
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


def fake_persistent_worker_command(tmp_path: Path, model_id: str = "model_a") -> list[str]:
    return [
        sys.executable, "-u", "infer_python.py", "--worker-protocol-test",
        f"--model-path={tmp_path}", f"--model-id={model_id}",
    ]


def test_worker_command_parser_accepts_only_declared_json_commands():
    assert normalize_worker_command('{"command":"release_hardware"}') == {
        "command": "release_hardware"
    }
    assert normalize_worker_command('{"command":"unknown"}') is None
    assert normalize_worker_command("continue") is None


def test_new_task_clears_actions_and_language_cache():
    class Policy:
        def __init__(self):
            self._queues = {"action": [1, 2, 3]}

    policy = Policy()
    cache = {"tokens": object(), "mask": object(), "embeddings": object()}
    reset_policy_task_state(policy, cache)

    assert policy._queues["action"] == []
    assert cache == {"tokens": None, "mask": None, "embeddings": None}


def test_worker_emits_model_ready_before_hardware_ready(tmp_path):
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "infer_python.py",
            "--worker-protocol-test",
            f"--model-path={tmp_path}",
            "--model-id=model_a",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        first = process.stdout.readline()
        process.stdin.write(json.dumps({
            "command": "run_task",
            "task": "pick",
            "max_steps": 1,
            "verification_image": str(tmp_path / "latest.jpg"),
        }) + "\n")
        process.stdin.flush()
        second = process.stdout.readline()

        assert first.startswith("MODEL_READY ")
        assert second.startswith("HARDWARE_READY ")
    finally:
        process.stdin.write('{"command":"shutdown"}\n')
        process.stdin.flush()
        process.wait(timeout=2)


def test_worker_releases_hardware_and_runs_new_task_without_restarting(tmp_path):
    process = subprocess.Popen(
        [
            sys.executable, "-u", "infer_python.py", "--worker-protocol-test",
            f"--model-path={tmp_path}", "--model-id=model_a",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    pid = process.pid

    def send(payload):
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    try:
        assert process.stdout.readline().startswith("MODEL_READY ")
        send({"command": "run_task", "task": "pick", "max_steps": 1})
        assert process.stdout.readline().startswith("HARDWARE_READY ")
        assert process.stdout.readline().startswith("CYCLE_READY ")
        send({"command": "release_hardware"})
        assert process.stdout.readline().startswith("HARDWARE_RELEASED ")
        send({"command": "run_task", "task": "place", "max_steps": 1})
        second_ready = process.stdout.readline()

        assert second_ready.startswith("HARDWARE_READY ")
        assert json.loads(second_ready.split(" ", 1)[1])["task"] == "place"
        assert process.pid == pid
        assert process.poll() is None
    finally:
        send({"command": "shutdown"})
        process.wait(timeout=2)


def test_ensure_model_reuses_ready_process_for_same_model(tmp_path):
    manager = VlaProcessManager(max_log_lines=20)
    command = fake_persistent_worker_command(tmp_path)
    try:
        first = manager.ensure_model(command, "model_a", timeout=3)
        second = manager.ensure_model(command, "model_a", timeout=3)

        assert first["ok"] is True
        assert second["ok"] is True
        assert first["pid"] == second["pid"]
        assert second["reused"] is True
        assert manager.status()["model_ready"] is True
    finally:
        manager.stop()


def test_manager_releases_hardware_and_reuses_worker_for_new_task(tmp_path):
    manager = VlaProcessManager(max_log_lines=20)
    try:
        ready = manager.ensure_model(fake_persistent_worker_command(tmp_path), "model_a", timeout=3)
        started = manager.run_task("pick", 1, str(tmp_path / "latest.jpg"), timeout=3)
        wait_until(lambda: manager.status()["state"] == "waiting_for_verification")
        released = manager.release_hardware(timeout=3)
        first_pid = manager.status()["pid"]
        second = manager.run_task("place", 1, str(tmp_path / "latest.jpg"), timeout=3)

        assert ready["ok"] is True
        assert started["ok"] is True
        assert released["ok"] is True
        assert second["ok"] is True
        assert manager.status()["pid"] == first_pid
        assert manager.status()["active_task"] == "place"
    finally:
        manager.stop()


def test_ensure_model_replaces_worker_when_model_changes(tmp_path):
    manager = VlaProcessManager(max_log_lines=20)
    try:
        first = manager.ensure_model(fake_persistent_worker_command(tmp_path, "model_a"), "model_a", timeout=3)
        second = manager.ensure_model(fake_persistent_worker_command(tmp_path, "model_b"), "model_b", timeout=3)

        assert first["ok"] is True
        assert second["ok"] is True
        assert second["reused"] is False
        assert second["pid"] != first["pid"]
        assert manager.status()["model_id"] == "model_b"
    finally:
        manager.stop()


def test_ensure_model_times_out_when_worker_never_becomes_ready():
    manager = VlaProcessManager(max_log_lines=20)
    command = [sys.executable, "-u", "-c", "import time; time.sleep(2)", "--persistent-worker"]
    try:
        result = manager.ensure_model(command, "model_a", timeout=0.05)

        assert result["ok"] is False
        assert result["code"] == "MODEL_LOAD_TIMEOUT"
    finally:
        manager.stop(timeout=0.1)


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


def test_inference_runtime_loads_opencv_before_torch_without_openmp_abort():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import infer_python; "
            "cv2, torch = infer_python.prepare_runtime(); "
            "print(cv2.__name__, torch.__name__)",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cv2 torch"


def test_cycle_boundary_and_control_commands():
    assert cycle_boundary_reached(99, 0, 100) is False
    assert cycle_boundary_reached(100, 0, 100) is True
    assert cycle_boundary_reached(200, 100, 100) is True
    assert normalize_cycle_command(" continue\n") == "continue"
    assert normalize_cycle_command("STOP") == "stop"
    assert normalize_cycle_command("unknown") == "stop"


def test_process_manager_resumes_same_process_after_cycle_boundary(tmp_path):
    manager = VlaProcessManager(max_log_lines=20)
    image = tmp_path / "latest.jpg"
    script = (
        "import json,sys; "
        f"print('CYCLE_READY '+json.dumps({{'step':100,'image':{str(image)!r}}}),flush=True); "
        "command=sys.stdin.readline().strip(); "
        "print('command='+command,flush=True); "
        f"print('CYCLE_READY '+json.dumps({{'step':200,'image':{str(image)!r}}}),flush=True); "
        "sys.stdin.readline()"
    )

    started = manager.start([sys.executable, "-u", "-c", script], "model_a", "pick")
    wait_until(lambda: manager.status()["state"] == "waiting_for_verification")
    first = manager.status()
    resumed = manager.continue_cycle()
    wait_until(lambda: manager.status()["cycle_ready_count"] == 2)
    second = manager.status()

    assert resumed["ok"] is True
    assert first["pid"] == second["pid"] == started["pid"]
    assert second["state"] == "waiting_for_verification"
    assert second["verification_image"] == str(image)
    assert "command=continue" in second["lines"]
    manager.stop()
