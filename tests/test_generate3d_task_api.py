import threading
import time
import asyncio

from fastapi.testclient import TestClient

import main
from generate3d_task import Generate3DTaskManager


OBJECTS = [
    {
        "name": "white star",
        "center_pixel": [392.5, 169.0],
        "estimated_size_cm": [3.0, 3.0, 1.0],
    },
    {
        "name": "teal bowl",
        "center_pixel": [509.0, 91.0],
        "estimated_size_cm": [10.0, 10.0, 5.0],
    },
]


def wait_until(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def setup_task_api(monkeypatch):
    manager = Generate3DTaskManager()
    monkeypatch.setattr(main, "g3d_task_manager", manager)
    monkeypatch.setitem(main.robot_state, "connected", True)
    calibration = {
        "model": {"type": "affine"},
        "last_image_size": [800, 600],
        "points": [
            {"pixel": [0, 0]},
            {"pixel": [799, 0]},
            {"pixel": [799, 599]},
            {"pixel": [0, 599]},
        ],
    }
    monkeypatch.setattr(main, "g3d_calib_state", calibration)
    monkeypatch.setattr(main, "g3d_detection_state", {
        "id": "det-1",
        "objects": OBJECTS,
        "image_size": [800, 600],
        "calibration_revision": main._g3d_calibration_revision(),
    })
    monkeypatch.setattr(main, "robot_operation_owner", None)
    return manager


def test_generate3d_task_start_resolves_objects_and_exposes_status(monkeypatch):
    manager = setup_task_api(monkeypatch)
    executions = []

    def fake_execute(source, target, image_size, target_height_cm, safety_height_cm, stop_event, publish, calibration=None):
        executions.append((source["name"], target["name"], image_size, target_height_cm, safety_height_cm, calibration))
        publish("placing", {"shoulder_pan": 8.5, "gripper": 0.0})

    monkeypatch.setattr(main, "_g3d_execute_pick_place", fake_execute)
    client = TestClient(main.app)

    response = client.post("/api/generate3d/task/start", json={
        "instruction": "pick up white star to teal bowl",
        "detection_id": "det-1",
        "target_height_cm": 1,
        "safety_height_cm": 10,
    })
    wait_until(lambda: not manager.status()["running"])
    status = client.get("/api/generate3d/task/status").json()

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert executions[0][:5] == ("white star", "teal bowl", [800, 600], 1.0, 10.0)
    assert executions[0][5] == main.g3d_calib_state
    assert executions[0][5] is not main.g3d_calib_state
    assert status["state"] == "completed"
    assert status["joints"] == {"shoulder_pan": 8.5, "gripper": 0.0}


def test_generate3d_task_start_rejects_instruction_without_two_object_names(monkeypatch):
    setup_task_api(monkeypatch)
    client = TestClient(main.app)

    response = client.post("/api/generate3d/task/start", json={
        "instruction": "pick up the star",
        "detection_id": "det-1",
    })

    assert response.status_code == 400
    assert response.json()["code"] == "OBJECT_MATCH_REQUIRED"


def test_generate3d_task_start_rejects_nonfinite_heights(monkeypatch):
    setup_task_api(monkeypatch)
    client = TestClient(main.app)

    response = client.post("/api/generate3d/task/start", json={
        "instruction": "pick up white star to teal bowl",
        "detection_id": "det-1",
        "target_height_cm": "nan",
    })

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_HEIGHT"


def test_generate3d_task_start_rejects_stale_detection(monkeypatch):
    setup_task_api(monkeypatch)
    client = TestClient(main.app)

    response = client.post("/api/generate3d/task/start", json={
        "instruction": "pick up white star to teal bowl",
        "detection_id": "older-detection",
    })

    assert response.status_code == 409
    assert response.json()["code"] == "STALE_DETECTION"


def test_generate3d_task_start_rejects_object_outside_calibrated_workspace(monkeypatch):
    setup_task_api(monkeypatch)
    main.g3d_calib_state["points"] = [
        {"pixel": [0, 0]},
        {"pixel": [300, 0]},
        {"pixel": [0, 300]},
    ]
    main.g3d_detection_state["calibration_revision"] = main._g3d_calibration_revision()
    client = TestClient(main.app)

    response = client.post("/api/generate3d/task/start", json={
        "instruction": "pick up white star to teal bowl",
        "detection_id": "det-1",
    })

    assert response.status_code == 400
    assert response.json()["code"] == "OUTSIDE_CALIBRATED_WORKSPACE"


def test_calibrated_workspace_hull_normalizes_different_image_sizes(monkeypatch):
    setup_task_api(monkeypatch)

    assert main._g3d_point_in_calibrated_workspace([800, 300], [1600, 1200]) is True
    assert main._g3d_point_in_calibrated_workspace([1700, 300], [1600, 1200]) is False


def test_generate3d_calibration_mutation_invalidates_detection(monkeypatch):
    setup_task_api(monkeypatch)
    monkeypatch.setattr(main, "save_g3d_calibration", lambda: None)
    client = TestClient(main.app)

    response = client.post("/api/generate3d/calibration/reset")

    assert response.status_code == 200
    assert main.g3d_detection_state["id"] is None
    assert main.g3d_detection_state["objects"] == []


def test_robot_lease_rejects_other_command_writers(monkeypatch):
    class FakeRobot:
        def get_observation(self):
            return {f"{name}.pos": 0.0 for name in main.ROBOT_JOINTS}

        def send_action(self, action):
            raise AssertionError("busy command must not reach hardware")

    monkeypatch.setitem(main.robot_state, "robot", FakeRobot())
    monkeypatch.setattr(main, "robot_operation_owner", "generate3d")

    try:
        main.robot_send_positions({"shoulder_pan": 5.0})
    except main.RobotBusyError:
        pass
    else:
        raise AssertionError("another command writer must be rejected while Generate 3D owns the robot")


def test_existing_direct_run_returns_busy_while_generate3d_owns_robot(monkeypatch):
    monkeypatch.setitem(main.robot_state, "connected", True)
    monkeypatch.setattr(main, "robot_operation_owner", "generate3d")

    response = TestClient(main.app).post("/api/run/step", json={
        "method_id": "act_gripper_grasp_v1",
    })

    assert response.status_code == 409
    assert response.json()["code"] == "HARDWARE_BUSY"


def test_calibration_move_returns_busy_while_generate3d_owns_robot(monkeypatch):
    monkeypatch.setitem(main.robot_state, "connected", True)
    monkeypatch.setitem(main.calib_state, "homography", True)
    monkeypatch.setattr(main, "robot_operation_owner", "generate3d")
    monkeypatch.setattr(main, "interpolate_joints_from_pixel", lambda *args, **kwargs: {
        name: 0.0 for name in main.ROBOT_JOINTS
    })

    response = TestClient(main.app).post("/api/calibrate/move-to", json={
        "pixel": [100, 100],
    })

    assert response.status_code == 409
    assert response.json()["code"] == "HARDWARE_BUSY"


def test_stale_eval_reader_cannot_release_newer_eval_run(monkeypatch):
    old_process = object()
    new_process = object()
    released = []
    monkeypatch.setattr(main, "eval_state", {
        "running": True,
        "process": new_process,
        "reader_thread": object(),
        "run_token": "new-token",
        "log_lines": [],
        "session_id": None,
    })
    monkeypatch.setattr(main, "release_robot_operation", released.append)

    main._eval_finish_run(old_process, "old-token")

    assert main.eval_state["running"] is True
    assert main.eval_state["process"] is new_process
    assert released == []


def test_eval_start_cleans_up_process_when_reader_cannot_start(monkeypatch):
    class FakeProcess:
        pid = 4321
        stdout = None

        def __init__(self):
            self.running = True
            self.terminated = False

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.terminated = True
            self.running = False

        def wait(self, timeout=None):
            self.running = False
            return 0

    class FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    process = FakeProcess()
    monkeypatch.setattr(main.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(main.threading, "Thread", FailingThread)
    monkeypatch.setattr(main.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(main.os, "killpg", lambda pid, sig: process.terminate())
    monkeypatch.setattr(main, "_eval_load_reset_position", lambda: None)
    monkeypatch.setattr(main, "load_config", lambda: {"robot": {"port": "x", "id": "r", "cameras": {}}})
    monkeypatch.setitem(main.teleop_state, "running", False)
    monkeypatch.setitem(main.robot_state, "connected", False)
    monkeypatch.setattr(main, "eval_state", {
        "running": False,
        "process": None,
        "reader_thread": None,
        "run_token": None,
        "log_lines": [],
        "session_id": None,
    })

    result = asyncio.run(main._eval_start_owned({"model_name": "model"}))

    assert result["ok"] is False
    assert process.terminated is True
    assert main.eval_state["process"] is None


def test_pick_place_motion_publishes_measured_joints_for_each_phase(monkeypatch):
    measured = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 50.0,
    }
    sent = []
    published = []

    def fake_predict(pixel, image_size=None, height_cm=0.0, calibration=None):
        return {
            "position_3d": [pixel[0] / 1000, height_cm / 100, pixel[1] / 1000],
            "joints": {
                "shoulder_pan": pixel[0] / 10,
                "shoulder_lift": -height_cm,
                "elbow_flex": 20.0,
                "wrist_flex": 30.0,
                "wrist_roll": 0.0,
                "gripper": 50.0,
            },
        }

    def fake_send(joints, owner=None):
        assert owner == "generate3d"
        measured.update(joints)
        sent.append(dict(joints))

    monkeypatch.setattr(main, "_g3d_predict_from_pixel", fake_predict)
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", fake_send)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)

    main._g3d_execute_pick_place(
        OBJECTS[0], OBJECTS[1], [800, 600], 1.0, 10.0,
        threading.Event(),
        lambda phase, joints: published.append((phase, dict(joints))),
    )

    phases = [phase for phase, _ in published]
    phase_order = [phase for index, phase in enumerate(phases) if index == 0 or phase != phases[index - 1]]
    assert phase_order == [
        "opening_gripper",
        "raising_to_safety",
        "moving_to_source",
        "descending_to_source",
        "grasping",
        "lifting_source",
        "moving_to_target",
        "placing",
        "releasing",
        "lifting_after_release",
    ]
    assert sent[-1]["gripper"] == 100.0
    assert published[-1][1] == measured

    carrying = [joints for phase, joints in published if phase in {"lifting_source", "moving_to_target", "placing"}]
    assert carrying
    assert all(joints["gripper"] == 0.0 for joints in carrying)


def test_task_move_waits_for_measured_convergence(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    target = {**measured, "shoulder_pan": 10.0}
    sends = []

    def lagging_send(command, owner=None):
        assert owner == "generate3d"
        sends.append(dict(command))
        for name, value in command.items():
            measured[name] += (value - measured[name]) * 0.5

    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", lagging_send)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)

    completed = main._g3d_task_move(
        target, "moving_to_source", threading.Event(), lambda phase, joints: None,
        n_steps=1, tolerance_deg=0.2,
    )

    assert completed is True
    assert len(sends) > 1
    assert abs(measured["shoulder_pan"] - 10.0) <= 0.2


def test_pick_place_accepts_gripper_stopping_on_grasped_object(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured["gripper"] = 100.0
    sent = []

    def fake_predict(pixel, image_size=None, height_cm=0.0, calibration=None):
        return {
            "position_3d": [0.0, height_cm / 100, 0.0],
            "joints": {
                "shoulder_pan": pixel[0] / 10,
                "shoulder_lift": -height_cm,
                "elbow_flex": 20.0,
                "wrist_flex": 30.0,
                "wrist_roll": 0.0,
                "gripper": 100.0,
            },
        }

    def object_blocking_send(command, owner=None):
        assert owner == "generate3d"
        sent.append(dict(command))
        for name, value in command.items():
            if name == "gripper" and value == 0.0:
                measured[name] = 20.0
            else:
                measured[name] = value

    monkeypatch.setattr(main, "_g3d_predict_from_pixel", fake_predict)
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", object_blocking_send)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)

    main._g3d_execute_pick_place(
        OBJECTS[0], OBJECTS[1], [800, 600], 1.0, 10.0,
        threading.Event(), lambda phase, joints: None,
    )

    assert any(command["gripper"] == 0.0 for command in sent)
    assert sent[-1]["gripper"] == 100.0
