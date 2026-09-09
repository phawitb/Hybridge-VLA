import threading
import time
import asyncio
import json

import pytest
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
        "execution_mode": "real",
        "motion_mode": "waypoint",
    })
    wait_until(lambda: not manager.status()["running"])
    status = client.get("/api/generate3d/task/status").json()

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert executions[0][:5] == ("white star", "teal bowl", [800, 600], 1.0, 10.0)
    assert executions[0][5] == main.g3d_calib_state
    assert executions[0][5] is not main.g3d_calib_state
    assert status["state"] == "completed"
    assert status["execution_mode"] == "real"
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


def test_generate3d_task_start_rejects_invalid_motion_mode_before_hardware(monkeypatch):
    setup_task_api(monkeypatch)
    monkeypatch.setattr(main, "acquire_robot_operation", lambda owner: (_ for _ in ()).throw(AssertionError("hardware acquired")))
    client = TestClient(main.app)

    response = client.post("/api/generate3d/task/start", json={
        "instruction": "pick up white star to teal bowl",
        "detection_id": "det-1",
        "execution_mode": "real",
        "motion_mode": "diagonal",
    })

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_MOTION_MODE"


def test_generate3d_task_defaults_to_smooth_motion(monkeypatch):
    manager = setup_task_api(monkeypatch)
    captured = []
    monkeypatch.setattr(main, "_g3d_build_smooth_pick_place_plan", lambda *args, **kwargs: captured.append(args) or [])
    client = TestClient(main.app)

    response = client.post("/api/generate3d/task/start", json={
        "instruction": "pick up white star to teal bowl",
        "detection_id": "det-1",
    })
    wait_until(lambda: not manager.status()["running"])

    assert response.status_code == 200
    assert len(captured) == 1


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


def test_generate3d_task_start_can_disable_workspace_enforcement(monkeypatch):
    manager = setup_task_api(monkeypatch)
    main.g3d_calib_state["points"] = [
        {"pixel": [0, 0]},
        {"pixel": [300, 0]},
        {"pixel": [0, 300]},
    ]
    main.g3d_detection_state["calibration_revision"] = main._g3d_calibration_revision()
    monkeypatch.setattr(main, "_g3d_execute_pick_place", lambda *args, **kwargs: None)
    client = TestClient(main.app)

    response = client.post("/api/generate3d/task/start", json={
        "instruction": "pick up white star to teal bowl",
        "detection_id": "det-1",
        "enforce_workspace": False,
        "execution_mode": "real",
        "motion_mode": "waypoint",
    })
    wait_until(lambda: not manager.status()["running"])

    assert response.status_code == 200
    assert manager.status()["state"] == "completed"


def test_generate3d_task_defaults_to_simulation_without_robot_hardware(monkeypatch):
    manager = setup_task_api(monkeypatch)
    monkeypatch.setitem(main.robot_state, "connected", False)
    monkeypatch.setattr(main, "robot_get_positions", lambda: (_ for _ in ()).throw(AssertionError("simulation read hardware")))
    monkeypatch.setattr(main, "robot_send_positions", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("simulation wrote hardware")))
    monkeypatch.setattr(main, "acquire_robot_operation", lambda owner: (_ for _ in ()).throw(AssertionError("simulation acquired hardware")))
    monkeypatch.setattr(main, "_g3d_predict_from_pixel", lambda pixel, image_size=None, height_cm=0, calibration=None: {
        "position_3d": [pixel[0] / 1000, height_cm / 100, pixel[1] / 1000],
        "joints": {
            "shoulder_pan": 0.0,
            "shoulder_lift": -height_cm,
            "elbow_flex": 20.0,
            "wrist_flex": 30.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
    })
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)
    client = TestClient(main.app)

    response = client.post("/api/generate3d/task/start", json={
        "instruction": "pick up white star to teal bowl",
        "detection_id": "det-1",
    })
    wait_until(lambda: not manager.status()["running"])

    assert response.status_code == 200
    assert manager.status()["state"] == "completed"
    assert manager.status()["execution_mode"] == "simulation"
    assert manager.status()["joints"] is not None


def test_generate3d_simulation_uses_requested_current_pose_without_hardware(monkeypatch):
    manager = setup_task_api(monkeypatch)
    requested = {
        "shoulder_pan": 4.0,
        "shoulder_lift": -12.0,
        "elbow_flex": 18.0,
        "wrist_flex": 25.0,
        "wrist_roll": 3.0,
        "gripper": 35.0,
    }
    captured = []
    monkeypatch.setattr(main, "robot_get_positions", lambda: (_ for _ in ()).throw(AssertionError("simulation read hardware")))
    monkeypatch.setattr(main, "robot_send_positions", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("simulation wrote hardware")))
    monkeypatch.setattr(main, "_g3d_build_pick_place_plan", lambda *args, **kwargs: captured.append(dict(args[5])) or [])
    client = TestClient(main.app)

    response = client.post("/api/generate3d/task/start", json={
        "instruction": "pick up white star to teal bowl",
        "detection_id": "det-1",
        "execution_mode": "simulation",
        "motion_mode": "waypoint",
        "initial_joints": requested,
    })
    wait_until(lambda: not manager.status()["running"])

    assert response.status_code == 200
    assert captured == [requested]
    assert manager.status()["joints"] == requested


def test_generate3d_simulation_publishes_initial_pose_before_motion(monkeypatch):
    initial = {name: float(index * 5) for index, name in enumerate(main.ROBOT_JOINTS)}
    target = {**initial, "shoulder_pan": 20.0}
    published = []
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)

    main._g3d_execute_simulation(
        [{"phase": "moving_to_source", "joints": target, "n_steps": 2}],
        initial,
        threading.Event(),
        lambda phase, joints: published.append((phase, dict(joints))),
    )

    assert published[0] == ("starting", initial)
    assert published[1][0] == "moving_to_source"


def test_real_plan_streams_intermediate_trajectory_then_converges_final_anchor(monkeypatch):
    initial = {name: 0.0 for name in main.ROBOT_JOINTS}
    trajectory = [
        {**initial, "shoulder_pan": 2.0},
        {**initial, "shoulder_pan": 5.0},
        {**initial, "shoulder_pan": 9.0},
    ]
    sent = []
    converged = []
    monkeypatch.setattr(main, "robot_send_positions", lambda joints, owner=None: sent.append((dict(joints), owner)))
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(initial))
    monkeypatch.setattr(main, "_g3d_task_move", lambda *args, **kwargs: converged.append((args, kwargs)) or True)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)

    main._g3d_execute_real_plan(
        [{
            "phase": "moving_to_target",
            "joints": trajectory[-1],
            "trajectory": trajectory,
            "convergence_names": main.ROBOT_JOINTS[:5],
            "tolerance_deg": 3.0,
        }],
        threading.Event(),
        lambda phase, joints: None,
    )

    assert sent == [(trajectory[0], "generate3d"), (trajectory[1], "generate3d")]
    assert converged[0][0][0] == trajectory[-1]
    assert converged[0][0][1] == "moving_to_target"


def test_real_plan_stops_during_trajectory_stream(monkeypatch):
    initial = {name: 0.0 for name in main.ROBOT_JOINTS}
    trajectory = [{**initial, "shoulder_pan": value} for value in (2.0, 5.0, 9.0)]
    stop_event = threading.Event()
    sent = []
    converged = []
    monkeypatch.setattr(main, "robot_send_positions", lambda joints, owner=None: sent.append(dict(joints)))
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(initial))
    monkeypatch.setattr(main, "_g3d_task_move", lambda *args, **kwargs: converged.append(args) or True)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)

    main._g3d_execute_real_plan(
        [{"phase": "lifting_source", "joints": trajectory[-1], "trajectory": trajectory}],
        stop_event,
        lambda phase, joints: stop_event.set(),
    )

    assert sent == [trajectory[0]]
    assert converged == []


def test_simulation_publishes_planned_trajectory_exactly(monkeypatch):
    initial = {name: 0.0 for name in main.ROBOT_JOINTS}
    trajectory = [{**initial, "shoulder_pan": value} for value in (2.0, 5.0, 9.0)]
    published = []
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)

    main._g3d_execute_simulation(
        [{"phase": "moving_to_target", "joints": trajectory[-1], "trajectory": trajectory}],
        initial,
        threading.Event(),
        lambda phase, joints: published.append((phase, dict(joints))),
    )

    assert published == [("starting", initial)] + [("moving_to_target", joints) for joints in trajectory]


def test_generate3d_pick_place_uses_partial_gripper_opening(monkeypatch):
    initial = {name: 0.0 for name in main.ROBOT_JOINTS}
    initial["gripper"] = 35.0

    monkeypatch.setattr(main, "_g3d_predict_from_pixel", lambda pixel, image_size=None, height_cm=0, calibration=None: {
        "position_3d": [0.0, height_cm / 100, 0.0],
        "joints": {
            "shoulder_pan": 0.0,
            "shoulder_lift": -height_cm,
            "elbow_flex": 20.0,
            "wrist_flex": 30.0,
            "wrist_roll": 0.0,
            "gripper": 35.0,
        },
    })

    plan = main._g3d_build_pick_place_plan(
        OBJECTS[0], OBJECTS[1], [800, 600], 1.0, 10.0, initial,
    )
    by_phase = {step["phase"]: step["joints"] for step in plan}

    assert by_phase["opening_gripper"]["gripper"] == 50.0
    assert by_phase["descending_to_source"]["gripper"] == 50.0
    assert by_phase["releasing"]["gripper"] == 50.0
    assert by_phase["lifting_after_release"]["gripper"] == 50.0


def test_smooth_pick_place_plan_curves_above_transfer_height(monkeypatch):
    initial = {name: 0.0 for name in main.ROBOT_JOINTS}
    initial["gripper"] = 35.0

    def fake_predict(pixel, image_size=None, height_cm=0, calibration=None):
        return {
            "position_3d": [pixel[0] / 1000, height_cm / 100, pixel[1] / 1000],
            "joints": {
                "shoulder_pan": pixel[0] / 10,
                "shoulder_lift": -height_cm,
                "elbow_flex": 20.0,
                "wrist_flex": 30.0,
                "wrist_roll": 0.0,
                "gripper": 35.0,
            },
        }

    monkeypatch.setattr(main, "_g3d_predict_from_pixel", fake_predict)

    plan = main._g3d_build_smooth_pick_place_plan(
        OBJECTS[0], OBJECTS[1], [800, 600], 5.0, 18.0, initial,
    )
    by_phase = {step["phase"]: step for step in plan}
    transfer = by_phase["moving_to_target"]

    assert len(transfer["trajectory"]) >= 8
    assert all(sample["height_cm"] >= 18.0 for sample in transfer["path_samples"])
    assert transfer["path_samples"][0]["pixel"] == OBJECTS[0]["center_pixel"]
    assert transfer["path_samples"][-1]["pixel"] == OBJECTS[1]["center_pixel"]
    assert max(sample["height_cm"] for sample in transfer["path_samples"]) > 18.0
    assert all(sample["pixel"] == OBJECTS[0]["center_pixel"] for sample in by_phase["lifting_source"]["path_samples"])
    assert all(sample["pixel"] == OBJECTS[1]["center_pixel"] for sample in by_phase["placing"]["path_samples"])


def test_smooth_pick_place_plan_rejects_invalid_intermediate_joint(monkeypatch):
    initial = {name: 0.0 for name in main.ROBOT_JOINTS}

    def invalid_middle_prediction(pixel, image_size=None, height_cm=0, calibration=None):
        shoulder_pan = 999.0 if 400 < pixel[0] < 500 else 0.0
        return {
            "position_3d": [0.0, height_cm / 100, 0.0],
            "joints": {
                "shoulder_pan": shoulder_pan,
                "shoulder_lift": -height_cm,
                "elbow_flex": 20.0,
                "wrist_flex": 30.0,
                "wrist_roll": 0.0,
                "gripper": 50.0,
            },
        }

    monkeypatch.setattr(main, "_g3d_predict_from_pixel", invalid_middle_prediction)

    with pytest.raises(RuntimeError, match="smooth moving_to_target trajectory"):
        main._g3d_build_smooth_pick_place_plan(
            OBJECTS[0], OBJECTS[1], [800, 600], 5.0, 18.0, initial,
        )


def test_generate3d_height_is_offset_above_calibrated_surface(monkeypatch):
    calibration = {
        "model": {
            "type": "affine",
            "world_coeff": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "joint_coeff": [[0.0] * len(main.ROBOT_JOINTS) for _ in range(3)],
        },
        "last_image_size": [640, 480],
        "points": [
            {"pixel": [0, 0], "position_3d": [0.0, 0.08, 0.0], "joints": {name: 0.0 for name in main.ROBOT_JOINTS}},
            {"pixel": [639, 0], "position_3d": [0.0, 0.09, 0.0], "joints": {name: 0.0 for name in main.ROBOT_JOINTS}},
            {"pixel": [0, 479], "position_3d": [0.0, 0.10, 0.0], "joints": {name: 0.0 for name in main.ROBOT_JOINTS}},
        ],
    }
    monkeypatch.setattr(
        main,
        "_g3d_ik_solve",
        lambda target, calibration=None, preferred_joints=None: [0.0] * 5,
    )

    prediction = main._g3d_predict_from_pixel(
        [320, 240], image_size=[640, 480], height_cm=10.0, calibration=calibration,
    )

    assert prediction["position_3d"][1] == pytest.approx(0.19)


def test_generate3d_bowl_height_move_preserves_calibrated_wrist_orientation():
    seed_joints = {
        "shoulder_pan": 24.04,
        "shoulder_lift": 39.34,
        "elbow_flex": -16.84,
        "wrist_flex": 70.15,
        "wrist_roll": -7.52,
        "gripper": 0.47,
    }
    bowl_joints = {
        "shoulder_pan": 24.32,
        "shoulder_lift": 38.93,
        "elbow_flex": -19.78,
        "wrist_flex": 66.27,
        "wrist_roll": -9.98,
        "gripper": 0.47,
    }
    calibration = {
        "model": {
            "type": "affine",
            "world_coeff": [[-0.09206, 0.28918], [0.0, 0.0], [0.0, 0.0]],
            "joint_coeff": [
                [bowl_joints[name] for name in main.ROBOT_JOINTS],
                [0.0] * len(main.ROBOT_JOINTS),
                [0.0] * len(main.ROBOT_JOINTS),
            ],
        },
        "last_image_size": [640, 480],
        "points": [{
            "pixel": [503, 105],
            "position_3d": [-0.07418, 0.0875, 0.27481],
            "joints": seed_joints,
        }],
    }

    prediction = main._g3d_predict_from_pixel(
        [506, 93.5], image_size=[640, 480], height_cm=5.0, calibration=calibration,
    )

    assert prediction["joints"]["wrist_flex"] == pytest.approx(66.27, abs=0.01)
    assert prediction["joints"]["wrist_roll"] == pytest.approx(-9.98, abs=0.01)
    actual_position = main._g3d_position_from_joints(prediction["joints"])
    assert actual_position == pytest.approx(prediction["position_3d"], abs=0.005)


@pytest.mark.parametrize("preferred_joints", [
    1.0,
    [0.0, 0.0, 0.0, float("nan"), 0.0],
    [0.0, 0.0, 0.0, 95.0, 0.0],
])
def test_generate3d_ik_rejects_invalid_calibrated_wrist(preferred_joints):
    calibration = {
        "points": [{
            "joints": {name: 0.0 for name in main.ROBOT_JOINTS},
        }],
    }

    assert main._g3d_ik_solve(
        [0.0394131, -0.3008737, 0.2870517],
        calibration=calibration,
        preferred_joints=preferred_joints,
    ) is None


@pytest.mark.parametrize(("predicted_joints", "world_xz"), [
    ([0.0, 0.0, 0.0, 95.0, 0.0, 0.0], [0.0394131, 0.3008737]),
    ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0]),
    ([float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0]),
])
def test_generate3d_calibration_move_rejects_invalid_prediction_before_hardware(
    monkeypatch, predicted_joints, world_xz,
):
    joints = {name: 0.0 for name in main.ROBOT_JOINTS}
    calibration = {
        "model": {
            "type": "affine",
            "world_coeff": [world_xz, [0.0, 0.0], [0.0, 0.0]],
            "joint_coeff": [
                predicted_joints,
                [0.0] * len(predicted_joints),
                [0.0] * len(predicted_joints),
            ],
        },
        "last_image_size": [640, 480],
        "points": [{
            "pixel": [320, 240],
            "position_3d": [0.0394131, 0.2870517, 0.3008737],
            "joints": joints,
        }],
    }
    sent = []
    monkeypatch.setattr(main, "g3d_calib_state", calibration)
    monkeypatch.setitem(main.robot_state, "connected", True)
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(joints))
    monkeypatch.setattr(main, "robot_send_positions", lambda command, owner=None: sent.append(dict(command)))

    response = TestClient(main.app).post("/api/generate3d/calibration/move-to", json={
        "pixel": [320, 240],
        "image_size": [640, 480],
    })

    assert response.json() == {"ok": False, "error": "Prediction failed"}
    assert sent == []


def test_pick_place_releases_five_centimeters_above_target(monkeypatch):
    calls = []

    def fake_predict(pixel, image_size=None, height_cm=0.0, calibration=None):
        calls.append((list(pixel), float(height_cm)))
        return {
            "position_3d": [0.0, height_cm / 100, 0.0],
            "joints": {
                **{name: 0.0 for name in main.ROBOT_JOINTS},
                "shoulder_lift": -float(height_cm),
            },
        }

    monkeypatch.setattr(main, "_g3d_predict_from_pixel", fake_predict)
    initial = {name: 0.0 for name in main.ROBOT_JOINTS}

    plan = main._g3d_build_pick_place_plan(
        OBJECTS[0], OBJECTS[1], [800, 600], 0.0, 10.0, initial,
    )

    assert calls == [
        ([392.5, 169.0], 0.0),
        ([392.5, 169.0], 10.0),
        ([509.0, 91.0], 10.0),
        ([509.0, 91.0], 10.0),
    ]
    by_phase = {step["phase"]: step["joints"] for step in plan}
    assert by_phase["moving_to_target"] == by_phase["lifting_source"]
    assert by_phase["placing"] == by_phase["moving_to_target"]
    assert by_phase["releasing"]["shoulder_lift"] == -10.0


def test_pick_place_preserves_target_heights_above_thirty_centimeters(monkeypatch):
    calls = []

    def fake_predict(pixel, image_size=None, height_cm=0.0, calibration=None):
        calls.append((list(pixel), float(height_cm)))
        return {
            "position_3d": [0.0, height_cm / 100, 0.0],
            "joints": {name: 0.0 for name in main.ROBOT_JOINTS},
        }

    monkeypatch.setattr(main, "_g3d_predict_from_pixel", fake_predict)
    initial = {name: 0.0 for name in main.ROBOT_JOINTS}
    tall_target = {**OBJECTS[1], "estimated_size_cm": [10.0, 10.0, 40.0]}

    main._g3d_build_pick_place_plan(
        OBJECTS[0], tall_target, [800, 600], 0.0, 10.0, initial,
    )

    assert calls[2:] == [
        ([509.0, 91.0], 45.0),
        ([509.0, 91.0], 45.0),
    ]


def test_generate3d_task_rejects_unknown_execution_mode(monkeypatch):
    setup_task_api(monkeypatch)
    client = TestClient(main.app)

    response = client.post("/api/generate3d/task/start", json={
        "instruction": "pick up white star to teal bowl",
        "detection_id": "det-1",
        "execution_mode": "unsafe",
    })

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_EXECUTION_MODE"


def test_calibrated_workspace_hull_normalizes_different_image_sizes(monkeypatch):
    setup_task_api(monkeypatch)

    assert main._g3d_point_in_calibrated_workspace([800, 300], [1600, 1200]) is True
    assert main._g3d_point_in_calibrated_workspace([1700, 300], [1600, 1200]) is False


def test_calibrated_workspace_allows_small_margin_around_sample_hull(monkeypatch):
    setup_task_api(monkeypatch)
    main.g3d_calib_state["last_image_size"] = [640, 480]
    main.g3d_calib_state["points"] = [
        {"pixel": [275, 224]},
        {"pixel": [401, 176]},
        {"pixel": [503, 105]},
        {"pixel": [585, 214]},
        {"pixel": [337, 357]},
        {"pixel": [463, 283]},
    ]

    assert main._g3d_point_in_calibrated_workspace([508.5, 95], [640, 480]) is True
    assert main._g3d_point_in_calibrated_workspace([331.5, 186], [640, 480]) is True


def test_generate3d_calibration_mutation_invalidates_detection(monkeypatch):
    setup_task_api(monkeypatch)
    monkeypatch.setattr(main, "save_g3d_calibration", lambda: None)
    client = TestClient(main.app)

    response = client.post("/api/generate3d/calibration/reset")

    assert response.status_code == 200
    assert main.g3d_detection_state["id"] is None
    assert main.g3d_detection_state["objects"] == []


def test_generate3d_detection_update_recomputes_bbox_center_and_world_position(monkeypatch):
    setup_task_api(monkeypatch)
    monkeypatch.setattr(main, "_g3d_predict_from_pixel", lambda pixel, image_size=None, height_cm=0, calibration=None: {
        "position_3d": [pixel[0] / 1000, 0.0, pixel[1] / 1000],
        "joints": {name: 0.0 for name in main.ROBOT_JOINTS},
    })
    client = TestClient(main.app)

    response = client.post("/api/generate3d/detection/update", json={
        "detection_id": "det-1",
        "objects": [
            {**OBJECTS[0], "name": "yellow star", "bbox": [100, 120, 300, 320]},
            {**OBJECTS[1], "bbox": [400, 100, 600, 300]},
        ],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["detection_id"] != "det-1"
    assert data["objects"][0]["name"] == "yellow star"
    assert data["objects"][0]["center_pixel"] == [200.0, 220.0]
    assert data["objects"][0]["position_3d"] == [0.2, 0.0, 0.22]
    assert main.g3d_detection_state["objects"] == data["objects"]


def test_generate3d_manual_detection_creates_scene_without_gemini(monkeypatch):
    setup_task_api(monkeypatch)
    monkeypatch.setattr(main, "_g3d_predict_from_pixel", lambda pixel, image_size=None, height_cm=0, calibration=None: {
        "position_3d": [pixel[0] / 1000, 0.0, pixel[1] / 1000],
        "joints": {name: 0.0 for name in main.ROBOT_JOINTS},
    })
    main._g3d_invalidate_detection()
    client = TestClient(main.app)

    response = client.post("/api/generate3d/detection/manual", json={
        "image_size": [800, 600],
        "objects": [{
            "name": "manual cube",
            "bbox": [100, 120, 300, 320],
            "color_hex": "#1a73e8",
            "shape_3d": "box",
            "estimated_size_cm": [3, 4, 5],
        }],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["detection_id"]
    assert data["objects"][0]["center_pixel"] == [200.0, 220.0]
    assert data["objects"][0]["position_3d"] == [0.2, 0.0, 0.22]
    assert main.g3d_detection_state["image_size"] == [800, 600]
    assert main.g3d_detection_state["calibration_revision"] == main._g3d_calibration_revision()


def test_generate3d_manual_detection_rejects_duplicate_names(monkeypatch):
    setup_task_api(monkeypatch)
    client = TestClient(main.app)

    response = client.post("/api/generate3d/detection/manual", json={
        "image_size": [800, 600],
        "objects": [
            {"name": "cube", "bbox": [100, 100, 200, 200]},
            {"name": " Cube ", "bbox": [300, 100, 400, 200]},
        ],
    })

    assert response.status_code == 400
    assert response.json()["code"] == "DUPLICATE_OBJECT_NAME"


def test_generate3d_manual_detection_rechecks_task_after_body_is_read(monkeypatch):
    setup_task_api(monkeypatch)
    monkeypatch.setattr(main, "_g3d_predict_from_pixel", lambda *args, **kwargs: {
        "position_3d": [0.0, 0.0, 0.0],
        "joints": {name: 0.0 for name in main.ROBOT_JOINTS},
    })
    running = {"value": False}
    monkeypatch.setattr(main.g3d_task_manager, "status", lambda: {"running": running["value"]})

    class RequestThatStartsTask:
        async def json(self):
            running["value"] = True
            return {
                "image_size": [800, 600],
                "objects": [{"name": "cube", "bbox": [100, 100, 200, 200]}],
            }

    response = asyncio.run(main.generate3d_detection_manual(RequestThatStartsTask()))

    assert response.status_code == 409
    assert json.loads(response.body)["code"] == "TASK_RUNNING"


def test_generate3d_detection_update_rejects_duplicate_names(monkeypatch):
    setup_task_api(monkeypatch)
    client = TestClient(main.app)

    response = client.post("/api/generate3d/detection/update", json={
        "detection_id": "det-1",
        "objects": [
            {**OBJECTS[0], "name": "cup", "bbox": [100, 100, 200, 200]},
            {**OBJECTS[1], "name": " Cup ", "bbox": [300, 100, 400, 200]},
        ],
    })

    assert response.status_code == 400
    assert response.json()["code"] == "DUPLICATE_OBJECT_NAME"


def test_generate3d_detection_update_treats_bbox_as_strict_pixels(monkeypatch):
    setup_task_api(monkeypatch)
    client = TestClient(main.app)

    for bbox in ([0, 0, 1, 1], [790, 590, 900, 700]):
        response = client.post("/api/generate3d/detection/update", json={
            "detection_id": "det-1",
            "objects": [{**OBJECTS[0], "bbox": bbox}],
        })
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_BBOX"


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
        "raising_to_safety",
        "moving_to_source",
        "opening_gripper",
        "descending_to_source",
        "grasping",
        "lifting_source",
        "moving_to_target",
        "placing",
        "releasing",
        "lifting_after_release",
    ]
    assert sent[-1]["gripper"] == main.G3D_TASK_OPEN_GRIPPER
    assert published[-1][1] == measured

    carrying = [joints for phase, joints in published if phase in {"lifting_source", "moving_to_target", "placing"}]
    assert carrying
    assert all(joints["gripper"] == 0.0 for joints in carrying)


def test_pick_place_can_hold_current_joints_outside_planned_range_during_preparation(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured.update(shoulder_lift=-93.23, gripper=105.0)
    sent = []

    def fake_predict(pixel, image_size=None, height_cm=0.0, calibration=None):
        return {
            "position_3d": [0.0, height_cm / 100, 0.0],
            "joints": {
                "shoulder_pan": 0.0,
                "shoulder_lift": -height_cm,
                "elbow_flex": 20.0,
                "wrist_flex": 30.0,
                "wrist_roll": 0.0,
                "gripper": 25.0,
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
        threading.Event(), lambda phase, joints: None,
    )

    assert sent
    assert sent[0]["shoulder_lift"] == -93.23
    assert sent[0]["gripper"] == 105.0
    assert sent[-1]["gripper"] == main.G3D_TASK_OPEN_GRIPPER


def test_pick_place_rejects_invalid_current_state_before_first_command(monkeypatch):
    current = {name: 0.0 for name in main.ROBOT_JOINTS}
    current["shoulder_lift"] = float("nan")
    moves = []

    def fake_predict(pixel, image_size=None, height_cm=0.0, calibration=None):
        return {
            "position_3d": [0.0, height_cm / 100, 0.0],
            "joints": {
                "shoulder_pan": 0.0,
                "shoulder_lift": -height_cm,
                "elbow_flex": 20.0,
                "wrist_flex": 30.0,
                "wrist_roll": 0.0,
                "gripper": 50.0,
            },
        }

    monkeypatch.setattr(main, "_g3d_predict_from_pixel", fake_predict)
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(current))
    monkeypatch.setattr(main, "_g3d_task_move", lambda *args, **kwargs: moves.append(args) or True)

    with pytest.raises(RuntimeError, match="Invalid current joint"):
        main._g3d_execute_pick_place(
            OBJECTS[0], OBJECTS[1], [800, 600], 1.0, 10.0,
            threading.Event(), lambda phase, joints: None,
        )

    assert moves == []


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


def test_pick_place_accepts_elbow_servo_residual_up_to_three_degrees(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured.update(elbow_flex=38.02, gripper=50.0)
    clock = iter(range(0, 10000, 10))

    def fake_predict(pixel, image_size=None, height_cm=0.0, calibration=None):
        return {
            "position_3d": [0.0, height_cm / 100, 0.0],
            "joints": {
                "shoulder_pan": 0.0,
                "shoulder_lift": -height_cm,
                "elbow_flex": 35.42,
                "wrist_flex": 30.0,
                "wrist_roll": 0.0,
                "gripper": 50.0,
            },
        }

    def sticky_elbow_send(command, owner=None):
        assert owner == "generate3d"
        for name, value in command.items():
            if name != "elbow_flex":
                measured[name] = value

    monkeypatch.setattr(main, "_g3d_predict_from_pixel", fake_predict)
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", sticky_elbow_send)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(main.time, "monotonic", lambda: float(next(clock)))

    main._g3d_execute_pick_place(
        OBJECTS[0], OBJECTS[1], [800, 600], 1.0, 10.0,
        threading.Event(), lambda phase, joints: None,
    )

    assert abs(measured["elbow_flex"] - 35.42) == pytest.approx(2.60)


def test_task_move_allows_intermediate_servo_lag_before_reaching_final_target(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured["elbow_flex"] = 96.04
    target = {**measured, "elbow_flex": 82.62}
    sent = []
    clock = iter([0.0, 10.0, 20.0])

    def lag_then_reach(command, owner=None):
        assert owner == "generate3d"
        sent.append(dict(command))
        if len(sent) == 1:
            measured["elbow_flex"] = 91.91
        else:
            measured.update(command)

    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", lag_then_reach)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(main.time, "monotonic", lambda: next(clock))

    completed = main._g3d_task_move(
        target, "moving_to_source", threading.Event(), lambda phase, joints: None,
        n_steps=2, tolerance_deg=2.5,
    )

    assert completed is True
    assert sent[0]["elbow_flex"] == 89.33
    assert measured["elbow_flex"] == 82.62


def test_task_move_still_rejects_final_arm_residual_over_tolerance(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured["elbow_flex"] = 91.91
    target = {**measured, "elbow_flex": 89.33}
    clock = iter([0.0, 10.0])
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(main.time, "monotonic", lambda: next(clock))

    with pytest.raises(RuntimeError, match=r"elbow_flex\(target=89.33, measured=91.91, error=2.58\)"):
        main._g3d_task_move(
            target, "moving_to_source", threading.Event(), lambda phase, joints: None,
            n_steps=1, tolerance_deg=2.5,
        )


def test_real_plan_rejects_elbow_residual_over_three_degrees(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured["elbow_flex"] = 38.43
    target = {**measured, "elbow_flex": 35.42}
    clock = iter([0.0, 10.0])
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(main.time, "monotonic", lambda: next(clock))

    with pytest.raises(RuntimeError, match=r"elbow_flex\(target=35.42, measured=38.43, error=3.01\)"):
        main._g3d_execute_real_plan(
            [{
                "phase": "moving_to_source",
                "joints": target,
                "n_steps": 1,
                "convergence_names": main.ROBOT_JOINTS[:5],
                "tolerance_deg": main.G3D_TASK_ARM_TOLERANCE_DEG,
            }],
            threading.Event(),
            lambda phase, joints: None,
        )


def test_real_plan_accepts_shoulder_lift_residual_up_to_three_degrees(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured["shoulder_lift"] = -15.25
    target = {**measured, "shoulder_lift": -17.83}
    clock = iter(range(0, 10000, 10))
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(main.time, "monotonic", lambda: next(clock))

    main._g3d_execute_real_plan(
        [{
            "phase": "lifting_source",
            "joints": target,
            "n_steps": 1,
            "convergence_names": main.ROBOT_JOINTS[:5],
            "tolerance_deg": main.G3D_TASK_ARM_TOLERANCE_DEG,
        }],
        threading.Event(),
        lambda phase, joints: None,
    )

    assert abs(measured["shoulder_lift"] - (-17.83)) == pytest.approx(2.58)


def test_lifting_source_accepts_loaded_shoulder_residual_up_to_four_degrees(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured["shoulder_lift"] = 16.04
    target = {**measured, "shoulder_lift": 12.50}
    clock = iter(range(0, 10000, 10))
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(main.time, "monotonic", lambda: next(clock))

    main._g3d_execute_real_plan(
        [{
            "phase": "lifting_source",
            "joints": target,
            "n_steps": 1,
            "convergence_names": main.ROBOT_JOINTS[:5],
            "tolerance_deg": main.G3D_TASK_ARM_TOLERANCE_DEG,
            "joint_tolerances": {"shoulder_lift": 4.0},
        }],
        threading.Event(),
        lambda phase, joints: None,
    )


def test_lifting_source_rejects_loaded_shoulder_residual_over_four_degrees(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured["shoulder_lift"] = 16.51
    target = {**measured, "shoulder_lift": 12.50}
    clock = iter([0.0, 10.0])
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(main.time, "monotonic", lambda: next(clock))

    with pytest.raises(RuntimeError, match=r"shoulder_lift\(target=12.50, measured=16.51, error=4.01\)"):
        main._g3d_execute_real_plan(
            [{
                "phase": "lifting_source",
                "joints": target,
                "n_steps": 1,
                "convergence_names": main.ROBOT_JOINTS[:5],
                "tolerance_deg": main.G3D_TASK_ARM_TOLERANCE_DEG,
                "joint_tolerances": {"shoulder_lift": 4.0},
            }],
            threading.Event(),
            lambda phase, joints: None,
        )


def test_real_plan_keeps_strict_gripper_tolerance(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured["gripper"] = 52.06
    target = {**measured, "gripper": 50.0}
    clock = iter([0.0, 10.0])
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(main.time, "monotonic", lambda: next(clock))

    with pytest.raises(RuntimeError, match=r"gripper\(target=50.00, measured=52.06, error=2.06\)"):
        main._g3d_execute_real_plan(
            [{
                "phase": "opening_gripper",
                "joints": target,
                "n_steps": 1,
                "convergence_names": ["gripper"],
                "tolerance_deg": main.G3D_TASK_GRIPPER_TOLERANCE,
            }],
            threading.Event(),
            lambda phase, joints: None,
        )


def test_waypoint_timeout_scales_with_joint_delta_and_stays_bounded():
    current = {name: 0.0 for name in main.ROBOT_JOINTS}

    small = main._g3d_waypoint_timeout(current, {**current, "shoulder_pan": 5.0}, main.ROBOT_JOINTS[:5])
    large = main._g3d_waypoint_timeout(current, {**current, "shoulder_pan": 100.0}, main.ROBOT_JOINTS[:5])

    assert small == 3.5
    assert large == 8.0


def test_task_move_timeout_reports_unconverged_joint_details(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    target = {**measured, "shoulder_pan": 10.0}
    clock = iter([0.0, 10.0])
    monkeypatch.setattr(main, "robot_get_positions", lambda: dict(measured))
    monkeypatch.setattr(main, "robot_send_positions", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(main.time, "monotonic", lambda: next(clock))

    with pytest.raises(RuntimeError) as exc_info:
        main._g3d_task_move(target, "moving_to_source", threading.Event(), lambda phase, joints: None, n_steps=1)

    message = str(exc_info.value)
    assert "moving_to_source" in message
    assert "shoulder_pan" in message
    assert "target=10.00" in message
    assert "measured=0.00" in message
    assert "error=10.00" in message


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
    assert sent[-1]["gripper"] == main.G3D_TASK_OPEN_GRIPPER
