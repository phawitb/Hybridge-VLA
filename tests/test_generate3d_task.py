import time

from generate3d_task import Generate3DTaskManager, TaskResolutionError, resolve_pick_place_objects


OBJECTS = [
    {"name": "white star", "center_pixel": [392.5, 169.0]},
    {"name": "teal bowl", "center_pixel": [509.0, 91.0]},
]


def wait_until(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_resolve_pick_place_objects_uses_instruction_mention_order():
    source, target = resolve_pick_place_objects(
        "Pick up the WHITE STAR and put it in the teal bowl",
        list(reversed(OBJECTS)),
    )

    assert source["name"] == "white star"
    assert target["name"] == "teal bowl"


def test_resolve_pick_place_objects_requires_exactly_two_named_objects():
    try:
        resolve_pick_place_objects("Pick up the star", OBJECTS)
    except TaskResolutionError as exc:
        assert exc.code == "OBJECT_MATCH_REQUIRED"
        assert "white star" in str(exc)
        assert "teal bowl" in str(exc)
    else:
        raise AssertionError("an instruction with fewer than two detected names must fail")


def test_resolve_pick_place_objects_does_not_match_inside_a_larger_word():
    try:
        resolve_pick_place_objects("move white starfish to teal bowl", OBJECTS)
    except TaskResolutionError as exc:
        assert exc.code == "OBJECT_MATCH_REQUIRED"
    else:
        raise AssertionError("white star must not match white starfish")


def test_resolve_pick_place_objects_rejects_duplicate_detected_names():
    duplicate_objects = [
        {"name": "white star", "center_pixel": [1, 1]},
        {"name": "white star", "center_pixel": [2, 2]},
    ]
    try:
        resolve_pick_place_objects("move white star to white star", duplicate_objects)
    except TaskResolutionError as exc:
        assert exc.code == "OBJECT_MATCH_REQUIRED"
    else:
        raise AssertionError("duplicate detected names must be ambiguous")


def test_task_manager_publishes_live_joints_and_completes():
    manager = Generate3DTaskManager()

    def execute(source, target, stop_event, publish):
        publish("moving_to_source", {"shoulder_pan": 10.0, "gripper": 100.0})
        publish("grasping", {"shoulder_pan": 12.0, "gripper": 0.0})

    started = manager.start("pick white star to teal bowl", OBJECTS[0], OBJECTS[1], execute)
    wait_until(lambda: not manager.status()["running"])
    status = manager.status()

    assert started["ok"] is True
    assert status["state"] == "completed"
    assert status["phase"] == "completed"
    assert status["source"]["name"] == "white star"
    assert status["target"]["name"] == "teal bowl"
    assert status["joints"] == {"shoulder_pan": 12.0, "gripper": 0.0}


def test_task_manager_stop_interrupts_active_execution():
    manager = Generate3DTaskManager()

    def execute(source, target, stop_event, publish):
        publish("moving_to_source", {"shoulder_pan": 5.0})
        stop_event.wait(1.0)

    manager.start("pick white star to teal bowl", OBJECTS[0], OBJECTS[1], execute)
    wait_until(lambda: manager.status()["phase"] == "moving_to_source")

    result = manager.stop()
    wait_until(lambda: not manager.status()["running"])

    assert result["ok"] is True
    assert result["state"] == "stopped"
    assert result["running"] is False
    assert manager.status()["state"] == "stopped"


def test_task_manager_rejects_overlapping_runs():
    manager = Generate3DTaskManager()

    def execute(source, target, stop_event, publish):
        stop_event.wait(1.0)

    manager.start("first", OBJECTS[0], OBJECTS[1], execute)
    second = manager.start("second", OBJECTS[0], OBJECTS[1], execute)
    manager.stop()

    assert second["ok"] is False
    assert second["code"] == "TASK_ALREADY_RUNNING"


def test_task_manager_does_not_release_hardware_slot_until_stopped_worker_exits():
    manager = Generate3DTaskManager()
    worker_exited = __import__("threading").Event()

    def execute(source, target, stop_event, publish):
        stop_event.wait(1.0)
        time.sleep(0.05)
        worker_exited.set()

    manager.start("first", OBJECTS[0], OBJECTS[1], execute)
    manager.stop(timeout=0.001)

    blocked = manager.start("second", OBJECTS[0], OBJECTS[1], execute)
    assert blocked["ok"] is False
    assert blocked["code"] == "TASK_ALREADY_RUNNING"

    wait_until(worker_exited.is_set)
    wait_until(lambda: not manager.status()["running"])
