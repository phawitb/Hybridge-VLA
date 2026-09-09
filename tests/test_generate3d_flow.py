import json
import time

import pytest

from generate3d_flow import FlowValidationError, Generate3DFlowManager, parse_flow_plan


def wait_until(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def sample_plan(count=2):
    return {"subtasks": [
        {"instruction": f"Pick item {i} to bowl", "source_name": f"item {i}", "target_name": "bowl"}
        for i in range(count)
    ]}


def test_parse_flow_plan_validates_atomic_subtasks():
    blocks = parse_flow_plan(json.dumps(sample_plan()))
    assert [block["source_name"] for block in blocks] == ["item 0", "item 1"]
    with pytest.raises(FlowValidationError):
        parse_flow_plan({"subtasks": []})
    with pytest.raises(FlowValidationError):
        parse_flow_plan(sample_plan(11))
    with pytest.raises(FlowValidationError):
        parse_flow_plan({"subtasks": [{"instruction": "move", "source_name": "x"}]})


def test_manager_creates_and_persists_flow(tmp_path):
    state_file = tmp_path / "state.json"
    manager = Generate3DFlowManager(state_file, tmp_path / "artifacts")
    state = manager.create_flow("long task", sample_plan()["subtasks"], planning={"prompt": "full", "raw": "{}"})
    assert state["status"] == "ready"
    assert len(state["blocks"]) == 2
    assert state["blocks"][0]["phase"] == "pending"
    assert json.loads(state_file.read_text())["flow_id"] == state["flow_id"]


def test_restart_marks_active_flow_interrupted_without_running_callbacks(tmp_path):
    state_file = tmp_path / "state.json"
    manager = Generate3DFlowManager(state_file, tmp_path / "artifacts")
    manager.create_flow("long task", sample_plan()["subtasks"])
    saved = manager.status()
    saved.update(status="running", active_block_index=0, run_scope="all")
    saved["blocks"][0]["phase"] = "executing"
    state_file.write_text(json.dumps(saved))

    restored = Generate3DFlowManager(state_file, tmp_path / "artifacts").status()
    assert restored["status"] == "interrupted"
    assert restored["blocks"][0]["phase"] == "interrupted"
    assert restored["active_block_index"] is None


def test_run_all_advances_only_after_success(tmp_path):
    manager = Generate3DFlowManager(tmp_path / "state.json", tmp_path / "artifacts")
    manager.create_flow("long task", sample_plan(3)["subtasks"])
    calls = []

    def runner(block, config, transition, should_stop):
        calls.append(block["index"])
        transition("capturing", {"input": block["instruction"]})
        status = "success" if block["index"] == 0 else "uncertain"
        return {"status": status, "verification": {"status": status, "reason": "checked", "visible_evidence": []}}

    assert manager.start("all", None, {"execution_mode": "simulation"}, runner)["ok"]
    wait_until(lambda: not manager.status()["running"])
    state = manager.status()
    assert calls == [0, 1]
    assert state["status"] == "uncertain"
    assert state["blocks"][2]["phase"] == "pending"


def test_block_run_and_stop_are_scoped_and_idempotent(tmp_path):
    manager = Generate3DFlowManager(tmp_path / "state.json", tmp_path / "artifacts")
    manager.create_flow("long task", sample_plan(2)["subtasks"])
    calls = []

    def runner(block, config, transition, should_stop):
        calls.append(block["index"])
        return {"status": "success"}

    manager.start("block", 1, {}, runner)
    wait_until(lambda: not manager.status()["running"])
    assert calls == [1]
    assert manager.status()["blocks"][0]["phase"] == "pending"
    assert manager.stop()["ok"] is True
    assert manager.stop()["ok"] is True


def test_artifact_path_rejects_traversal(tmp_path):
    manager = Generate3DFlowManager(tmp_path / "state.json", tmp_path / "artifacts")
    flow = manager.create_flow("long task", sample_plan(1)["subtasks"])
    valid = manager.artifact_path(flow["flow_id"], "block-0-before.jpg")
    assert valid.parent.name == flow["flow_id"]
    with pytest.raises(FlowValidationError):
        manager.artifact_path(flow["flow_id"], "../secret")
