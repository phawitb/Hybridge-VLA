import time
import threading

from verified_execution import VerifiedExecutionManager


def step(index=1, task="pick up the bow", model="model_a"):
    return {
        "step_index": index,
        "description": task,
        "target_bbox": None,
        "method_id": "vla_model",
        "model_id": model,
    }


def settings(cycles=5, replans=3):
    return {
        "actions_per_cycle": 100,
        "cycles_before_replan": cycles,
        "max_replans": replans,
    }


def wait_terminal(manager, timeout=2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = manager.status()
        if state["state"] in {"completed", "needs_human_review", "stopped"}:
            return state
        time.sleep(0.01)
    raise AssertionError(manager.status())


def test_success_advances_through_remaining_plan_with_snapshotted_actions():
    calls = []
    manager = VerifiedExecutionManager()
    plan = {"task_id": "two", "total_steps": 2, "steps": [step(), step(2, "place the bow")]}

    result = manager.start(
        "pick and place",
        plan,
        0,
        settings(),
        lambda current, actions, stop: calls.append((current["step_index"], actions)) or {"ok": True},
        lambda current: {"ok": True, "status": "success", "reason": "done", "visible_evidence": "released"},
        lambda context: {"ok": False, "error": "must not replan"},
    )
    state = wait_terminal(manager)

    assert result["ok"] is True
    assert calls == [(1, 100), (2, 100)]
    assert state["state"] == "completed"
    assert [item["step_index"] for item in state["completed_steps"]] == [1, 2]


def test_prepare_finishes_before_first_step_executes():
    events = []
    manager = VerifiedExecutionManager()
    manager.start(
        "pick",
        {"steps": [step()]},
        0,
        settings(),
        lambda current, actions, stop: events.append("execute") or {"ok": True},
        lambda current: {"ok": True, "status": "success"},
        lambda context: {"ok": False},
        prepare=lambda plan, index: events.append("prepare") or {"ok": True},
    )

    state = wait_terminal(manager)
    assert events == ["prepare", "execute"]
    assert state["state"] == "completed"


def test_prepare_failure_prevents_first_step():
    events = []
    manager = VerifiedExecutionManager()
    manager.start(
        "pick", {"steps": [step()]}, 0, settings(),
        lambda current, actions, stop: events.append("execute") or {"ok": True},
        lambda current: {"ok": True, "status": "success"},
        lambda context: {"ok": False},
        prepare=lambda plan, index: {"ok": False, "error": "model load failed"},
    )

    state = wait_terminal(manager)
    assert events == []
    assert state["state"] == "needs_human_review"
    assert state["error"] == "model load failed"


def test_stop_during_prepare_prevents_first_step():
    entered = threading.Event()
    release = threading.Event()
    events = []
    manager = VerifiedExecutionManager()

    def prepare(plan, index):
        entered.set()
        release.wait(1)
        return {"ok": True}

    manager.start(
        "pick", {"steps": [step()]}, 0, settings(),
        lambda current, actions, stop: events.append("execute") or {"ok": True},
        lambda current: {"ok": True, "status": "success"},
        lambda context: {"ok": False},
        prepare=prepare,
    )
    assert entered.wait(1)
    manager.stop()
    release.set()

    state = wait_terminal(manager)
    assert state["state"] == "stopped"
    assert events == []


def test_continue_and_uncertain_retry_same_task_until_success():
    verdicts = iter(["continue", "uncertain", "success"])
    executions = []
    manager = VerifiedExecutionManager()
    manager.start(
        "pick",
        {"steps": [step()]},
        0,
        settings(),
        lambda current, actions, stop: executions.append(current["description"]) or {"ok": True},
        lambda current: {"ok": True, "status": next(verdicts), "reason": "check", "visible_evidence": "scene"},
        lambda context: {"ok": False},
    )

    state = wait_terminal(manager)
    assert executions == ["pick up the bow"] * 3
    assert state["cycle"] == 3
    assert [item["status"] for item in state["verification_history"]] == ["continue", "uncertain", "success"]


def test_cycle_limit_replans_only_remaining_work_with_completed_context():
    contexts = []
    manager = VerifiedExecutionManager()
    manager.start(
        "pick then place",
        {"steps": [step(1, "pick"), step(2, "place")]},
        0,
        settings(cycles=2),
        lambda current, actions, stop: {"ok": True},
        lambda current: {"ok": True, "status": "success" if current["description"] == "pick" else "continue", "reason": "x", "visible_evidence": "x"},
        lambda context: contexts.append(context) or {"ok": True, "plan": {"steps": [step(1, "recover place")] }},
    )

    deadline = time.time() + 2
    while time.time() < deadline and not contexts:
        time.sleep(0.01)
    manager.stop()

    assert contexts[0]["original_instruction"] == "pick then place"
    assert [item["description"] for item in contexts[0]["completed_steps"]] == ["pick"]
    assert contexts[0]["failed_step"]["description"] == "place"
    assert len(contexts[0]["verification_history"]) == 2


def test_replacement_plan_is_prepared_before_its_first_step():
    events = []
    manager = VerifiedExecutionManager()
    verdicts = iter(["continue", "success"])
    manager.start(
        "pick", {"steps": [step()]}, 0, settings(cycles=1),
        lambda current, actions, stop: events.append(("execute", current["description"])) or {"ok": True},
        lambda current: {"ok": True, "status": next(verdicts)},
        lambda context: {"ok": True, "plan": {"steps": [step(1, "replacement")] }},
        prepare=lambda plan, index: events.append(("prepare", plan["steps"][index]["description"])) or {"ok": True},
    )

    state = wait_terminal(manager)
    assert state["state"] == "completed"
    assert events == [
        ("prepare", "pick up the bow"),
        ("execute", "pick up the bow"),
        ("prepare", "replacement"),
        ("execute", "replacement"),
    ]


def test_replan_limit_enters_human_review():
    manager = VerifiedExecutionManager()
    manager.start(
        "pick",
        {"steps": [step()]},
        0,
        settings(cycles=1, replans=1),
        lambda current, actions, stop: {"ok": True},
        lambda current: {"ok": True, "status": "continue", "reason": "not done", "visible_evidence": "held"},
        lambda context: {"ok": True, "plan": {"steps": [step(1, "pick up the bow")] }},
    )

    state = wait_terminal(manager)
    assert state["state"] == "needs_human_review"
    assert state["replan_count"] == 1
    assert "re-plan limit" in state["error"]


def test_execution_failure_enters_human_review_without_verification():
    verified = []
    manager = VerifiedExecutionManager()
    manager.start(
        "pick",
        {"steps": [step()]},
        0,
        settings(),
        lambda current, actions, stop: {"ok": False, "error": "motor failed"},
        lambda current: verified.append(True) or {"ok": True, "status": "success"},
        lambda context: {"ok": False},
    )

    state = wait_terminal(manager)
    assert state["state"] == "needs_human_review"
    assert state["error"] == "motor failed"
    assert verified == []


def test_stop_cancels_loop_and_prevents_restart():
    manager = VerifiedExecutionManager()

    def execute(current, actions, stop_event):
        while not stop_event.wait(0.01):
            pass
        return {"ok": False, "error": "stopped"}

    manager.start(
        "pick", {"steps": [step()]}, 0, settings(), execute,
        lambda current: {"ok": True, "status": "success"},
        lambda context: {"ok": False},
    )
    time.sleep(0.03)
    result = manager.stop()
    state = wait_terminal(manager)

    assert result["ok"] is True
    assert state["state"] == "stopped"
    assert state["completed_steps"] == []


def test_single_step_mode_stops_after_selected_step_success():
    calls = []
    manager = VerifiedExecutionManager()
    manager.start(
        "pick then place", {"steps": [step(1, "pick"), step(2, "place")]}, 1,
        settings(),
        lambda current, actions, stop: calls.append(current["description"]) or {"ok": True},
        lambda current: {"ok": True, "status": "success", "reason": "done"},
        lambda context: {"ok": False},
        run_mode="step",
    )

    state = wait_terminal(manager)
    assert calls == ["place"]
    assert state["state"] == "completed"
    assert state["run_mode"] == "step"
    assert [item["description"] for item in state["completed_steps"]] == ["place"]


def test_single_step_mode_stops_after_replan_for_user_selection():
    manager = VerifiedExecutionManager()
    manager.start(
        "pick", {"steps": [step()]}, 0, settings(cycles=1),
        lambda current, actions, stop: {"ok": True},
        lambda current: {"ok": True, "status": "continue", "reason": "not done"},
        lambda context: {"ok": True, "plan": {"steps": [step(1, "recover")] }},
        run_mode="step",
    )

    deadline = time.time() + 2
    while time.time() < deadline:
        state = manager.status()
        if not state["running"]:
            break
        time.sleep(0.01)
    assert state["state"] == "awaiting_step_selection"
    assert state["plan"]["steps"][0]["description"] == "recover"
