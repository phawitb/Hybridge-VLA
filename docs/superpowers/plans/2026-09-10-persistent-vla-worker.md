# Persistent VLA Worker and Outcome Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep one VLA checkpoint warm across verified Run steps and sessions while safely handing robot/camera ownership between the worker and IK, and make completion verification accept clearly achieved final states.

**Architecture:** `infer_python.py` will support a persistent JSON-line worker mode that loads the model before connecting hardware. `VlaProcessManager` will own readiness and hardware-handoff acknowledgements, while `VerifiedExecutionManager` will prepare the first model before executing any step. The existing FastAPI layer will build worker commands, coordinate ownership, and expose lifecycle phases to the Run UI.

**Tech Stack:** Python 3, FastAPI, subprocess/stdin/stdout JSON-line protocol, threading, LeRobot policies and robot APIs, pytest, vanilla JavaScript.

**Spec:** `docs/superpowers/specs/2026-09-10-persistent-vla-worker-design.md`

## Global Constraints

- Keep at most one VLA checkpoint loaded at a time.
- Never allow the main process and VLA worker to own the robot serial port or cameras simultaneously.
- Keep model weights and pre/post processors loaded while hardware is released.
- Reset action queues and all task-specific language caches whenever task text changes.
- Do not start plan step 1 until the first required VLA model emits `MODEL_READY`.
- Reload only for a different model, explicit Stop, server shutdown, or worker failure.
- Preserve action-cycle verification, retry, and re-plan behavior.
- Tests must use fake processes and fake hardware; no physical robot or checkpoint is required.

## File Structure

- `main.py`: build capability-aware verification prompts, build persistent worker commands, and adapt FastAPI verified-run callbacks.
- `planner_config.py`: capability-aware planning verifier template already introduced by the preceding bug fix.
- `infer_python.py`: persistent worker command loop, task activation, cache reset, hardware connect/release, and protocol events.
- `vla_execution.py`: controller-side worker lifecycle, protocol event parsing, readiness waits, hardware handoffs, task commands, and model reuse.
- `verified_execution.py`: preparation phase before the first step and safe terminal behavior.
- `static/index.html`: Run status labels for model loading/readiness and hardware handoffs.
- `tests/test_planner_api.py`, `tests/test_planner_config.py`: capability-aware planning-verifier regression coverage already introduced by the preceding bug fix.
- `tests/test_verified_completion.py`: focused completion-prompt contract tests.
- `tests/test_vla_execution.py`: fake-worker protocol and manager lifecycle tests.
- `tests/test_verified_execution.py`: preparation ordering and failure tests.
- `tests/test_multi_model_ui.js`: status rendering tests.

---

### Task 1: Final-state completion verification

**Files:**
- Modify: `main.py` near `_verified_completion`
- Create: `tests/test_verified_completion.py`

**Interfaces:**
- Consumes: a plan step dictionary with `description`, `method_id`, and `model_id`.
- Produces: `_verified_completion_prompt(step: dict) -> str`, consumed by `_verified_completion(step: dict) -> dict`.

- [ ] **Step 1: Write the failing prompt-contract tests**

```python
import main


def test_pick_place_completion_prompt_prioritizes_requested_final_state():
    prompt = main._verified_completion_prompt({
        "description": "pick up the bow to the green bowl",
        "method_id": "vla_model",
        "model_id": "model_a",
    })

    assert "inside or on the named destination" in prompt
    assert "does not require seeing the earlier pickup" in prompt
    assert "arm is above or near the destination" in prompt
    assert "status, reason, and visible_evidence must agree" in prompt


def test_pick_place_completion_prompt_distinguishes_continue_from_uncertain():
    prompt = main._verified_completion_prompt({"description": "move cube to green bowl"})

    assert "continue only when" in prompt
    assert "uncertain when" in prompt
    assert "specific named destination" in prompt
```

- [ ] **Step 2: Run the tests and verify the missing helper causes failure**

Run: `pytest -q tests/test_verified_completion.py`

Expected: FAIL because `main._verified_completion_prompt` does not exist.

- [ ] **Step 3: Extract and strengthen the completion prompt**

```python
def _verified_completion_prompt(step: dict) -> str:
    task = str(step.get("description", ""))
    return f"""You verify whether the current image visibly satisfies a robot task.
Task: {task}

Judge the requested final spatial state, not whether you observed every earlier motion.
For pick-and-place, return success when the named object is visibly released inside or on the named destination. This does not require seeing the earlier pickup. The fact that the arm is above or near the destination is not evidence of failure when the object is already released correctly.
Return continue only when the object is clearly outside the requested destination or clearly still held by the gripper. Return uncertain when object identity, containment, release, or the specific named destination cannot be determined.
The status, reason, and visible_evidence must agree. Evidence that directly satisfies the requested final state cannot accompany continue.

Return ONLY JSON: {{"status":"success|continue|uncertain","reason":"brief reason","visible_evidence":"what is visible"}}.
"""
```

Change `_verified_completion` to call `_verified_completion_prompt(step)` without changing response parsing.

- [ ] **Step 4: Run focused and planner tests**

Run: `pytest -q tests/test_verified_completion.py tests/test_planner_config.py tests/test_planner_api.py`

Expected: PASS.

- [ ] **Step 5: Commit the completion-verifier fix**

```bash
git add main.py planner_config.py tests/test_verified_completion.py tests/test_planner_config.py tests/test_planner_api.py
git commit -m "fix: verify VLA tasks from visible final state"
```

---

### Task 2: Persistent worker command and event protocol

**Files:**
- Modify: `infer_python.py`
- Modify: `tests/test_vla_execution.py`

**Interfaces:**
- Consumes: newline-delimited JSON commands on stdin.
- Produces: `normalize_worker_command(value: str) -> dict | None`, `reset_policy_task_state(policy, language_cache: dict) -> None`, and prefixed JSON events on stdout.
- Protocol commands: `run_task`, `continue`, `release_hardware`, and `shutdown`.
- Protocol events: `MODEL_READY`, `HARDWARE_READY`, `CYCLE_READY`, `HARDWARE_RELEASED`, and `WORKER_ERROR`.

- [ ] **Step 1: Write failing pure protocol tests**

```python
from infer_python import normalize_worker_command, reset_policy_task_state


def test_worker_command_parser_accepts_only_declared_json_commands():
    assert normalize_worker_command('{"command":"release_hardware"}') == {
        "command": "release_hardware"
    }
    assert normalize_worker_command('{"command":"unknown"}') is None
    assert normalize_worker_command('continue') is None


def test_new_task_clears_actions_and_language_cache():
    class Policy:
        _queues = {"action": [1, 2, 3]}

    cache = {"tokens": object(), "mask": object(), "embeddings": object()}
    reset_policy_task_state(Policy(), cache)

    assert list(Policy._queues["action"]) == []
    assert cache == {"tokens": None, "mask": None, "embeddings": None}
```

- [ ] **Step 2: Run the pure tests and verify they fail because the protocol helpers are absent**

Run: `pytest -q tests/test_vla_execution.py -k 'worker_command_parser or new_task'`

Expected: FAIL at import for the new helpers.

- [ ] **Step 3: Implement strict command parsing and task-state reset**

```python
WORKER_COMMANDS = {"run_task", "continue", "release_hardware", "shutdown"}


def normalize_worker_command(value: str) -> dict | None:
    try:
        command = json.loads(str(value or ""))
    except json.JSONDecodeError:
        return None
    if not isinstance(command, dict) or command.get("command") not in WORKER_COMMANDS:
        return None
    return command


def reset_policy_task_state(policy, language_cache: dict) -> None:
    queue = policy._queues.get("action")
    if hasattr(queue, "clear"):
        queue.clear()
    else:
        policy._queues["action"] = []
    language_cache.update(tokens=None, mask=None, embeddings=None)
```

- [ ] **Step 4: Write a failing subprocess test proving readiness precedes hardware activation**

Add a lightweight `--worker-protocol-test` path that uses fake model/hardware callbacks and test it through `subprocess.Popen`:

```python
def test_worker_emits_model_ready_before_hardware_ready(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-u", "infer_python.py", "--worker-protocol-test", "--model-path", str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    first = process.stdout.readline()
    process.stdin.write(json.dumps({"command": "run_task", "task": "pick", "max_steps": 1}) + "\n")
    process.stdin.flush()
    second = process.stdout.readline()

    assert first.startswith("MODEL_READY ")
    assert second.startswith("HARDWARE_READY ")
    process.stdin.write('{"command":"shutdown"}\n')
    process.stdin.flush()
    process.wait(timeout=2)
```

- [ ] **Step 5: Run the subprocess test and verify the current one-shot CLI fails it**

Run: `pytest -q tests/test_vla_execution.py::test_worker_emits_model_ready_before_hardware_ready`

Expected: FAIL because worker mode and readiness events do not exist.

- [ ] **Step 6: Refactor the runtime into load, connect, run, and release phases**

Implement these focused helpers in `infer_python.py`:

```python
def emit_worker_event(name: str, payload: dict) -> None:
    print(f"{name} {json.dumps(payload, separators=(',', ':'))}", flush=True)


def release_worker_hardware(runtime: dict) -> None:
    robot = runtime.pop("robot", None)
    if robot is not None:
        robot.disconnect()
    runtime["hardware_connected"] = False


def activate_worker_task(runtime: dict, command: dict) -> None:
    task = str(command.get("task", "")).strip()
    if not task:
        raise ValueError("run_task requires task")
    reset_policy_task_state(runtime["policy"], runtime["language_cache"])
    runtime["task"] = task
    runtime["max_steps"] = max(1, int(command.get("max_steps", 1)))
    runtime["verification_image"] = str(command.get("verification_image", ""))
    runtime["verification_timeout"] = float(command.get("verification_timeout", 120.0))
```

Move policy/preprocessor loading before robot construction. In persistent mode, emit `MODEL_READY`, wait for `run_task`, connect hardware, emit `HARDWARE_READY`, and execute until `CYCLE_READY`. At the boundary, accept `continue`, `release_hardware`, or `shutdown`. Preserve the original one-shot CLI path for non-verified inference endpoints.

- [ ] **Step 7: Add and pass hardware-release and task-reset subprocess tests**

Test that `release_hardware` yields `HARDWARE_RELEASED` without process exit, then a second `run_task` yields `HARDWARE_READY` from the same PID and logs `TASK_RESET` for the new task.

Run: `pytest -q tests/test_vla_execution.py -k 'worker_'`

Expected: PASS.

- [ ] **Step 8: Commit the persistent worker runtime**

```bash
git add infer_python.py tests/test_vla_execution.py
git commit -m "feat: add persistent VLA worker protocol"
```

---

### Task 3: Controller-side preload and hardware handoff

**Files:**
- Modify: `vla_execution.py`
- Modify: `tests/test_vla_execution.py`

**Interfaces:**
- Consumes: worker commands/events defined in Task 2.
- Produces:
  - `ensure_model(command: list[str], model_id: str, timeout: float = 120.0) -> dict`
  - `run_task(task: str, max_steps: int, verification_image: str, verification_timeout: float = 120.0, timeout: float = 30.0) -> dict`
  - `release_hardware(timeout: float = 10.0) -> dict`
  - existing `continue_cycle() -> dict`, now using JSON input.
- `status()` adds `model_ready: bool`, `hardware_connected: bool`, and `active_task: str | None`.

- [ ] **Step 1: Write a failing fake-worker model-reuse test**

```python
def test_ensure_model_reuses_ready_process_for_same_model(tmp_path):
    manager = VlaProcessManager(max_log_lines=20)
    command = fake_persistent_worker_command(tmp_path)

    first = manager.ensure_model(command, "model_a", timeout=2)
    second = manager.ensure_model(command, "model_a", timeout=2)

    assert first["ok"] is True
    assert second["ok"] is True
    assert first["pid"] == second["pid"]
    assert manager.status()["model_ready"] is True
    manager.stop()
```

- [ ] **Step 2: Run it and verify failure because `ensure_model` is absent**

Run: `pytest -q tests/test_vla_execution.py::test_ensure_model_reuses_ready_process_for_same_model`

Expected: FAIL with missing method.

- [ ] **Step 3: Add event state and condition-based acknowledgement waits**

Add a `threading.Condition` sharing the manager lock. In `_reader`, parse protocol prefixes and update state:

```python
if name == "MODEL_READY":
    self._model_ready = True
    self._state = "model_ready"
elif name == "HARDWARE_READY":
    self._hardware_connected = True
    self._active_task = str(event.get("task", ""))
    self._state = "running"
elif name == "HARDWARE_RELEASED":
    self._hardware_connected = False
    self._active_task = None
    self._state = "model_ready"
self._condition.notify_all()
```

Implement a private `_wait_for(predicate, timeout, timeout_code, timeout_error)` using `time.monotonic()` and the condition; never busy-poll.

- [ ] **Step 4: Implement same-model reuse and controlled model switching**

```python
def ensure_model(self, command, model_id, timeout=120.0):
    current = self.status()
    if current["running"] and current["model_id"] == model_id and current["model_ready"]:
        return {"ok": True, "pid": current["pid"], "reused": True}
    if current["running"]:
        released = self.release_hardware()
        if not released["ok"]:
            return released
        self.stop()
    started = self.start(command, model_id, task=None)
    if not started["ok"]:
        return started
    return self._wait_for_model_ready(timeout)
```

Allow `start(..., task: str | None)` and reset all new state fields when starting or stopping.

- [ ] **Step 5: Add failing tests for run, release, timeout, and model switch**

Cover these observable behaviors with the fake worker:

- `run_task` waits for `HARDWARE_READY` and records the task.
- `release_hardware` waits for `HARDWARE_RELEASED` while PID remains unchanged.
- `continue_cycle` writes `{"command":"continue"}`.
- an unresponsive worker returns `WORKER_TIMEOUT`.
- `ensure_model` with `model_b` terminates the `model_a` PID and waits for a new ready PID.

- [ ] **Step 6: Implement JSON command writes and acknowledged transitions**

```python
def _send_command(self, payload: dict) -> dict:
    line = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
    # Resolve and validate the live stdin stream under the lock, then write outside it.


def release_hardware(self, timeout=10.0):
    state = self.status()
    if not state["hardware_connected"]:
        return {"ok": True, "pid": state["pid"], "already_released": True}
    sent = self._send_command({"command": "release_hardware"})
    if not sent["ok"]:
        return sent
    return self._wait_for_hardware_released(timeout)
```

Implement `run_task` with task, cycle limits, image path, and timeout fields. Make `stop()` send `shutdown` first and retain signal/kill fallback.

- [ ] **Step 7: Run all manager tests**

Run: `pytest -q tests/test_vla_execution.py`

Expected: PASS.

- [ ] **Step 8: Commit the controller lifecycle**

```bash
git add vla_execution.py tests/test_vla_execution.py
git commit -m "feat: manage warm VLA workers and hardware handoffs"
```

---

### Task 4: Verified execution preparation phase

**Files:**
- Modify: `verified_execution.py`
- Modify: `tests/test_verified_execution.py`

**Interfaces:**
- Consumes: `prepare(plan: dict, start_index: int) -> dict` callback supplied to `VerifiedExecutionManager.start`.
- Produces: preparation phases `loading_model` and `model_ready`; no step callback runs before preparation succeeds.

- [ ] **Step 1: Write the failing ordering test**

```python
def test_prepare_finishes_before_first_step_executes():
    events = []
    manager = VerifiedExecutionManager()
    manager.start(
        "pick", {"steps": [step()]}, 0, settings(),
        lambda plan, index: events.append("prepare") or {"ok": True},
        lambda current, actions, stop: events.append("execute") or {"ok": True},
        lambda current: {"ok": True, "status": "success"},
        lambda context: {"ok": False},
    )

    state = wait_terminal(manager)
    assert events == ["prepare", "execute"]
    assert state["state"] == "completed"
```

- [ ] **Step 2: Run it and verify the start signature/order fails**

Run: `pytest -q tests/test_verified_execution.py::test_prepare_finishes_before_first_step_executes`

Expected: FAIL because `start` has no preparation callback.

- [ ] **Step 3: Add preparation to the state machine**

Change `start` to accept `prepare` before `execute_step`, pass it to `_worker`, and begin `_worker` with:

```python
self._update(phase="loading_model")
prepared = prepare(self.status()["plan"], self.status()["current_step_index"])
if self._stop_requested():
    return
if not prepared.get("ok"):
    self._review(prepared.get("error", "VLA model preparation failed"))
    return
self._update(phase="model_ready")
```

Update every `VerifiedExecutionManager.start` call in `tests/test_verified_execution.py` to pass an immediate successful preparation callback, so the existing tests continue to express their original execution behavior.

- [ ] **Step 4: Add preparation failure and stop-during-load tests**

Verify that a failed preparation executes no steps and enters `needs_human_review`, while Stop during a blocked preparation leaves the session `stopped` and executes no step.

- [ ] **Step 5: Run all state-machine tests**

Run: `pytest -q tests/test_verified_execution.py`

Expected: PASS.

- [ ] **Step 6: Commit the preparation phase**

```bash
git add verified_execution.py tests/test_verified_execution.py
git commit -m "feat: preload VLA before verified execution"
```

---

### Task 5: FastAPI orchestration and exclusive hardware ownership

**Files:**
- Modify: `main.py` near `_verified_run_terminal`, `_verified_execute_step`, `_verified_completion`, `run_session_start`, and `run_stop`
- Modify: `vla_execution.py` command builder
- Modify: `tests/test_planner_api.py`

**Interfaces:**
- Consumes: Task 3 manager methods and Task 4 preparation callback.
- Produces:
  - `build_worker_command(...) -> list[str]`
  - `_verified_prepare(plan: dict, start_index: int) -> dict`
  - `_verified_release_vla_hardware() -> dict`
- Preserves `/api/run/session/start`, `/api/run/status`, and `/api/run/stop` response shapes while adding lifecycle phases.

- [ ] **Step 1: Write a failing API test for preload-before-IK**

Create a plan with IK step 1 and VLA step 2. Replace the manager and execution callbacks with fakes that append events:

```python
def test_verified_run_preloads_model_before_first_ik(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(main.vla_manager, "ensure_model", lambda command, model_id, timeout=120: events.append(("ready", model_id)) or {"ok": True})
    monkeypatch.setattr(main, "_verified_execute_step", lambda step, actions, stop: events.append(("step", step["method_id"])) or {"ok": True})
    # Start the session and wait for terminal state.
    assert events[0] == ("ready", "model_a")
    assert events[1] == ("step", "ik_reach_object_v1")
```

- [ ] **Step 2: Run it and verify the first IK currently runs without preload**

Run: `pytest -q tests/test_planner_api.py::test_verified_run_preloads_model_before_first_ik`

Expected: FAIL because `_verified_prepare` and `ensure_model` are not wired into the session.

- [ ] **Step 3: Split command construction into persistent and one-shot forms**

In `vla_execution.py`, add `build_worker_command` using model path, model ID, robot configuration, cameras, frame settings, device, and `--persistent-worker`. Do not include task or action-cycle limits; those belong to `run_task` commands. Retain `build_infer_command` for standalone inference endpoints.

- [ ] **Step 4: Implement `_verified_prepare`**

```python
def _verified_prepare(plan: dict, start_index: int) -> dict:
    first_vla = next(
        (step for step in plan.get("steps", [])[start_index:] if step.get("method_id") == "vla_model"),
        None,
    )
    if first_vla is None:
        return {"ok": True, "model_required": False}
    # Resolve the selected local model, build its persistent worker command,
    # and call vla_manager.ensure_model(...). Return its exact safe error.
```

Use the same model path traversal and camera validation already enforced by `_verified_execute_step`.

- [ ] **Step 5: Write failing hardware-handoff tests**

Cover:

- IK calls `release_hardware` before `connect_robot`.
- VLA calls `disconnect_robot` before `run_task`.
- completion `success` calls `release_hardware`, not `stop`.
- release timeout prevents IK movement and returns an error.
- two different tasks with the same model reuse the same manager PID.
- a re-plan with a different model invokes `ensure_model` for that model before its first replacement step.

- [ ] **Step 6: Replace process start/stop with task activation/release**

In `_verified_execute_step`:

```python
if step.get("method_id") == "ik_reach_object_v1":
    released = vla_manager.release_hardware(timeout=10.0)
    if not released.get("ok"):
        return released
    if not robot_state.get("connected"):
        connect_robot()
    # Execute existing IK interpolation and motion.

if step.get("method_id") == "vla_model":
    if robot_state.get("connected"):
        disconnect_robot()
    active = vla_manager.status()
    if active.get("model_id") != step.get("model_id") or not active.get("model_ready"):
        prepared = _verified_prepare({"steps": [step]}, 0)
        if not prepared.get("ok"):
            return prepared
    if active.get("state") == "waiting_for_verification" and active.get("active_task") == step.get("description"):
        return vla_manager.continue_cycle()
    return vla_manager.run_task(
        str(step.get("description", "")), actions_per_cycle,
        str(ROOT / "data" / "run_verification" / "latest.jpg"), 120.0,
    )
```

Keep the existing wait for the next `CYCLE_READY` counter after activation/continuation.

- [ ] **Step 7: Change success and terminal cleanup semantics**

On a successful completion verdict, call `vla_manager.release_hardware()` instead of `vla_manager.stop()`. Change `_verified_run_terminal` to inspect the verified manager's terminal state: `completed` releases hardware and operation ownership while leaving the ready worker alive; `stopped` and `needs_human_review` stop the worker before releasing ownership. `/api/run/stop` and server shutdown still call `vla_manager.stop()`.

- [ ] **Step 8: Run API, manager, and state-machine tests**

Run: `pytest -q tests/test_planner_api.py tests/test_vla_execution.py tests/test_verified_execution.py tests/test_verified_completion.py`

Expected: PASS.

- [ ] **Step 9: Commit orchestration and ownership changes**

```bash
git add main.py vla_execution.py tests/test_planner_api.py
git commit -m "feat: hand off hardware without unloading VLA"
```

---

### Task 6: Run lifecycle status UI

**Files:**
- Modify: `static/index.html` near `verifiedRunLabel`, `runBuildFlow`, and Run polling
- Modify: `tests/test_multi_model_ui.js`

**Interfaces:**
- Consumes: `/api/run/status` fields `phase`, `model_id`, `model_ready`, and `hardware_connected`.
- Produces: human-readable status labels without changing Run controls.

- [ ] **Step 1: Write failing UI label tests**

```javascript
assert.equal(verifiedRunLabel({phase:'loading_model', model_id:'model_a'}), 'Loading model model_a…');
assert.equal(verifiedRunLabel({phase:'model_ready', model_id:'model_a'}), 'Model model_a ready');
assert.equal(verifiedRunLabel({phase:'releasing_hardware'}), 'Releasing VLA hardware…');
assert.equal(verifiedRunLabel({phase:'connecting_vla_hardware'}), 'Connecting VLA hardware…');
```

- [ ] **Step 2: Run the UI test and verify unknown phases fail**

Run: `node tests/test_multi_model_ui.js`

Expected: FAIL because the new phase labels are missing.

- [ ] **Step 3: Add concise phase labels and surface readiness**

Extend `verifiedRunLabel` with exact mappings from Step 1. During polling, show the current lifecycle label in `runStatus`; keep per-step cycle/evidence rendering unchanged.

- [ ] **Step 4: Run the UI test**

Run: `node tests/test_multi_model_ui.js`

Expected: PASS.

- [ ] **Step 5: Commit UI status changes**

```bash
git add static/index.html tests/test_multi_model_ui.js
git commit -m "feat: show VLA preload and handoff status"
```

---

### Task 7: Full verification and operator handoff

**Files:**
- Verify all modified files from Tasks 1-6

**Interfaces:**
- Consumes: all preceding task outputs.
- Produces: a verified branch ready for physical robot smoke testing.

- [ ] **Step 1: Run whitespace and syntax checks**

Run: `git diff --check && python -m py_compile main.py planner_config.py infer_python.py vla_execution.py verified_execution.py`

Expected: exit 0 with no output.

- [ ] **Step 2: Run the complete Python suite**

Run: `pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 3: Run every JavaScript test file**

Run: `for test_file in tests/*.js; do node "$test_file" || exit 1; done`

Expected: exit 0 and no assertion failures.

- [ ] **Step 4: Inspect lifecycle diff and repository state**

Run: `git diff --stat HEAD~6..HEAD && git status --short`

Expected: only the files named by this plan are changed; no uncommitted implementation remains.

- [ ] **Step 5: Prepare physical smoke-test instructions**

Document these operator checks in the final handoff:

1. Start a Run whose plan begins with IK and confirm `Model ready` appears before arm motion.
2. Observe the worker PID before and after IK→VLA→IK→VLA; it must remain unchanged for the same model.
3. Confirm the serial port and cameras reconnect cleanly at both handoffs.
4. Run two different declared tasks on the same model and confirm the second task text appears without model reload.
5. Place the bow visibly in the green bowl and confirm completion returns `success` even while the arm remains above the bowl.

- [ ] **Step 6: Confirm the implementation history is complete**

Run: `git log --oneline -7`

Expected: the implementation commits from Tasks 1-6 appear after the approved design and plan commits. Do not create an empty verification commit.
