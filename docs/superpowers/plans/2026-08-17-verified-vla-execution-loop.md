# Verified VLA Execution Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute each VLA task in configurable bounded action cycles, verify completion from a fresh camera image with Gemini, and re-plan only unfinished work after the configured cycle limit.

**Architecture:** Add a framework-independent execution state machine in `verified_execution.py`. It owns session state and sequencing but receives injected adapters for robot execution, image verification, and re-planning. `main.py` supplies hardware and Gemini adapters and exposes start/status/stop APIs; `static/index.html` supplies the full plan and renders server-owned state. Configuration normalization stays in `planner_config.py`.

**Tech Stack:** Python 3.12, FastAPI, threading, httpx, OpenCV, pytest, browser JavaScript, Node.js assertions

## Global Constraints

- Defaults are 100 actions per cycle, 5 cycles before re-plan, and 3 automatic re-plans.
- Allowed ranges are actions `1–1000`, cycles `1–20`, and re-plans `1–10`.
- A session snapshots settings at start.
- Stop is checked before execution, verification, retry, and re-plan transitions.
- No inference process runs without an action bound.
- Re-planning includes completed steps and forbids repeating them.
- Invalid Gemini responses, camera errors, process errors, and exhausted re-plans end in `needs_human_review`.
- Preserve the user's unrelated `config.yaml` edits.

---

### Task 1: Execution Loop Configuration

**Files:**
- Modify: `planner_config.py`
- Modify: `main.py`
- Modify: `tests/test_planner_config.py`
- Modify: `tests/test_planner_api.py`

**Interfaces:**
- Produces: `execution_loop_settings(config: dict) -> dict` with integer keys `actions_per_cycle`, `cycles_before_replan`, and `max_replans`.
- Extends: `GET /api/config` response with `execution_loop`.
- Extends: config save payload validation and persistence for `execution_loop`.

- [ ] Write tests for defaults, valid values, lower/upper range rejection, API read, and API persistence.
- [ ] Run focused tests and confirm failures are caused by missing configuration behavior.
- [ ] Implement normalization constants and API validation without writing defaults during reads.
- [ ] Run focused tests until green.
- [ ] Commit with `Add verified execution loop settings`.

### Task 2: Framework-Independent Execution State Machine

**Files:**
- Create: `verified_execution.py`
- Create: `tests/test_verified_execution.py`

**Interfaces:**
- Produces: `VerifiedExecutionManager.start(original_instruction, plan, start_index, settings, execute_step, verify_step, replan) -> dict`.
- Produces: `VerifiedExecutionManager.status() -> dict` and `stop() -> dict`.
- Adapter `execute_step(step, actions_per_cycle, stop_event) -> dict` returns `{ok, error?}`.
- Adapter `verify_step(step) -> dict` returns `{status: success|continue|uncertain, reason, visible_evidence}`.
- Adapter `replan(context: dict) -> dict` returns `{ok, plan?, error?}`.

- [ ] Write deterministic tests for success advancement, continue/uncertain retries, configurable cycle threshold, completed-step re-plan context, replacement-plan adoption, re-plan limit, Stop, execution failure, verification failure, and invalid replacement.
- [ ] Run the new test file and confirm RED.
- [ ] Implement a locked state snapshot, daemon worker, stop event, and explicit transition methods with no FastAPI/hardware imports.
- [ ] Run the new tests until green and refactor duplicated state updates.
- [ ] Commit with `Add verified VLA execution state machine`.

### Task 3: Hardware and Gemini Adapters

**Files:**
- Modify: `main.py`
- Modify: `vla_execution.py`
- Modify: `tests/test_planner_api.py`
- Modify: `tests/test_vla_execution.py`

**Interfaces:**
- `POST /api/run/session/start` consumes `original_instruction`, `plan`, and `start_index`.
- `GET /api/run/status` returns manager state including phase, cycle, limits, active plan, completed steps, verification history, and re-plan count.
- `POST /api/run/stop` cancels both the session and active subprocess.
- Completion Gemini schema: `{"status":"success|continue|uncertain","reason":"...","visible_evidence":"..."}`.

- [ ] Write failing API tests using injected fake execute/verify/re-plan adapters.
- [ ] Write failing tests proving VLA commands always use the snapshotted action count and Stop terminates the subprocess.
- [ ] Implement bounded subprocess execution and polling adapter.
- [ ] Implement fresh configured-camera capture after the subprocess releases hardware.
- [ ] Implement conservative Gemini verification: invalid schema becomes `uncertain`; API/camera failure becomes adapter failure.
- [ ] Implement Gemini re-plan prompt with original instruction, completed steps, failed-step history, selected model tasks, IK mode, and local `validate_plan` enforcement.
- [ ] Run focused API/process/state-machine tests until green.
- [ ] Commit with `Connect verified execution to robot and Gemini`.

### Task 4: Config & Test and Run UI

**Files:**
- Modify: `static/index.html`
- Modify: `tests/test_multi_model_ui.js`

**Interfaces:**
- Config payload includes `execution_loop` numeric settings.
- Run start payload includes original instruction, current plan, and selected index.
- Status renderer consumes `phase`, `cycle`, `cycles_before_replan`, `actions_per_cycle`, `plan`, `completed_steps`, `replan_count`, `state`, and `error`.

- [ ] Add failing Node assertions for three config controls, payload parsing, Run session payload, phase labels, replacement-plan rendering, completed-step rendering, automatic advancement, review state, and Stop endpoint usage.
- [ ] Run `node tests/test_multi_model_ui.js` and confirm RED.
- [ ] Add the Execution Loop field group with defaults/ranges and config load/save behavior.
- [ ] Replace client-side completion-on-process-exit with server session start/poll/render behavior.
- [ ] Keep log display and make Stop cancel the session.
- [ ] Run Node tests and inline-script syntax check until green.
- [ ] Commit with `Add verified execution controls and status UI`.

### Task 5: Full Verification and Integration

**Files:**
- Verify all changed production/test files.

**Interfaces:**
- No new interfaces; verifies the complete design contract.

- [ ] Run Python suites: model registry, planner config/API, VLA execution, and verified execution.
- [ ] Run multi-model UI and Camera WebSocket Node tests.
- [ ] Run Python compilation, JavaScript syntax, JSON parsing, and `git diff --check`.
- [ ] Run a no-hardware end-to-end test with fake execution, image, Gemini verification, and re-plan adapters.
- [ ] Run a real Gemini completion-classification smoke test on a generated image without moving hardware.
- [ ] Review changed files against every design requirement and fix any gap with a failing test first.
- [ ] Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch` to integrate and clean the worktree.
