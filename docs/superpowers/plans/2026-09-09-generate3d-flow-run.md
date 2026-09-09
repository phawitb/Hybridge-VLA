# Generate 3D Flow Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a persistent, verified, sequential Flow Run subtab for long Generate 3D pick-and-place tasks.

**Architecture:** Add a focused `Generate3DFlowManager` that owns validated flow state, persistence, transitions, cancellation, and sequential orchestration while calling injected capture/detect/plan/execute/verify callbacks. Add thin FastAPI adapters around the existing Generate 3D and Gemini facilities, then add a Flow Run view to the existing monolithic frontend that renders server-owned progress and selected-block evidence.

**Tech Stack:** Python 3, FastAPI, threading, Gemini REST integration, OpenCV/Pillow image handling, vanilla HTML/CSS/JavaScript, Node assertion tests, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-generate3d-flow-run-design.md`

## Global Constraints

- Work only on branch `codex/generate3d-flow-run`; do not merge into `main`.
- Support 1–10 ordered atomic pick-and-place blocks only.
- A later block starts only after the previous block verifies as `success`.
- `failed`, `uncertain`, `stopped`, and `interrupted` always stop `Run All`.
- Every execution or retry captures and detects from a fresh frame.
- Never resume camera or robot work automatically after process restart.
- Pick Height defaults to `0 cm`; Place Height defaults to detected target height plus `2 cm`.
- `Use calibrated workspace safety` defaults unchecked.
- `Return to Rest Position on success` defaults checked.
- Gemini prompts are server-generated, displayed in full, and read-only.
- Existing 3D Simulation behavior must remain compatible.

---

### Task 1: Flow domain model, planning validation, and persistence

**Files:**
- Create: `generate3d_flow.py`
- Create: `tests/test_generate3d_flow.py`

**Interfaces:**
- Produces: `FlowValidationError`, `parse_flow_plan(raw: object) -> list[dict]`, and `Generate3DFlowManager(state_file: Path, artifact_root: Path)`.
- Produces manager methods `create_flow(...)`, `status()`, `start(scope, block_index, config, runner)`, `stop(stop_active=None)`, and `artifact_path(flow_id, filename)`.
- Persists schema version 1 through atomic temporary-file replacement.

- [ ] **Step 1: Write failing parser and flow-creation tests**

Cover valid ordered blocks, malformed JSON, unsupported/missing fields, empty plans, more than 10 blocks, unique block IDs, default status, and config defaults.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run --with pytest --with-requirements requirements.txt pytest -q tests/test_generate3d_flow.py`

Expected: collection/import failure because `generate3d_flow.py` does not exist.

- [ ] **Step 3: Implement the minimal parser and immutable status snapshots**

Use strict string validation and normalize only an unambiguous JSON object or existing Python dictionary. Create block records with phase `pending`, terminal result `null`, per-phase timestamps, empty artifacts/prompts/errors, and deterministic indices.

- [ ] **Step 4: Add failing persistence and restart tests**

Assert atomic JSON persistence after creation and each transition, restoration of ready/terminal state, active phase conversion to `interrupted`, no callback execution on initialization, and safe artifact-path rejection for traversal.

- [ ] **Step 5: Implement persistence, transition validation, and artifact resolution**

Define allowed state-machine edges exactly as the spec. Persist via a sibling temporary file followed by `Path.replace`. Resolve artifacts and require the resolved path to remain below `artifact_root / flow_id`.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run --with pytest --with-requirements requirements.txt pytest -q tests/test_generate3d_flow.py`

Commit: `feat: add persistent generate3d flow state manager`

### Task 2: Sequential runner and cancellation

**Files:**
- Modify: `generate3d_flow.py`
- Modify: `tests/test_generate3d_flow.py`

**Interfaces:**
- Consumes: manager and state types from Task 1.
- Produces runner callback contract `runner(block, config, transition, should_stop) -> dict`, where the returned dictionary contains terminal `status`, verification, artifacts, and optional error.

- [ ] **Step 1: Write failing execution tests**

Test that `all` executes blocks in order, preserves successful blocks, begins at the first non-success block, advances only on success, and stops on every unsuccessful terminal state. Test that `block` runs only the selected index.

- [ ] **Step 2: Verify RED**

Run the flow manager tests and confirm missing execution behavior is the failure.

- [ ] **Step 3: Implement background execution and transition callback**

Enforce one worker, freeze a per-block config snapshot, update active index/run scope, accept the runner's phase transitions, persist after each update, and clear activity in `finally`.

- [ ] **Step 4: Write failing cancellation tests**

Test idempotent stop, stop between blocks, propagation of `should_stop`, active physical stop callback, and terminal `stopped` state.

- [ ] **Step 5: Implement cooperative cancellation**

Use `threading.Event`; never launch the next block after it is set. Call the injected stop function only when a worker is active.

- [ ] **Step 6: Run focused tests and commit**

Run the flow manager tests.

Commit: `feat: orchestrate sequential generate3d flow blocks`

### Task 3: Gemini flow planning and verification services

**Files:**
- Modify: `main.py`
- Modify: `tests/test_generate3d_task_api.py`

**Interfaces:**
- Consumes: `parse_flow_plan` from Task 1 and existing Gemini request/JSON utilities.
- Produces: `_g3d_flow_planning_prompt(instruction)`, `_g3d_flow_verification_prompt(block)`, `_g3d_plan_flow(...)`, and `_g3d_verify_flow_block(...)`.

- [ ] **Step 1: Write failing prompt and parse tests**

Assert full source/target schema, pick/place-only restriction, 10-block limit, exact order, and absence of Gemini height recommendation fields.

- [ ] **Step 2: Verify RED**

Run the named new pytest cases and confirm the helpers are absent.

- [ ] **Step 3: Implement planning prompt and Gemini adapter**

Reuse configured Gemini credentials/model selection and raw-output capture. Return both validated blocks and the exact prompt/raw output record.

- [ ] **Step 4: Write failing verification tests**

Assert only `success`, `failed`, and `uncertain` are accepted; malformed output, API errors, and missing evidence map to `uncertain`.

- [ ] **Step 5: Implement verification adapter**

Include atomic instruction, names, and before/after images. Preserve complete prompt, raw output, status, reason, and evidence.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run --with pytest --with-requirements requirements.txt pytest -q tests/test_generate3d_task_api.py`

Commit: `feat: add Gemini flow planning and verification`

### Task 4: Flow block pipeline and shared Generate 3D integration

**Files:**
- Modify: `main.py`
- Modify: `generate3d_task.py` only if a task-manager wait/snapshot helper is required
- Modify: `tests/test_generate3d_task_api.py`
- Modify: `tests/test_generate3d_task.py` only if task-manager behavior changes

**Interfaces:**
- Produces: `_run_g3d_flow_block(block, config, transition, should_stop) -> dict`.
- Reuses existing camera capture, instruction-guided detection, calibration mapping, deterministic height computation, natural/waypoint planning, task manager, and rest-position movement.

- [ ] **Step 1: Write failing simulation pipeline tests**

Assert phase order, fresh capture/detection on every run, simulation-state verification, default heights, path/artifact storage, and block-only isolation.

- [ ] **Step 2: Verify RED**

Run the named new pipeline tests and confirm the runner does not exist.

- [ ] **Step 3: Implement simulation pipeline minimally**

Factor shared detection logic without changing its public response. Store audit images and plan metadata, launch the existing task manager, poll through a bounded wait, and derive terminal simulation verification from task status/simulated placement.

- [ ] **Step 4: Write failing real-mode safety tests**

Cover missing object, unavailable robot position, missing robot, hardware owner conflict, workspace safety opt-in, missing rest pose when return is enabled, execution timeout, and stop delegation.

- [ ] **Step 5: Implement real pipeline and preflight**

Acquire the existing robot operation lock exactly once for the physical block sequence, prevent direct Generate 3D overlap, execute through existing motion code, capture verification frame, call Gemini verification, and release ownership in `finally`.

- [ ] **Step 6: Add failing successful-flow rest test and implement it**

Assert automatic rest movement occurs once only after all blocks in `Run All` succeed, never after block-only, failed, uncertain, or stopped execution.

- [ ] **Step 7: Run focused regression and commit**

Run Generate 3D manager and API tests.

Commit: `feat: execute and verify generate3d flow blocks`

### Task 5: Flow REST API and startup recovery

**Files:**
- Modify: `main.py`
- Modify: `tests/test_generate3d_task_api.py`

**Interfaces:**
- Produces endpoints `/api/generate3d/flow/plan`, `/status`, `/start`, `/stop`, and `/artifact/{flow_id}/{filename}`.

- [ ] **Step 1: Write failing endpoint contract tests**

Test empty status, valid plan, request validation, refusal to replace active flow, all/block starts, invalid block index, idempotent stop, and status snapshots.

- [ ] **Step 2: Verify RED**

Run the new endpoint tests and confirm 404 responses.

- [ ] **Step 3: Implement thin FastAPI handlers**

Handlers validate request bodies, call the manager/services, convert domain errors to stable 400/409 responses, and never accept a client-provided system prompt.

- [ ] **Step 4: Write failing artifact and restart tests**

Test known artifacts, unknown files, path traversal, persisted flow restoration, and active-to-interrupted startup conversion.

- [ ] **Step 5: Implement controlled artifact responses and manager initialization**

Use FastAPI file responses only for manager-validated paths. Initialize the manager from the configured data paths without starting work.

- [ ] **Step 6: Run API tests and commit**

Run Generate 3D API tests.

Commit: `feat: expose generate3d flow run api`

### Task 6: Flow Run layout and reusable controls

**Files:**
- Modify: `static/index.html`
- Modify: `tests/test_generate3d_task_ui.js`

**Interfaces:**
- Produces DOM section `g3d-flow`, client state `G3D_FLOW`, subtab navigation, long-task prompt/image controls, flow execution controls, flowchart, block detail panel, diagnostics, and rest controls.

- [ ] **Step 1: Write failing static UI contract tests**

Assert the third subtab, required element IDs, read-only prompt area before Detect & Generate Flow, safety unchecked, return-to-rest checked, and all four execution buttons.

- [ ] **Step 2: Verify RED**

Run: `node tests/test_generate3d_task_ui.js`

Expected: assertions fail because Flow Run markup is absent.

- [ ] **Step 3: Add semantic Flow Run markup and responsive CSS**

Follow existing cards, status, inner-tabs, and form controls. Use text labels plus status colors, provide accessible button labels, and preserve the 3D Simulation DOM unchanged.

- [ ] **Step 4: Run UI test and commit**

Run the JS test.

Commit: `feat: add generate3d flow run workspace`

### Task 7: Flow frontend behavior and progress rendering

**Files:**
- Modify: `static/index.html`
- Modify: `tests/test_generate3d_task_ui.js`

**Interfaces:**
- Produces: `g3dFlowLoadPrompt`, `g3dFlowPlan`, `g3dFlowLoadStatus`, `g3dFlowRender`, `g3dFlowSelectBlock`, `g3dFlowStart`, `g3dFlowRetryBlock`, `g3dFlowStop`, and polling helpers.

- [ ] **Step 1: Write failing JavaScript behavior tests**

Assert endpoint usage, every state label, selected-block rendering, evidence/error/prompt/raw-output rendering, control enablement, terminal polling stop, and preservation of selected index across snapshots.

- [ ] **Step 2: Verify RED**

Run the JS test and confirm missing functions/behavior.

- [ ] **Step 3: Implement planning, status, and flowchart behavior**

Use server snapshots as truth, escape user/model text, create artifact URLs only from server-returned values, and update controls from activity/phase capabilities.

- [ ] **Step 4: Implement selected-block scene/path reuse**

Adapt existing object cards, editing, and Three.js preview through narrowly extracted shared render helpers. Do not share mutable `G3D` and `G3D_FLOW` state.

- [ ] **Step 5: Implement execution and polling controls**

Start all or one block with current config, retry explicitly, stop idempotently, show network errors without inventing state, and stop polling only when the server reports no active worker.

- [ ] **Step 6: Run UI tests and commit**

Run all JS tests.

Commit: `feat: render and control generate3d flow progress`

### Task 8: Regression, browser validation, and branch completion

**Files:**
- Modify tests or implementation files only for defects discovered by verification
- Update: `docs/superpowers/plans/2026-09-09-generate3d-flow-run.md` checkbox state

**Interfaces:**
- Produces a tested feature branch ready for user review, not merged into `main`.

- [ ] **Step 1: Run complete Python suite**

Run: `uv run --with pytest --with-requirements requirements.txt pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run complete JavaScript suite**

Run: `for test_file in tests/*.js; do node "$test_file" || exit 1; done`

Expected: every file exits zero.

- [ ] **Step 3: Run manual browser smoke verification**

Verify subtab navigation, responsive flowchart, prompt ordering, generated multi-block preview, selected-block details, block-only run, Run All success progression, failure stop, Stop Flow, refresh restoration, simulation path rendering, and unchanged 3D Simulation controls.

- [ ] **Step 4: Inspect git diff and persistence safety**

Confirm no runtime flow artifacts, credentials, uploaded frames, or unrelated user files are staged. Confirm `data/generate3d_flow_state.json` and `data/generate3d_flows/` remain runtime-only according to existing ignore policy or add precise ignore entries if needed.

- [ ] **Step 5: Commit final fixes and use branch-finishing workflow**

Commit any verification-only fixes with a focused message. Invoke `superpowers:verification-before-completion`, then `superpowers:requesting-code-review`, then `superpowers:finishing-a-development-branch`. Keep the result on `codex/generate3d-flow-run` unless the user separately requests merge or push.
