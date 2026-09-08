# Generate 3D Simulation Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users manually define Generate 3D objects and run one validated pick-and-place plan either as a 3D-only simulation or on the real robot with better convergence handling.

**Architecture:** Extract task waypoint construction from physical execution so both executors consume the same phase targets. Store manually drawn scenes in the existing detection state through a dedicated API, then reuse the current edit and task APIs. Keep browser animation driven by task-status joints in both modes; only the real executor reads or writes robot hardware.

**Tech Stack:** FastAPI, Python threads, NumPy/SciPy calibration helpers, vanilla JavaScript, Three.js, pytest, Node.js assertions.

**Spec:** `docs/superpowers/specs/2026-09-08-generate3d-simulation-mode-design.md`

## Global Constraints

- Change only Generate 3D > 3D Simulation; do not alter VLA or other tabs.
- `simulation` is the default and must never call robot read/write or acquire hardware ownership.
- Both modes use the same calibrated and safety-validated phase plan.
- Real mode retains workspace checks, joint limits, safe-height motion, cancellation, and exclusive hardware ownership.
- Manual scenes require a captured/uploaded image and computed Generate 3D calibration.

---

### Task 1: Manual object scene creation

**Files:**
- Modify: `main.py` near the Generate 3D detection-update endpoint
- Modify: `static/index.html` in the Generate 3D image editor
- Test: `tests/test_generate3d_task_api.py`
- Test: `tests/test_generate3d_task_ui.js`

**Interfaces:**
- Produces: `POST /api/generate3d/detection/manual` accepting `{image_size, objects}` and returning `{ok, detection_id, objects}`.
- Consumes: bbox validation, `_g3d_predict_from_pixel`, `_g3d_calibration_revision`, and `g3d_detection_state`.

- [ ] **Step 1: Write failing API tests**

Test that one valid object without an existing detection ID creates a scene, normalizes its bbox/center, predicts position/joints, assigns a detection ID, and records the calibration revision. Add rejection tests for invalid image size, duplicate/empty names, invalid bbox, and failed prediction.

- [ ] **Step 2: Verify RED**

```bash
/opt/miniconda3/envs/lerobot/bin/python -m pytest -q tests/test_generate3d_task_api.py -k manual
```

Expected: failure because the manual endpoint does not exist.

- [ ] **Step 3: Implement endpoint and editor flow**

Extract object normalization into a helper shared by manual creation and detection update. Store a fresh UUID, objects, image size, and calibration revision. Enable Add Object whenever a current image exists, allow pointer creation without a detection ID, call the manual endpoint for first sync, and remove Gemini raw output as a readiness requirement.

- [ ] **Step 4: Write failing UI tests**

With an image loaded and `detectionId: null`, assert Add Object is enabled, drawing works, first sync calls the manual endpoint, and the response updates the detection ID, overlays, and 3D scene.

- [ ] **Step 5: Verify GREEN and commit**

```bash
/opt/miniconda3/envs/lerobot/bin/python -m pytest -q tests/test_generate3d_task_api.py -k manual
node tests/test_generate3d_task_ui.js
git add main.py static/index.html tests/test_generate3d_task_api.py tests/test_generate3d_task_ui.js
git commit -m "Add manual Generate 3D object scenes"
```

---

### Task 2: Shared plan and simulation executor

**Files:**
- Modify: `main.py` around `_g3d_execute_pick_place` and task start
- Modify: `generate3d_task.py` task status
- Modify: `static/index.html` task controls and payload
- Test: `tests/test_generate3d_task_api.py`
- Test: `tests/test_generate3d_task_ui.js`

**Interfaces:**
- Produces: `_g3d_build_pick_place_plan(...) -> list[dict]`, each item containing `phase`, `joints`, and executor metadata.
- Produces: `_g3d_execute_simulation(plan, stop_event, publish) -> None`.
- Extends: task start with `execution_mode: "simulation" | "real"`; task status includes `execution_mode`.

- [ ] **Step 1: Write failing plan and simulation tests**

Assert the planner emits the ten existing ordered phases, validates all poses before execution, and keeps the gripper unchanged until opening. Start simulation with the robot disconnected and assert no ownership/read/write helper is called, status publishes plan joints, stop works, omitted mode defaults to simulation, and unknown modes are rejected.

- [ ] **Step 2: Verify RED**

```bash
/opt/miniconda3/envs/lerobot/bin/python -m pytest -q tests/test_generate3d_task_api.py -k 'plan or simulation or execution_mode'
```

Expected: failure because planning is embedded in the real executor and task start requires a robot.

- [ ] **Step 3: Implement shared planning and simulation**

Move prediction, preflight, and phase construction into `_g3d_build_pick_place_plan`. Interpolate and publish simulation joints while honoring `stop_event`. Apply connection, busy checks, ownership, and measured state only to real mode; use the first calibration point's six recorded joints as the simulation initial state.

- [ ] **Step 4: Write failing UI mode tests**

Assert both radio inputs exist, Simulation only is selected by default, payloads contain the selected mode, and simulated status joints update the Three.js robot.

- [ ] **Step 5: Implement radio controls and verify GREEN**

Add accessible radios named `g3dExecutionMode`, defaulting to `simulation`; add the selected mode to `g3dBuildTaskPayload()` and visible task status.

```bash
/opt/miniconda3/envs/lerobot/bin/python -m pytest -q tests/test_generate3d_task_api.py -k 'plan or simulation or execution_mode'
node tests/test_generate3d_task_ui.js
git add main.py generate3d_task.py static/index.html tests/test_generate3d_task_api.py tests/test_generate3d_task_ui.js
git commit -m "Add Generate 3D simulation execution mode"
```

---

### Task 3: Adaptive real-robot convergence

**Files:**
- Modify: `main.py` in `_g3d_task_move` and the real executor
- Test: `tests/test_generate3d_task_api.py`

**Interfaces:**
- Produces: `_g3d_waypoint_timeout(current, target, checked_names) -> float`, clamped to a documented minimum and maximum.
- Changes: timeout error lists phase and unconverged joint target, measured value, and residual.

- [ ] **Step 1: Write failing timeout tests**

Assert larger deltas receive longer bounded timeouts. Simulate a slow arm that exceeds the old two-second limit and completes. Simulate a stalled joint and assert the error identifies its name, target, measured value, and residual.

- [ ] **Step 2: Verify RED**

```bash
/opt/miniconda3/envs/lerobot/bin/python -m pytest -q tests/test_generate3d_task_api.py -k 'timeout or convergence'
```

- [ ] **Step 3: Implement adaptive convergence**

Derive timeout as `clamp(3.0 + max_delta_degrees / 10.0, 3.0, 8.0)` and calculate residuals after each measured update. Pass `ROBOT_JOINTS[:5]` as convergence names for `raising_to_safety`, `moving_to_source`, `descending_to_source`, `lifting_source`, `moving_to_target`, `placing`, and `lifting_after_release`, so an unchanged/clamped gripper cannot block arm motion. At deadline, raise a detailed error and never continue.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/opt/miniconda3/envs/lerobot/bin/python -m pytest -q tests/test_generate3d_task_api.py -k 'timeout or convergence'
git add main.py tests/test_generate3d_task_api.py
git commit -m "Improve Generate 3D robot convergence handling"
```

---

### Task 4: Integrated verification and delivery

**Files:**
- Verify: `main.py`, `generate3d_task.py`, `static/index.html`
- Verify: `tests/test_generate3d_task_api.py`, `tests/test_generate3d_task_ui.js`

**Interfaces:**
- Consumes: all interfaces from Tasks 1–3.
- Produces: verified commits on `v3`, pushed to `origin/v3`.

- [ ] **Step 1: Run full verification**

```bash
/opt/miniconda3/envs/lerobot/bin/python -m pytest -q
for test_file in tests/*.js; do node "$test_file" || exit 1; done
/opt/miniconda3/envs/lerobot/bin/python -m py_compile main.py generate3d_task.py
git diff --check
```

- [ ] **Step 2: Perform browser QA**

Confirm Add Object works after capture/upload and before detection, two manual objects render in 3D, Simulation only is preselected, Run Task animates without a robot, Use real robot is explicit, and other tabs are unchanged.

- [ ] **Step 3: Request review and resolve findings**

Review the complete diff against the pre-feature commit, fix every Critical and Important finding, and rerun step 1.

- [ ] **Step 4: Push**

```bash
git push origin v3
git status --short --branch
```

Expected: `v3` matches `origin/v3` with a clean working tree.
