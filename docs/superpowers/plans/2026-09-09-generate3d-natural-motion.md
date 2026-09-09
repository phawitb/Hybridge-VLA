# Generate 3D Natural Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add selectable Smooth Natural and Waypoint motion whose preview, simulation, and real-robot execution share the same safe trajectory rules.

**Architecture:** Preserve the existing waypoint planner and add a smooth planner that samples eased pixel/height trajectories, converts each sample through calibration, and validates the complete plan before execution. Extend plan steps with optional `trajectory` samples so real and simulated executors share one representation; the frontend renders the same deterministic curve semantics in workspace coordinates.

**Tech Stack:** FastAPI/Python, vanilla JavaScript, Three.js, pytest, Node.js assertions

**Spec:** `docs/superpowers/specs/2026-09-09-generate3d-natural-motion-design.md`

## Global Constraints

- `Smooth Natural` is the default; `Waypoint` remains a selectable fallback.
- API values are exactly `smooth` and `waypoint`.
- No hardware command is sent until every trajectory sample passes prediction and joint-limit validation.
- Transfer samples never fall below the effective transfer height; near-object motion remains vertical.
- Existing workspace enforcement, clearance, hardware ownership, stop behavior, and joint limits remain active.
- Loaded lift phases allow `shoulder_lift` residuals up to 4 degrees; other arm joints remain at 3 degrees and gripper at 2 degrees.
- Do not add obstacle avoidance, dynamic replanning, collision sensing, force control, or learned motion.

---

### Task 1: Finish scoped loaded-lift convergence

**Files:**
- Modify: `main.py:6822-7045`
- Test: `tests/test_generate3d_task_api.py:1025-1115`

**Interfaces:**
- Consumes: `_g3d_task_move(..., tolerance_deg, convergence_names)`
- Produces: `_g3d_task_move(..., joint_tolerances: dict[str, float] | None = None)` and `G3D_TASK_LOADED_LIFT_TOLERANCES`

- [ ] **Step 1: Keep failing boundary tests based on the reported residual**

```python
def test_lifting_source_accepts_loaded_shoulder_residual_up_to_four_degrees(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured["shoulder_lift"] = 16.04
    target = {**measured, "shoulder_lift": 12.50}
    # Freeze measurement and expect a lifting_source step with
    # joint_tolerances={"shoulder_lift": 4.0} to finish.

def test_lifting_source_rejects_loaded_shoulder_residual_over_four_degrees(monkeypatch):
    measured = {name: 0.0 for name in main.ROBOT_JOINTS}
    measured["shoulder_lift"] = 16.51
    target = {**measured, "shoulder_lift": 12.50}
    # Expect RuntimeError containing error=4.01.
```

- [ ] **Step 2: Verify the acceptance test fails before the fix**

Run:

```bash
uv run --with pytest --with-requirements requirements.txt pytest tests/test_generate3d_task_api.py -q -k loaded_shoulder_residual
```

Expected before implementation: the 3.54-degree case fails with the reported timeout.

- [ ] **Step 3: Apply per-joint tolerance only when configured**

```python
joint_tolerances = joint_tolerances or {}

def tolerance_for(name):
    return float(joint_tolerances.get(name, tolerance_deg))
```

Use `tolerance_for(name)` in convergence and residual reporting. Forward `step.get("joint_tolerances")` from the real executor. Attach `{"shoulder_lift": 4.0}` only to `lifting_source` and `lifting_after_release`.

- [ ] **Step 4: Run the convergence group**

```bash
uv run --with pytest --with-requirements requirements.txt pytest tests/test_generate3d_task_api.py -q -k "loaded_shoulder_residual or rejects_elbow_residual or strict_gripper_tolerance or task_move"
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_generate3d_task_api.py
git commit -m "Fix Generate 3D loaded lift convergence"
```

### Task 2: Add motion controls and preview curves

**Files:**
- Modify: `static/index.html:2310-2345,11945-12870`
- Test: `tests/test_generate3d_task_ui.js`

**Interfaces:**
- Produces: `g3dSelectedMotionMode(): "smooth" | "waypoint"`, mode-aware `g3dBuildTaskPathPoints(...)`, and payload field `motion_mode`

- [ ] **Step 1: Write failing control and payload tests**

```javascript
assert.equal(elements.g3dMotionSmooth.checked, true);
assert.equal(context.g3dBuildTaskPayload().motion_mode, 'smooth');
elements.g3dMotionSmooth.checked = false;
elements.g3dMotionWaypoint.checked = true;
assert.equal(context.g3dBuildTaskPayload().motion_mode, 'waypoint');
```

Assert both control IDs exist in HTML.

- [ ] **Step 2: Write failing shape tests with literal expectations**

```javascript
const smooth = context.g3dBuildTaskPathPoints(source, target, 5, 18, 'smooth');
assert.deepEqual(smooth[0], [0.12, 0.05, 0.08]);
assert.deepEqual(smooth.at(-1), [-0.09, 0.11, 0.14]);
assert.ok(smooth.length >= 20);
assert.ok(smooth.slice(1, 5).every(p => p[0] === 0.12 && p[2] === 0.08));
assert.ok(smooth.slice(-5, -1).every(p => p[0] === -0.09 && p[2] === 0.14));
assert.ok(smooth.slice(5, -5).every(p => p[1] >= 0.18));

const waypoint = context.g3dBuildTaskPathPoints(source, target, 5, 18, 'waypoint');
assert.deepEqual(waypoint, [
  [0.12, 0.05, 0.08], [0.12, 0.18, 0.08],
  [-0.09, 0.18, 0.14], [-0.09, 0.11, 0.14],
]);
```

- [ ] **Step 3: Verify UI tests fail**

```bash
node tests/test_generate3d_task_ui.js
```

Expected: failure for missing mode controls or mode-aware curve output.

- [ ] **Step 4: Implement controls and pure sampling helpers**

Add checked `g3dMotionSmooth` and unchecked `g3dMotionWaypoint` radios. Implement:

```javascript
function g3dSmoothstep(t) { return t * t * (3 - 2 * t); }

function g3dSampleSegment(from, to, count, minimumHeight=null) {
  return Array.from({length: count}, (_, index) => {
    const eased = g3dSmoothstep((index + 1) / count);
    const point = from.map((value, axis) => value + (to[axis] - value) * eased);
    if (minimumHeight !== null) point[1] = Math.max(point[1], minimumHeight);
    return point;
  });
}
```

Build Smooth Natural from vertical lift samples, a transfer curve with a modest crown, and vertical descent samples. Avoid duplicate segment anchors.

- [ ] **Step 5: Connect mode selection, payload, and refresh**

Pass the selected mode to the curve builder, add `motion_mode` to `g3dBuildTaskPayload`, and make both radio `change` events call `g3dPreviewTaskPath`.

- [ ] **Step 6: Run UI tests and commit**

```bash
node tests/test_generate3d_task_ui.js
node tests/test_camera_websocket.js
git add static/index.html tests/test_generate3d_task_ui.js
git commit -m "Add Generate 3D natural path preview"
```

Expected: both tests pass before commit.

### Task 3: Build and validate smooth backend plans

**Files:**
- Modify: `main.py:6890-7210`
- Test: `tests/test_generate3d_task_api.py`

**Interfaces:**
- Consumes: `_g3d_predict_from_pixel`, `_g3d_validate_joint_target`, existing plan inputs
- Produces: `_g3d_build_smooth_pick_place_plan(...) -> list[dict]`; movement steps may include `trajectory: list[dict[str, float]]`

- [ ] **Step 1: Write failing API mode tests**

Verify missing mode defaults to `smooth`, both supported values are accepted, and `diagonal` returns HTTP 400 with `INVALID_MOTION_MODE`. Patch hardware acquisition to raise in the invalid case, proving validation precedes hardware access.

- [ ] **Step 2: Write failing planner tests**

Patch prediction to derive joints from pixel/height inputs, then assert:

```python
plan = main._g3d_build_smooth_pick_place_plan(
    OBJECTS[0], OBJECTS[1], [800, 600], 5.0, 18.0,
    initial_joints, calibration={"model": "fixture"},
)
transfer = next(step for step in plan if step["phase"] == "moving_to_target")
assert len(transfer["trajectory"]) >= 8
assert all(sample["height_cm"] >= 18.0 for sample in transfer["path_samples"])
assert transfer["path_samples"][0]["pixel"] == OBJECTS[0]["center_pixel"]
assert transfer["path_samples"][-1]["pixel"] == OBJECTS[1]["center_pixel"]
```

Add a prediction that returns one out-of-limit joint and assert the builder raises before executor invocation.

- [ ] **Step 3: Verify planner tests fail**

```bash
uv run --with pytest --with-requirements requirements.txt pytest tests/test_generate3d_task_api.py -q -k "motion_mode or smooth_pick_place_plan"
```

Expected: missing validation and builder failures.

- [ ] **Step 4: Implement shared eased sampling**

```python
def _g3d_smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)

def _g3d_sample_motion(start_pixel, end_pixel, start_height, end_height,
                       count, *, minimum_height=None):
    samples = []
    for index in range(1, count + 1):
        eased = _g3d_smoothstep(index / count)
        pixel = [
            start_pixel[axis] + (end_pixel[axis] - start_pixel[axis]) * eased
            for axis in range(2)
        ]
        height = start_height + (end_height - start_height) * eased
        if minimum_height is not None:
            height = max(height, minimum_height)
        samples.append({"pixel": pixel, "height_cm": height})
    return samples
```

Hold pixels fixed for source lift and target descent. Interpolate pixels during transfer, add a non-negative smooth crown, and clamp to effective transfer height.

- [ ] **Step 5: Implement all-or-nothing prediction**

Generate all path samples first. Predict and validate every joint target before returning a plan. Wrap failures with `Could not calculate smooth <phase> trajectory`. Preserve correct gripper state for lift, transfer, descent, grasp, and release.

- [ ] **Step 6: Validate and route the API field**

At the beginning of task start:

```python
motion_mode = str(data.get("motion_mode", "smooth")).strip().lower()
if motion_mode not in {"smooth", "waypoint"}:
    return JSONResponse(status_code=400, content={
        "ok": False, "code": "INVALID_MOTION_MODE",
        "error": "Motion mode must be smooth or waypoint",
    })
```

Use the smooth builder for `smooth` and existing builder for `waypoint` in both execution modes.

- [ ] **Step 7: Run API tests and commit**

```bash
uv run --with pytest --with-requirements requirements.txt pytest tests/test_generate3d_task_api.py -q
git add main.py tests/test_generate3d_task_api.py
git commit -m "Plan safe Generate 3D natural trajectories"
```

Expected: all Generate 3D API tests pass before commit.

### Task 4: Stream trajectories in real and simulation execution

**Files:**
- Modify: `main.py:7010-7060`
- Test: `tests/test_generate3d_task_api.py`

**Interfaces:**
- Consumes: steps with optional `trajectory`, final `joints`, phase, convergence names, and tolerances
- Produces: real streaming with final-anchor convergence and simulation publication of the planned sequence

- [ ] **Step 1: Write failing real streaming tests**

Give the executor a three-sample trajectory. Record sends and assert the samples are sent in order, intermediate samples do not convergence-poll, the final anchor uses the existing convergence path, and every send uses owner `generate3d`. Add a stop event set after the first publication and assert later samples are not sent.

- [ ] **Step 2: Write failing simulation parity test**

Pass the same three-sample step to simulation and assert published `(phase, joints)` values exactly equal the planned trajectory after the initial `starting` event.

- [ ] **Step 3: Verify executor tests fail**

```bash
uv run --with pytest --with-requirements requirements.txt pytest tests/test_generate3d_task_api.py -q -k "stream_trajectory or simulation_publishes_planned_trajectory"
```

Expected: failures because existing executors ignore `trajectory`.

- [ ] **Step 4: Implement bounded real streaming**

For trajectory steps, send all intermediate samples at 0.04-second cadence with a stop check and measured-state publication for each. Use `_g3d_task_move` once on the final anchor with the step's convergence metadata. Keep waypoint and grip branches unchanged.

- [ ] **Step 5: Implement simulation parity**

For trajectory steps, publish each sample at simulation cadence with stop checks and set `current` to the final sample. Retain existing interpolation for waypoint steps.

- [ ] **Step 6: Run Generate 3D tests and commit**

```bash
uv run --with pytest --with-requirements requirements.txt pytest tests/test_generate3d_task.py tests/test_generate3d_task_api.py -q
node tests/test_generate3d_task_ui.js
git add main.py tests/test_generate3d_task_api.py
git commit -m "Execute Generate 3D natural trajectories"
```

Expected: all selected tests pass before commit.

### Task 5: Full verification and handoff

**Files:**
- Verify: `main.py`
- Verify: `static/index.html`
- Verify: `tests/test_generate3d_task_api.py`
- Verify: `tests/test_generate3d_task_ui.js`

**Interfaces:**
- Consumes: completed frontend, planner, and executor
- Produces: fresh full-suite and patch-integrity evidence

- [ ] **Step 1: Run every JavaScript test**

```bash
for test_file in tests/*.js; do node "$test_file" || exit 1; done
```

Expected: every file exits 0.

- [ ] **Step 2: Run the full Python suite**

```bash
uv run --with pytest --with-requirements requirements.txt pytest -q
```

Expected: zero failures; record dependency warnings separately.

- [ ] **Step 3: Check patch integrity and scope**

```bash
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors and only intended implementation/test files plus known pre-existing user changes.

- [ ] **Step 4: Check requirements against the spec**

Confirm from code and tests that Smooth Natural is default, Waypoint is unchanged, preview/backend curve semantics agree, transfer height is bounded, invalid plans issue no hardware writes, stop checks remain active, and tolerance exceptions remain phase-scoped.

- [ ] **Step 5: Commit final test-only corrections if any**

```bash
git add tests/test_generate3d_task_api.py tests/test_generate3d_task_ui.js
git commit -m "Test Generate 3D motion modes end to end"
```
