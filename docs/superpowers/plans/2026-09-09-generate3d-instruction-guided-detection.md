# Generate 3D Instruction-Guided Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require a user instruction before Generate 3D detection, keep only found source/target objects relevant to that instruction, and apply Gemini's absolute pick/place height recommendations to the editable controls.

**Architecture:** Keep Gemini request construction and untrusted-response validation in `main.py`, while extending the existing pure task resolver in `generate3d_task.py` to prefer validated role tags and retain its legacy exact-name fallback. The browser owns instruction/image gating and stale-detection invalidation, renders partial results, applies independently valid height suggestions, and enables Run only for a complete role pair or a valid legacy pair.

**Tech Stack:** FastAPI multipart forms, Python 3, Pydantic/Gemini JSON schema, vanilla JavaScript/HTML, pytest, Node.js assertion tests.

**Spec:** `docs/superpowers/specs/2026-09-09-generate3d-instruction-guided-detection-design.md`

## Global Constraints

- Pick Height and Place Height are absolute centimeters from the calibrated floor.
- Keep initial defaults at Pick `0` cm and Place `5` cm.
- Clamp valid Gemini recommendations independently to `0–30` cm; omit invalid or missing values.
- Gemini may return zero, one, or two objects, but never more than one validated `source` and one validated `target`.
- Partial detection is successful and visible; it must not enable Run Task.
- Preserve legacy/manual exact-name resolution when no object has a role.
- Do not add multi-step planning, role-selection controls, retries, a second Gemini call, obstacle avoidance, or force-feedback height logic.
- Add no new runtime dependency.

---

### Task 1: Role-aware pick/place resolution

**Files:**
- Modify: `generate3d_task.py:21-45`
- Test: `tests/test_generate3d_task.py`

**Interfaces:**
- Consumes: `resolve_pick_place_objects(instruction: str, objects: list[dict])` and the existing `TaskResolutionError` contract.
- Produces: the same resolver signature, returning `(source: dict, target: dict)`; role-bearing collections require exactly one `task_role == "source"` and one `task_role == "target"`, while entirely role-free collections retain exact-name matching.

- [ ] **Step 1: Add failing tests for complete roles, incomplete roles, and legacy fallback**

```python
def test_resolve_pick_place_objects_prefers_roles_over_display_names():
    objects = [
        {"name": "item", "task_role": "source", "x": 1},
        {"name": "container", "task_role": "target", "x": 2},
    ]
    source, target = resolve_pick_place_objects("pick red cube into blue bowl", objects)
    assert source["x"] == 1
    assert target["x"] == 2


def test_resolve_pick_place_objects_rejects_incomplete_role_set():
    objects = [{"name": "item", "task_role": "source"}]
    with pytest.raises(TaskResolutionError) as exc_info:
        resolve_pick_place_objects("pick item into missing bowl", objects)
    assert exc_info.value.code == "OBJECT_MATCH_REQUIRED"
    assert "target" in str(exc_info.value).lower()


def test_resolve_pick_place_objects_keeps_legacy_exact_name_matching():
    objects = [{"name": "white star"}, {"name": "teal bowl"}]
    source, target = resolve_pick_place_objects("pick white star to teal bowl", objects)
    assert source["name"] == "white star"
    assert target["name"] == "teal bowl"
```

- [ ] **Step 2: Run the focused resolver tests and confirm the new role cases fail**

Run: `uv run --with pytest --with-requirements requirements.txt pytest tests/test_generate3d_task.py -q`

Expected: the role-preference/incomplete-role tests fail because the resolver currently relies only on exact names; existing tests continue to pass.

- [ ] **Step 3: Implement role-first resolution without changing the public signature**

```python
def resolve_pick_place_objects(instruction: str, objects: list[dict]) -> tuple[dict, dict]:
    role_objects = [obj for obj in objects if obj.get("task_role") in {"source", "target"}]
    if role_objects:
        sources = [obj for obj in role_objects if obj["task_role"] == "source"]
        targets = [obj for obj in role_objects if obj["task_role"] == "target"]
        missing = [role for role, matches in (("source", sources), ("target", targets)) if len(matches) != 1]
        if missing:
            raise TaskResolutionError(
                "OBJECT_MATCH_REQUIRED",
                f"Detected task objects are incomplete; missing or ambiguous: {', '.join(missing)}",
            )
        return copy.deepcopy(sources[0]), copy.deepcopy(targets[0])

    # Leave the existing exact-name parser and ambiguity checks below unchanged.
```

Ensure `deepcopy` and `TaskResolutionError` use the file's existing imports/types, and do not silently fall back when any role tag is present.

- [ ] **Step 4: Run all pure task tests**

Run: `uv run --with pytest --with-requirements requirements.txt pytest tests/test_generate3d_task.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the resolver change**

```bash
git add generate3d_task.py tests/test_generate3d_task.py
git commit -m "feat: resolve Generate 3D tasks by detected roles"
```

---

### Task 2: Validate instruction-guided Gemini detections and height recommendations

**Files:**
- Modify: `main.py:6265-6270`
- Modify: `main.py:7407-7515`
- Modify: `main.py` detection invalidation/manual-object rebuild helpers near the Generate 3D routes
- Test: `tests/test_generate3d_task_api.py`

**Interfaces:**
- Consumes: multipart `image: UploadFile`, required `instruction: str`, optional `model: str`; existing Gemini client, bbox normalization, position prediction, size normalization, and joint prediction helpers.
- Produces: `/api/generate3d/detect-image` JSON containing `objects: list[dict]`, optional `recommended_pick_height_cm: float`, optional `recommended_place_height_cm: float`; stored state gains `instruction: str`, and validated objects retain `task_role`.
- Produces: task start continues calling `resolve_pick_place_objects(instruction, objects)` from Task 1.

- [ ] **Step 1: Add an API test proving empty instruction is rejected before Gemini runs**

```python
def test_detect_image_requires_instruction_before_calling_gemini(client, monkeypatch):
    called = False

    async def forbidden_call(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Gemini must not run")

    monkeypatch.setattr(main, "_g3d_call_gemini", forbidden_call)
    response = client.post(
        "/api/generate3d/detect-image",
        files={"image": ("scene.png", PNG_BYTES, "image/png")},
        data={"instruction": "   "},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INSTRUCTION_REQUIRED"
    assert called is False
```

Adapt only the fixture/helper names (`PNG_BYTES`, Gemini call seam, error envelope) to those already used in this test module; keep the assertions and behavior exact.

- [ ] **Step 2: Add failing API tests for prompt boundaries and validated partial output**

```python
def test_detect_image_filters_roles_and_returns_partial_result(client, gemini_stub):
    gemini_stub.result = {
        "objects": [
            valid_candidate("cube", "source"),
            valid_candidate("lamp", "unrelated"),
            valid_candidate("second cube", "source"),
        ],
        "recommended_pick_height_cm": -4,
        "recommended_place_height_cm": 42,
    }
    response = post_detection(client, instruction="pick cube into bowl")
    assert response.status_code == 200
    payload = response.json()
    assert [(obj["name"], obj["task_role"]) for obj in payload["objects"]] == [("cube", "source")]
    assert payload["recommended_pick_height_cm"] == 0
    assert payload["recommended_place_height_cm"] == 30
    prompt = gemini_stub.last_prompt
    assert "<task_instruction>\npick cube into bowl\n</task_instruction>" in prompt
    assert "at most one source and one target" in prompt.lower()
```

Add companion cases that return `{objects: []}` successfully, omit `NaN`/string/missing recommendations, and skip a malformed source while retaining a valid target. Use the module's real Gemini stub and valid image fixture rather than introducing a second mocking style.

- [ ] **Step 3: Run the focused endpoint tests and confirm failure**

Run: `uv run --with pytest --with-requirements requirements.txt pytest tests/test_generate3d_task_api.py -q -k 'detect_image or role or recommendation'`

Expected: FAIL because `instruction`, role filtering, and recommendation fields are not implemented.

- [ ] **Step 4: Require and delimit the instruction in the detection endpoint**

```python
async def generate3d_detect_image(
    image: UploadFile = File(...),
    instruction: str = Form(""),
    model: str = Form(""),
):
    task_instruction = instruction.strip()
    if not task_instruction:
        raise HTTPException(
            status_code=400,
            detail={"code": "INSTRUCTION_REQUIRED", "message": "Enter a task instruction before detection."},
        )

    prompt = f"""Locate only the visible objects required to perform this task.
Return at most one source and one target. Omit a role if it cannot be located confidently.
Never return unrelated workspace objects. Heights are absolute centimeters from the calibrated floor.
Pick height is the gripper target height for grasping the source.
Place height is the gripper release height from the floor with no implicit clearance addition.
Return JSON only.
<task_instruction>
{task_instruction}
</task_instruction>"""
```

Update the Gemini response schema so every object requires `task_role` with enum `source|target`, and the root allows numeric `recommended_pick_height_cm` and `recommended_place_height_cm`. Remove the browser-controlled arbitrary prompt replacement path.

- [ ] **Step 5: Filter candidates independently and normalize recommendations**

```python
def _g3d_recommended_height(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return max(0.0, min(30.0, value))


seen_roles: set[str] = set()
objects: list[dict] = []
for idx, candidate in enumerate(parsed.get("objects", [])):
    if not isinstance(candidate, dict):
        continue
    role = candidate.get("task_role")
    if role not in {"source", "target"} or role in seen_roles:
        continue
    try:
        bbox = _g3d_normalize_bbox(candidate, img_w, img_h)
        x1, y1, x2, y2 = bbox
        center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        prediction = _g3d_predict_from_pixel(center, image_size=[img_w, img_h])
        if not prediction:
            continue
        size = candidate.get("estimated_size_cm") or [3, 3, 3]
        if not isinstance(size, list) or len(size) < 3:
            size = [3, 3, 3]
        size = [float(value) for value in size[:3]]
        if not all(math.isfinite(value) for value in size):
            raise ValueError
        confidence = float(candidate.get("confidence", 1.0))
        if not math.isfinite(confidence):
            raise ValueError
        obj = {
            "name": candidate.get("name") or f"object {idx + 1}",
            "task_role": role,
            "bbox": bbox,
            "bbox_raw": candidate.get("bbox_raw", []),
            "center_pixel": [round(center[0], 2), round(center[1], 2)],
            "position_3d": prediction["position_3d"],
            "predicted_joints": prediction["joints"],
            "color_hex": candidate.get("color_hex", "#888888"),
            "shape_3d": candidate.get("shape_3d", "box"),
            "estimated_size_cm": size,
            "confidence": max(0.0, min(1.0, confidence)),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        continue
    objects.append(obj)
    seen_roles.add(role)
```

Validate `color_hex` and `shape_3d` with the same fallbacks already used by `_g3d_rebuild_detection_objects`. Append only after every stage succeeds, and keep the first successfully validated object per role. Add recommendation keys to the response only when `_g3d_recommended_height` returns a number.

- [ ] **Step 6: Store instruction/roles and preserve roles through manual edits**

```python
g3d_detection_state.update({
    "id": detection_id,
    "objects": objects,
    "image_size": image_size,
    "calibration_revision": calibration_revision,
    "instruction": task_instruction,
})

# In the manual rebuild path, copy only a valid existing role.
role = existing_object.get("task_role")
if role in {"source", "target"}:
    rebuilt_object["task_role"] = role
```

Initialize and clear `instruction` alongside the other detection-state fields. Newly drawn objects receive no role. Confirm task start passes the active objects to Task 1's resolver so an incomplete role set returns `OBJECT_MATCH_REQUIRED` with the missing role in the message.

- [ ] **Step 7: Run the complete Generate 3D backend tests**

Run: `uv run --with pytest --with-requirements requirements.txt pytest tests/test_generate3d_task.py tests/test_generate3d_task_api.py -q`

Expected: PASS, including zero-object, partial-object, duplicate-role, recommendation, role-based start, incomplete-role, and legacy cases.

- [ ] **Step 8: Commit the backend behavior**

```bash
git add main.py tests/test_generate3d_task_api.py
git commit -m "feat: guide Generate 3D detection with task instructions"
```

---

### Task 3: Gate detection in the UI and display partial results

**Files:**
- Modify: `static/index.html:2288-2335`
- Modify: `static/index.html:12245-13020`
- Test: `tests/test_generate3d_task_ui.js`

**Interfaces:**
- Consumes: Task 2 response objects with optional `task_role`, `recommended_pick_height_cm`, and `recommended_place_height_cm`.
- Produces: multipart detection request with trimmed `instruction`; `g3dHasCompleteTaskObjects(objects)` and `g3dCanDetect()` UI predicates; stale detections are cleared when instruction text changes.
- Produces: `g3dResolveTaskObjects(instruction)` prefers a complete role pair and otherwise retains the legacy exact-name result only when no roles are present.

- [ ] **Step 1: Add failing markup and state-predicate tests**

```javascript
assert.ok(
  html.indexOf('id="g3dTaskInstruction"') < html.indexOf('id="g3dDetectBtn"'),
  'Task instruction must appear before Detect & Generate'
);

elements.g3dTaskInstruction.value = '   ';
context.G3D.sourceImageDataUrl = 'data:image/png;base64,abc';
assert.equal(context.g3dCanDetect(), false);
elements.g3dTaskInstruction.value = '  pick cube into bowl  ';
assert.equal(context.g3dCanDetect(), true);

context.G3D.objects = [{name: 'cube', task_role: 'source'}];
assert.equal(context.g3dCanRunTask(), false);
context.G3D.objects.push({name: 'bowl', task_role: 'target'});
assert.equal(context.g3dCanRunTask(), true);
```

Extend the test DOM with `g3dDetectBtn` and any status/object elements touched by the extracted functions. Add `g3dCanDetect` and `g3dHasCompleteTaskObjects` to the extraction list.

- [ ] **Step 2: Run the UI tests and confirm failure**

Run: `node tests/test_generate3d_task_ui.js`

Expected: FAIL because the instruction is below Detect, and the new predicates do not exist.

- [ ] **Step 3: Move the instruction and implement detection/run predicates**

```javascript
function g3dCanDetect() {
  const instruction = document.getElementById('g3dTaskInstruction').value.trim();
  return Boolean(G3D.sourceImageDataUrl && instruction && !G3D.sceneBusy);
}

function g3dHasCompleteTaskObjects(objects = G3D.objects) {
  const roleObjects = objects.filter(obj => obj.task_role === 'source' || obj.task_role === 'target');
  if (roleObjects.length) {
    return roleObjects.filter(obj => obj.task_role === 'source').length === 1
      && roleObjects.filter(obj => obj.task_role === 'target').length === 1;
  }
  return Boolean(g3dResolveTaskObjects(document.getElementById('g3dTaskInstruction').value));
}

function g3dCanRunTask() {
  return Boolean(G3D.detectionId && !G3D.sceneBusy && g3dHasCompleteTaskObjects());
}
```

Place the existing instruction input in the Top View Image card before the Detect button and remove its old duplicate from the hidden task-controls area. Make `g3dSetSceneBusy` set `g3dDetectBtn.disabled = !g3dCanDetect()`.

- [ ] **Step 4: Add failing request, preservation, recommendation, and invalidation tests**

```javascript
elements.g3dTaskInstruction.value = '  pick cube into bowl  ';
await context.g3dDetectObjects();
assert.equal(fakeFormData.get('instruction'), 'pick cube into bowl');
assert.equal(elements.g3dTaskInstruction.value, '  pick cube into bowl  ');
assert.equal(elements.g3dTaskPickHeight.value, '2.5');
assert.equal(elements.g3dTaskPlaceHeight.value, '8');

context.G3D.detectionId = 'det-1';
context.G3D.objects = [{task_role: 'source'}, {task_role: 'target'}];
elements.g3dTaskInstruction.dispatchEvent({type: 'input'});
assert.equal(context.G3D.detectionId, null);
assert.deepEqual(context.G3D.objects, []);
assert.equal(elements.g3dRunTaskBtn.disabled, true);
```

Use the test file's existing VM/event style. Stub `fetch` with a partial response in a second case and assert that its one object is retained/rendered while Run remains disabled. Add a response with no recommendation keys and assert pre-existing height values remain unchanged.

- [ ] **Step 5: Run the UI tests and confirm the new integration cases fail**

Run: `node tests/test_generate3d_task_ui.js`

Expected: FAIL because the request lacks `instruction`, detection overwrites/clears the instruction, recommendations are ignored, and edits do not invalidate stale state.

- [ ] **Step 6: Send the instruction, apply suggestions, and preserve partial objects**

```javascript
const instruction = document.getElementById('g3dTaskInstruction').value.trim();
if (!instruction) {
  g3dSetStatus('Enter a task instruction before detection.', 'error');
  return;
}
const form = new FormData();
form.append('image', imageBlob, 'scene.png');
form.append('instruction', instruction);

const result = await response.json();
G3D.objects = Array.isArray(result.objects) ? result.objects : [];
if (Number.isFinite(result.recommended_pick_height_cm)) {
  document.getElementById('g3dTaskPickHeight').value = String(result.recommended_pick_height_cm);
}
if (Number.isFinite(result.recommended_place_height_cm)) {
  document.getElementById('g3dTaskPlaceHeight').value = String(result.recommended_place_height_cm);
}
g3dSetTaskReady(g3dHasCompleteTaskObjects());
g3dPreviewTaskPath();
```

Do not call `g3dDefaultTaskInstruction`/`g3dUpdateDefaultInstruction` after detection, and remove instruction-clearing assignments from capture/upload handlers. Report status from returned roles: no objects = `No requested objects found`; source only = `Found source; target not found`; target only = inverse; complete pair = ready.

- [ ] **Step 7: Invalidate detection when the instruction changes**

```javascript
document.getElementById('g3dTaskInstruction').addEventListener('input', () => {
  G3D.instructionAuto = false;
  if (G3D.detectionId) {
    G3D.detectionId = null;
    G3D.objects = [];
    G3D.taskPath = [];
    g3dSetTaskReady(false);
    g3dRenderObjects();
    g3dRenderScene();
    g3dSetStatus('Instruction changed. Detect the scene again.', 'info');
  }
  g3dSetSceneBusy(G3D.sceneBusy);
});
```

Call the actual existing object/path render helpers by their current names. Update `g3dResolveTaskObjects` to return the unique source/target pair first; if any role exists but the pair is incomplete return `null`; only then run its current exact-name logic for role-free objects.

- [ ] **Step 8: Run the UI tests**

Run: `node tests/test_generate3d_task_ui.js`

Expected: PASS.

- [ ] **Step 9: Commit the frontend behavior**

```bash
git add static/index.html tests/test_generate3d_task_ui.js
git commit -m "feat: require instructions before Generate 3D detection"
```

---

### Task 4: End-to-end regression verification

**Files:**
- Verify: `generate3d_task.py`
- Verify: `main.py`
- Verify: `static/index.html`
- Verify: `tests/test_generate3d_task.py`
- Verify: `tests/test_generate3d_task_api.py`
- Verify: `tests/test_generate3d_task_ui.js`

**Interfaces:**
- Consumes: all interfaces produced by Tasks 1–3.
- Produces: a fully verified branch ready for local merge and push when requested.

- [ ] **Step 1: Run all JavaScript tests**

```bash
for test_file in tests/*.js; do node "$test_file" || exit 1; done
```

Expected: every JavaScript test exits successfully.

- [ ] **Step 2: Run the full Python suite**

Run: `uv run --with pytest --with-requirements requirements.txt pytest -q`

Expected: PASS with no failures.

- [ ] **Step 3: Review the final diff for scope and stale behavior**

Run: `git diff --check && git diff --stat HEAD~3..HEAD && git status --short`

Expected: no whitespace errors; changes are limited to the resolver, API/state, Generate 3D UI, and their tests; working tree is clean after commits.

- [ ] **Step 4: Perform a manual browser smoke test when the app is runnable locally**

Verify this exact sequence:

1. With no image or blank instruction, Detect & Generate is disabled.
2. After adding both, detection sends once and shows only returned source/target objects.
3. A one-role result remains visible and Run Task stays disabled.
4. Valid suggested heights replace `0`/`5`; omitted suggestions preserve current values.
5. Editing either height refreshes the path and remains user-editable.
6. Editing the instruction clears the detected objects/path and requires re-detection.
7. A complete role pair enables Run even when display names differ from instruction wording.
8. A role-free manually edited legacy pair still resolves by exact names.

- [ ] **Step 5: Commit any verification-only corrections, if the prior checks required changes**

```bash
git add generate3d_task.py main.py static/index.html tests/test_generate3d_task.py tests/test_generate3d_task_api.py tests/test_generate3d_task_ui.js
git commit -m "test: cover instruction-guided Generate 3D workflow"
```

Skip this commit when verification required no corrections; do not create an empty commit.
