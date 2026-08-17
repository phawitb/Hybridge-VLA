# Run Plan Invalid-Format Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Run Plan recover from legacy schema-less prompts and report a specific error when Gemini still returns a response without a top-level `steps` array.

**Architecture:** Keep prompt compatibility logic in `planner_config.py`, where saved planner settings are normalized. Keep inference failure reporting in `main.py`, preserving the existing response contract's `error` field so the Run UI displays it without a new client protocol. Add focused Python regression tests and a static UI contract assertion.

**Tech Stack:** Python 3.12, FastAPI, pytest, JavaScript, Node.js assertions

## Global Constraints

- Do not overwrite or reserialize the user's `config.yaml`; legacy migration is read-time only.
- Preserve valid custom templates that explicitly specify the required top-level `steps` array.
- Preserve successful inference responses, exact-task validation, IK validation, and verification behavior.
- Use test-driven development: every production behavior change must first have a test that fails for the expected reason.

---

### Task 1: Read-Time Legacy Prompt Fallback

**Files:**
- Modify: `tests/test_planner_config.py`
- Modify: `planner_config.py:54-72`

**Interfaces:**
- Consumes: `planner_settings(config: dict) -> dict`
- Produces: `_prompt_with_schema(value: object, fallback: str) -> str`, returning the custom template only when it contains `{instruction}`, `{available_models}`, and an explicit `steps` schema marker; otherwise returning the built-in mode-specific fallback.

- [ ] **Step 1: Write failing tests for legacy fallback and custom preservation**

```python
def test_planner_settings_replaces_saved_template_without_steps_schema():
    settings = planner_settings({
        "planner": {
            "use_ik": False,
            "prompt_templates": {
                "no_ik": "Instruction {instruction}; models {available_models}; each step has fields"
            },
        }
    })
    assert settings["prompt_templates"]["no_ik"] == DEFAULT_NO_IK_PROMPT


def test_planner_settings_preserves_schema_complete_custom_template():
    custom = 'Input {instruction}; models {available_models}; output {"steps": []}'
    settings = planner_settings({
        "planner": {"prompt_templates": {"no_ik": custom}}
    })
    assert settings["prompt_templates"]["no_ik"] == custom
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `/opt/miniconda3/envs/lerobot/bin/python -m pytest tests/test_planner_config.py -v`

Expected: the legacy-template assertion fails because the current implementation preserves any non-empty custom string.

- [ ] **Step 3: Implement the minimal read-time normalization**

```python
def _prompt_with_schema(value: object, fallback: str) -> str:
    prompt = str(value or "")
    required = ("{instruction}", "{available_models}", '"steps"')
    return prompt if all(marker in prompt for marker in required) else fallback
```

Use the helper for both `use_ik` and `no_ik` values returned by `planner_settings`.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run: `/opt/miniconda3/envs/lerobot/bin/python -m pytest tests/test_planner_config.py -v`

Expected: all planner configuration tests pass.

- [ ] **Step 5: Commit the isolated behavior**

```bash
git add planner_config.py tests/test_planner_config.py
git commit -m "Fallback from legacy planner prompts"
```

### Task 2: Structured Invalid Plan Format API Error

**Files:**
- Modify: `tests/test_planner_api.py`
- Modify: `main.py:853-1034`

**Interfaces:**
- Consumes: the existing inference loop's `plan_data`, `raw_text`, `verify_info`, and `total_elapsed` values.
- Produces: an HTTP 200 JSON response with `error_code: "INVALID_PLAN_FORMAT"`, a user-facing `error`, `raw_response`, `verify`, and `elapsed` after retries exhaust without a valid `steps` array.

- [ ] **Step 1: Write a failing endpoint regression test**

```python
def test_infer_reports_invalid_plan_format_after_all_attempts(monkeypatch, tmp_path):
    setup_infer(monkeypatch, tmp_path, '{"step":{"description":"pick up the bow"}}')
    client = TestClient(main.app)
    response = client.post(
        "/api/infer",
        data={"model": "gemini-test", "instruction": "pick up the bow"},
        files={"image": ("capture.jpg", image_bytes(), "image/jpeg")},
    )
    body = response.json()
    assert body["error_code"] == "INVALID_PLAN_FORMAT"
    assert "steps" in body["error"]
    assert body["raw_response"] == '{"step":{"description":"pick up the bow"}}'
    assert body["plan"] is None
```

- [ ] **Step 2: Run the focused endpoint test and confirm RED**

Run: `/opt/miniconda3/envs/lerobot/bin/python -m pytest tests/test_planner_api.py::test_infer_reports_invalid_plan_format_after_all_attempts -v`

Expected: FAIL because `error_code` is absent.

- [ ] **Step 3: Add the minimal post-loop error response**

Insert before capability-validation error handling:

```python
if not isinstance(plan_data, dict) or not isinstance(plan_data.get("steps"), list):
    return {
        "error": "Planner returned an invalid format: expected a top-level steps array.",
        "error_code": "INVALID_PLAN_FORMAT",
        "plan": None,
        "raw_response": raw_text,
        "verify": verify_info,
        "elapsed": round(total_elapsed, 2),
    }
```

- [ ] **Step 4: Run planner API tests and confirm GREEN**

Run: `/opt/miniconda3/envs/lerobot/bin/python -m pytest tests/test_planner_api.py -v`

Expected: all planner API tests pass.

- [ ] **Step 5: Commit the API behavior**

```bash
git add main.py tests/test_planner_api.py
git commit -m "Report invalid planner response formats"
```

### Task 3: UI Contract and Full Verification

**Files:**
- Modify: `tests/test_multi_model_ui.js`
- Verify: `static/index.html`

**Interfaces:**
- Consumes: `/api/infer` response field `error`.
- Produces: regression coverage that `runPlan()` checks `d.error` before the defensive missing-plan fallback.

- [ ] **Step 1: Add a static contract assertion**

```javascript
const runPlanSource = extractFunction('runPlan');
assert.ok(runPlanSource.indexOf('if (d.error)') < runPlanSource.indexOf('if (!d.plan || !d.plan.steps)'));
assert.match(runPlanSource, /throw new Error\(d\.error\)/);
```

- [ ] **Step 2: Run the UI test**

Run: `node tests/test_multi_model_ui.js`

Expected: PASS because the UI already honors the backend error; this test locks the required ordering. No production UI change is needed unless the assertion exposes a mismatch.

- [ ] **Step 3: Run full verification**

```bash
/opt/miniconda3/envs/lerobot/bin/python -m pytest tests/test_model_registry.py tests/test_planner_config.py tests/test_planner_api.py tests/test_vla_execution.py -v
node tests/test_multi_model_ui.js
node tests/test_camera_websocket.js
python3 -m py_compile main.py model_registry.py planner_config.py vla_execution.py infer_python.py
perl -0777 -ne 'while (/<script(?:\s[^>]*)?>(.*?)<\/script>/sg) { print "$1\n" }' static/index.html > /tmp/hybridge-inline-final.js
node --check /tmp/hybridge-inline-final.js
python3 -m json.tool fakecam_params.json >/dev/null
git diff --check
```

Expected: all Python and Node tests pass, compilation and syntax checks exit zero, and `git diff --check` reports no errors.

- [ ] **Step 4: Run an actual Gemini smoke test through `/api/infer`**

Submit `pick up the bow to the green bowl` with a generated JPEG through FastAPI `TestClient`. Expected: HTTP 200, a non-null `plan.steps` array, and no `error_code`.

- [ ] **Step 5: Commit the verification contract**

```bash
git add tests/test_multi_model_ui.js
git commit -m "Verify Run Plan error display contract"
```
