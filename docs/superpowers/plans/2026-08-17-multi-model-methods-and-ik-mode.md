# Multi-Model Methods and IK Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let operators select multiple task-aware local VLA checkpoints, switch between IK-assisted and VLA-only planning prompts, and execute the checkpoint selected for each VLA plan step.

**Architecture:** Add focused Python modules for model discovery, planner configuration/validation, and VLA process lifecycle while keeping FastAPI route wiring in `main.py`. The browser consumes a stable model-registry API, saves model IDs plus two prompt templates, and sends explicit `method_id`/`model_id` fields through planning and Run Step. Existing `infer_python.py` remains the policy runtime and gains bounded-step execution so a VLA plan step can complete without manual process cleanup.

**Tech Stack:** Python 3, FastAPI, PyYAML, PyArrow, pytest, vanilla JavaScript, Node.js `assert`/`vm`, LeRobot CLI/runtime.

## Global Constraints

- Existing configurations migrate to `planner.use_ik: true`.
- A selected model must be downloaded, have a supported policy type, expose non-empty training tasks, and have satisfiable camera inputs.
- Training tasks come from the dataset referenced by `train_config.json`; model-name inference is fallback-only.
- The client sends model IDs, never checkpoint paths; the server resolves and validates all filesystem paths.
- Each VLA step uses `method_id: "vla_model"` and an explicit `model_id`; IK steps use `method_id: "ik_reach_object_v1"` and `model_id: null`.
- Not use IK mode rejects every IK step.
- Use IK mode requires an IK step immediately before each VLA step.
- No automatic checkpoint fallback, ensemble execution, or capability inference from model names.
- Preserve the uncommitted Camera & Teleop WebSocket fix in `static/index.html` and `tests/test_camera_websocket.js`; do not overwrite or fold it into feature commits.

---

## File Map

- Create `model_registry.py`: local/remote model normalization, training-dataset resolution, task extraction, camera compatibility, and selectability.
- Create `planner_config.py`: defaults, legacy migration, prompt rendering, and returned-plan validation.
- Create `vla_execution.py`: safe command construction and process lifecycle independent of FastAPI.
- Modify `infer_python.py`: bounded `--max-steps` execution and non-zero exit on inference failure.
- Modify `main.py`: API integration only; load modules, expose routes, validate config/infer/run requests, and coordinate hardware ownership.
- Modify `config.yaml`: persisted planner mode, selected model IDs, and complete default prompts.
- Modify `static/index.html`: model registry UI, IK radio controls, dual prompt editor, plan schema rendering, and VLA step lifecycle.
- Create `tests/test_model_registry.py`: registry unit tests with temporary model/dataset fixtures.
- Create `tests/test_planner_config.py`: migration, prompt rendering, and plan validation tests.
- Create `tests/test_vla_execution.py`: command, lifecycle, and termination tests using a real harmless subprocess.
- Create `tests/test_multi_model_ui.js`: browser-function behavior tests using Node.js and minimal DOM fixtures.
- Modify `requirements.txt`: add `pyarrow`, which production registry task extraction imports.

---

### Task 1: Build the Local Model Capability Registry

**Files:**
- Create: `model_registry.py`
- Create: `tests/test_model_registry.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `load_local_models(root: Path, configured_cameras: set[str]) -> list[dict]`
- Produces: `load_remote_models(models: list[dict], dataset_ids: list[str], task_loader: Callable[[str], list[str]]) -> list[dict]`
- Produces: `merge_model_records(local: list[dict], remote: list[dict]) -> list[dict]`
- Produces: `get_model_record(records: list[dict], model_id: str) -> dict | None`
- Record keys: `id`, `repo_id`, `display_name`, `policy_type`, `downloaded`, `local_path`, `dataset_id`, `tasks`, `camera_features`, `selectable`, `unavailable_reason`

- [ ] **Step 1: Write fixture-based failing tests for task and camera discovery**

```python
def test_load_local_models_reads_exact_training_tasks(tmp_path):
    root = make_model_fixture(
        tmp_path,
        model_name="smolvla_demo",
        policy_type="smolvla",
        dataset_name="demo_ds",
        tasks=["pick up the bow", "place the bow in the bowl"],
        cameras=["observation.images.top", "observation.images.wrist"],
    )
    [record] = load_local_models(root, {"top", "wrist"})
    assert record["tasks"] == ["pick up the bow", "place the bow in the bowl"]
    assert record["camera_features"] == [
        "observation.images.top",
        "observation.images.wrist",
    ]
    assert record["selectable"] is True


def test_load_local_models_disables_missing_task_metadata(tmp_path):
    root = make_model_fixture_without_tasks(tmp_path, "act_unknown", "act")
    [record] = load_local_models(root, {"top", "wrist"})
    assert record["selectable"] is False
    assert record["unavailable_reason"] == "Training instructions unavailable"


def test_load_local_models_disables_missing_camera(tmp_path):
    root = make_model_fixture(
        tmp_path,
        model_name="smolvla_two_cam",
        policy_type="smolvla",
        dataset_name="demo_ds",
        tasks=["pick up the bow"],
        cameras=["observation.images.top", "observation.images.wrist"],
    )
    [record] = load_local_models(root, {"top"})
    assert record["selectable"] is False
    assert record["unavailable_reason"] == "Missing configured cameras: wrist"
```

- [ ] **Step 2: Run registry tests and verify RED**

Run: `python3 -m pytest tests/test_model_registry.py -v`

Expected: FAIL during import with `ModuleNotFoundError: No module named 'model_registry'`.

- [ ] **Step 3: Implement local registry parsing with resolved-path containment**

```python
SUPPORTED_POLICY_TYPES = {"smolvla", "act", "pi0", "pi05"}


def load_local_models(root: Path, configured_cameras: set[str]) -> list[dict]:
    root = root.resolve()
    records = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        train_path = model_dir / "train_config.json"
        config_path = model_dir / "config.json"
        if not train_path.exists() or not config_path.exists():
            continue
        train = json.loads(train_path.read_text())
        policy = json.loads(config_path.read_text())
        dataset = train.get("dataset", {})
        tasks = _read_dataset_tasks(root.parent, dataset)
        camera_features = sorted(
            key for key in policy.get("input_features", {})
            if key.startswith("observation.images.")
        )
        camera_names = {key.rsplit(".", 1)[-1] for key in camera_features}
        missing = sorted(camera_names - configured_cameras)
        policy_type = str(policy.get("type", "unknown"))
        reason = None
        if policy_type not in SUPPORTED_POLICY_TYPES:
            reason = f"Unsupported policy type: {policy_type}"
        elif not tasks:
            reason = "Training instructions unavailable"
        elif missing:
            reason = f"Missing configured cameras: {', '.join(missing)}"
        resolved = model_dir.resolve()
        if root not in resolved.parents:
            continue
        records.append(_record_from_local(model_dir, dataset, policy_type, tasks, camera_features, reason))
    return records
```

Use `table = pyarrow.parquet.read_table(tasks_path)` followed by `table.to_pydict()["task"]`, and preserve the parquet row order while removing exact duplicates.

- [ ] **Step 4: Add merge tests for Hugging Face remote-only records**

```python
def test_merge_model_records_prefers_local_capabilities():
    merged = merge_model_records(
        [local_record(id="smolvla_demo", tasks=["pick up the bow"])],
        [remote_record(id="smolvla_demo"), remote_record(id="act_remote")],
    )
    assert [item["id"] for item in merged] == ["act_remote", "smolvla_demo"]
    assert merged[0]["downloaded"] is False
    assert merged[0]["selectable"] is False
    assert merged[0]["unavailable_reason"] == "Download required"
    assert merged[1]["tasks"] == ["pick up the bow"]


def test_load_remote_models_reads_tasks_through_injected_hub_loader():
    records = load_remote_models(
        [{"id": "phawitbinabik/smolvla_demo", "name": "smolvla_demo"}],
        ["phawitbinabik/demo"],
        task_loader=lambda dataset_id: ["pick up the bow"] if dataset_id == "phawitbinabik/demo" else [],
    )
    assert records[0]["dataset_id"] == "phawitbinabik/demo"
    assert records[0]["tasks"] == ["pick up the bow"]
    assert records[0]["selectable"] is False
    assert records[0]["unavailable_reason"] == "Download required"
```

- [ ] **Step 5: Implement remote capability loading, deterministic merge, and lookup**

Reuse `_match_model_to_dataset` semantics behind a registry helper, then call an injected `task_loader(dataset_id)` for the matched dataset. In FastAPI integration, that loader calls `huggingface_hub.hf_hub_download(repo_id=dataset_id, filename="meta/tasks.parquet", repo_type="dataset", token=token)` and reads the returned cached file. A task-download failure returns an empty task list and does not fail the entire registry response. Remote-only records use `downloaded: false`, `selectable: false`, and `unavailable_reason: "Download required"`. A matching local record supplies capability fields while retaining the Hugging Face `repo_id`.

- [ ] **Step 6: Add `pyarrow` to runtime requirements and run tests GREEN**

Run: `python3 -m pytest tests/test_model_registry.py -v`

Expected: all registry tests PASS.

- [ ] **Step 7: Commit the registry**

```bash
git add model_registry.py tests/test_model_registry.py requirements.txt
git commit -m "Add task-aware model capability registry"
```

---

### Task 2: Add Planner Configuration, Prompt Rendering, and Plan Validation

**Files:**
- Create: `planner_config.py`
- Create: `tests/test_planner_config.py`
- Modify: `config.yaml`

**Interfaces:**
- Consumes: registry records from `load_local_models`
- Produces: `planner_settings(config: dict) -> dict`
- Produces: `render_available_models(records: list[dict], selected_ids: list[str]) -> str`
- Produces: `render_planner_prompt(config: dict, instruction: str, records: list[dict]) -> tuple[str, list[str]]`
- Produces: `validate_plan(plan: dict, use_ik: bool, selected_records: list[dict]) -> list[dict]`
- Validation errors use `{"code": str, "message": str, "step_index": int | None}`

- [ ] **Step 1: Write failing legacy migration and prompt rendering tests**

```python
def test_planner_settings_migrates_legacy_config_to_use_ik():
    settings = planner_settings({"available_methods": ["smolVLA_v1"], "prompt_template": "legacy"})
    assert settings["use_ik"] is True
    assert settings["selected_models"] == []
    assert settings["prompt_templates"]["use_ik"] == DEFAULT_USE_IK_PROMPT
    assert settings["prompt_templates"]["no_ik"] == DEFAULT_NO_IK_PROMPT


def test_render_available_models_includes_only_selected_exact_tasks():
    text = render_available_models(
        [
            model_record("model_a", ["pick up the bow"]),
            model_record("model_b", ["open the drawer"]),
        ],
        ["model_b"],
    )
    assert "model_b" in text
    assert "open the drawer" in text
    assert "model_a" not in text
    assert "pick up the bow" not in text
```

- [ ] **Step 2: Run planner tests and verify RED**

Run: `python3 -m pytest tests/test_planner_config.py -v`

Expected: FAIL during import with `ModuleNotFoundError: No module named 'planner_config'`.

- [ ] **Step 3: Implement defaults and exact prompt expansion**

`DEFAULT_USE_IK_PROMPT` must explicitly require an immediate IK predecessor, selected-model IDs, declared training tasks, normalized IK bounding boxes, and structured JSON output with `model_id`.

`DEFAULT_NO_IK_PROMPT` must explicitly forbid IK, require an end-to-end trained task per VLA step, set `method_id` to `vla_model`, and return a structured planning error when no selected model covers the instruction.

```python
def render_planner_prompt(config, instruction, records):
    settings = planner_settings(config)
    selected = _selected_records(records, settings["selected_models"])
    template_key = "use_ik" if settings["use_ik"] else "no_ik"
    template = settings["prompt_templates"][template_key]
    model_text = render_available_models(records, settings["selected_models"])
    return (
        template.replace("{instruction}", instruction).replace("{available_models}", model_text),
        [item["id"] for item in selected],
    )
```

- [ ] **Step 4: Write failing plan validation tests for both IK modes**

```python
def test_validate_plan_requires_ik_before_vla_in_use_ik_mode():
    plan = {"steps": [vla_step(1, "model_a", "pick up the bow")]}
    errors = validate_plan(plan, True, [model_record("model_a", ["pick up the bow"])])
    assert errors == [{
        "code": "IK_REQUIRED",
        "message": "VLA step 1 must be immediately preceded by an IK step",
        "step_index": 1,
    }]


def test_validate_plan_rejects_ik_in_no_ik_mode():
    errors = validate_plan({"steps": [ik_step(1)]}, False, [model_record("model_a", ["pick up the bow"])])
    assert errors[0]["code"] == "IK_FORBIDDEN"


def test_validate_plan_rejects_unselected_model_and_task_mismatch():
    errors = validate_plan(
        {"steps": [vla_step(1, "model_b", "open the drawer")]},
        False,
        [model_record("model_a", ["pick up the bow"])],
    )
    assert errors[0]["code"] == "MODEL_NOT_SELECTED"
```

- [ ] **Step 5: Implement fail-closed plan validation**

Task compatibility is normalized case-insensitive equality for execution safety. The planner may describe an IK step freely, but a VLA `description` must equal one declared training task after trimming and case folding.

- [ ] **Step 6: Persist complete planner defaults in config and run tests GREEN**

Run: `python3 -m pytest tests/test_planner_config.py -v`

Expected: all planner tests PASS.

- [ ] **Step 7: Commit planner configuration**

```bash
git add planner_config.py tests/test_planner_config.py config.yaml
git commit -m "Add IK-aware multi-model planner configuration"
```

---

### Task 3: Expose Registry and Planner Contracts Through FastAPI

**Files:**
- Modify: `main.py:266-340`
- Modify: `main.py:714-875`
- Create: `tests/test_planner_api.py`

**Interfaces:**
- Consumes: `load_local_models`, `merge_model_records`, `planner_settings`, `render_planner_prompt`, `validate_plan`
- Produces: `GET /api/models/registry`
- Extends: `GET /api/config`, `POST /api/config/save`, `POST /api/infer`

- [ ] **Step 1: Write failing API tests with FastAPI TestClient**

```python
def test_config_save_rejects_empty_selected_models(client):
    response = client.post("/api/config/save", json={
        "planner": {
            "use_ik": False,
            "selected_models": [],
            "prompt_templates": {"use_ik": "A", "no_ik": "B"},
        }
    })
    assert response.status_code == 400
    assert response.json()["code"] == "NO_SELECTED_MODELS"


def test_model_registry_never_returns_arbitrary_absolute_client_paths(client):
    response = client.get("/api/models/registry")
    assert response.status_code == 200
    assert all(not item["local_path"].startswith("/") for item in response.json()["models"] if item["local_path"])
```

Patch only registry discovery and config-file I/O at the module boundary; do not mock FastAPI routing or response serialization.

- [ ] **Step 2: Run API tests and verify RED**

Run: `/opt/miniconda3/envs/lerobot/bin/python -m pytest tests/test_planner_api.py -v`

Expected: FAIL because `/api/models/registry` is 404 and planner save validation is absent.

- [ ] **Step 3: Add registry route and planner config response fields**

```python
@app.get("/api/models/registry")
async def model_registry():
    cfg = load_config()
    records = _load_model_registry(cfg)
    return {"ok": True, "models": records}
```

Return only workspace-relative `local_path` values. Keep `models` for Gemini planner model choices separate from robot-policy registry records.

- [ ] **Step 4: Validate and persist planner config atomically**

Validate selected IDs against current selectable registry records before writing. Write YAML to a sibling temporary file and replace `config.yaml` only after successful serialization. Preserve unrelated config keys.

- [ ] **Step 5: Route planning through server-rendered prompts and validate output**

`POST /api/infer` must ignore a browser-supplied prompt for the Run flow when planner fields are present, render the selected template server-side, and append validation failures to the existing verifier retry feedback. After retries, return `error_code: "INVALID_PLAN"` plus `validation_errors` instead of an executable invalid plan.

- [ ] **Step 6: Run API, registry, and planner tests GREEN**

Run: `/opt/miniconda3/envs/lerobot/bin/python -m pytest tests/test_model_registry.py tests/test_planner_config.py tests/test_planner_api.py -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit API integration**

```bash
git add main.py tests/test_planner_api.py
git commit -m "Expose multi-model planner APIs"
```

---

### Task 4: Add Bounded VLA Step Execution

**Files:**
- Create: `vla_execution.py`
- Create: `tests/test_vla_execution.py`
- Modify: `infer_python.py`
- Modify: `main.py:7000-7410`

**Interfaces:**
- Consumes: a validated registry record and robot camera config
- Produces: `build_infer_command(python: str, script: Path, model: dict, task: str, robot: dict, cameras: dict, max_steps: int) -> list[str]`
- Produces: `VlaProcessManager.start(command: list[str], model_id: str, task: str) -> dict`
- Produces: `VlaProcessManager.status() -> dict`
- Produces: `VlaProcessManager.stop(timeout: float = 5.0) -> dict`
- Extends: `infer_python.py --max-steps`

- [ ] **Step 1: Write failing safe-command and lifecycle tests**

```python
def test_build_infer_command_uses_registry_path_and_exact_task(tmp_path):
    command = build_infer_command(
        python="/env/bin/python",
        script=Path("infer_python.py"),
        model=selectable_model(tmp_path / "models" / "model_a", ["observation.images.top"]),
        task="pick up the bow",
        robot={"port": "/dev/follower", "id": "arm"},
        cameras={"top": {"index": 1, "w": 320, "h": 240}},
        max_steps=100,
    )
    assert "--model-path=" + str(tmp_path / "models" / "model_a") in command
    assert "--task=pick up the bow" in command
    assert '--cameras={"top": 1}' in command
    assert "--max-steps=100" in command


def test_process_manager_tracks_real_subprocess_completion():
    manager = VlaProcessManager(max_log_lines=20)
    manager.start([sys.executable, "-c", "print('ready')"], "model_a", "pick up the bow")
    wait_until(lambda: not manager.status()["running"])
    status = manager.status()
    assert status["state"] == "completed"
    assert status["exit_code"] == 0
    assert status["lines"] == ["ready"]
```

- [ ] **Step 2: Run execution tests and verify RED**

Run: `python3 -m pytest tests/test_vla_execution.py -v`

Expected: FAIL during import with `ModuleNotFoundError: No module named 'vla_execution'`.

- [ ] **Step 3: Implement command construction and process ownership**

Use `subprocess.Popen(command, stdin=DEVNULL, stdout=PIPE, stderr=STDOUT, start_new_session=True)` without shell invocation. Manager states are exactly `idle`, `starting`, `running`, `completed`, `failed`, and `stopped`. Protect shared state with one `threading.Lock` and cap logs at the constructor limit.

- [ ] **Step 4: Add failing `infer_python.py --max-steps` parser/loop tests**

Extract `should_continue(running: bool, step: int, max_steps: int) -> bool` so it can be tested without importing torch or robot dependencies.

```python
def test_should_continue_stops_at_bounded_step_count():
    assert should_continue(True, 99, 100) is True
    assert should_continue(True, 100, 100) is False
    assert should_continue(True, 100, 0) is True
    assert should_continue(False, 0, 0) is False
```

- [ ] **Step 5: Implement bounded inference and failure exit codes**

Add `--max-steps` with default `0`. Replace `while _running` with `while should_continue(_running, step, args.max_steps)`. On an inference exception, print the traceback, disconnect in `finally`, and return exit code `1`; normal bounded completion returns `0`.

- [ ] **Step 6: Integrate the process manager with Run Step routes**

For `method_id: "vla_model"`, `POST /api/run/step` validates selected ID, exact declared task, selectable status, hardware ownership, and camera keys before disconnecting slider-mode robot and starting the process. Add:

```python
@app.get("/api/run/status")
async def run_status():
    return vla_manager.status()


@app.post("/api/run/stop")
async def run_stop():
    return vla_manager.stop()
```

Use `max_steps` from request with a server default of `100` and bounds `1..1000`.

- [ ] **Step 7: Run execution and API tests GREEN**

Run: `/opt/miniconda3/envs/lerobot/bin/python -m pytest tests/test_vla_execution.py tests/test_planner_api.py -v`

Expected: all tests PASS and no robot hardware is opened by tests.

- [ ] **Step 8: Commit execution support**

```bash
git add vla_execution.py infer_python.py main.py tests/test_vla_execution.py tests/test_planner_api.py
git commit -m "Execute selected VLA models from plan steps"
```

---

### Task 5: Replace the Methods Editor With the Model Registry UI

**Files:**
- Modify: `static/index.html:95-110`
- Modify: `static/index.html:490-515`
- Modify: `static/index.html:3200-3420`
- Create: `tests/test_multi_model_ui.js`

**Interfaces:**
- Consumes: `GET /api/models/registry`, planner fields from `GET /api/config`, download endpoints
- Produces: `cfgModelRegistry`, `cfgSelectedModelIds`, `cfgUseIk`, and planner payload for `POST /api/config/save`

- [ ] **Step 1: Write failing Node behavior tests for selection and mode switching**

```javascript
assert.deepEqual(
  selectableModelIds([
    {id: 'a', selectable: true},
    {id: 'b', selectable: false},
  ]),
  ['a'],
);

assert.throws(
  () => buildPlannerPayload({selectedIds: [], useIk: false, useIkPrompt: 'A', noIkPrompt: 'B'}),
  /Select at least one downloaded model/,
);

assert.deepEqual(
  buildPlannerPayload({selectedIds: ['a', 'b'], useIk: false, useIkPrompt: 'A', noIkPrompt: 'B'}),
  {
    use_ik: false,
    selected_models: ['a', 'b'],
    prompt_templates: {use_ik: 'A', no_ik: 'B'},
  },
);
```

Extract and execute the real functions from `static/index.html` with `vm.runInNewContext`; assertions must exercise function output, not grep source lines.

- [ ] **Step 2: Run UI tests and verify RED**

Run: `node tests/test_multi_model_ui.js`

Expected: FAIL because `selectableModelIds` and `buildPlannerPayload` do not exist.

- [ ] **Step 3: Implement pure selection/payload functions and model cards**

Cards show checkbox, display name, policy type, local/remote status, dataset, camera keys, and an expandable `<ul>` of exact task strings. Remote-only cards show Download and cannot toggle selection. Missing-capability records display their `unavailable_reason`.

- [ ] **Step 4: Implement IK radio and dual prompt editor state**

Store both textarea values in DOM-backed state before switching which editor is visible. Labels are exactly `Use IK` and `Not use IK`. Mode switching updates explanatory copy but never rewrites either prompt.

- [ ] **Step 5: Wire refresh/download/save**

Refresh calls the registry route and preserves selected IDs still present and selectable. Download uses the existing batch/single endpoint, polls existing download status, refreshes registry on completion, and does not auto-select the model. Save sends one `planner` object built by `buildPlannerPayload`.

- [ ] **Step 6: Run UI syntax and behavior tests GREEN**

Run: `node tests/test_multi_model_ui.js`

Run: `perl -0777 -ne 'while (/<script(?:\s[^>]*)?>(.*?)<\/script>/sg) { print "$1\n" }' static/index.html > /tmp/hybridge-inline.js && node --check /tmp/hybridge-inline.js`

Expected: behavior test prints PASS and Node syntax check exits 0.

- [ ] **Step 7: Commit the Config UI without staging the prior WebSocket lines**

Use interactive patch staging or create a temporary focused patch so the pre-existing `calConnectWs` change remains unstaged.

```bash
git add tests/test_multi_model_ui.js
git add -p static/index.html
git commit -m "Add task-aware multi-model Methods UI"
```

---

### Task 6: Update Planning and Run UI for Explicit Model Steps

**Files:**
- Modify: `static/index.html:4200-4430`
- Modify: `static/index.html:3770-3860`
- Modify: `tests/test_multi_model_ui.js`

**Interfaces:**
- Consumes: planner config, returned plan `model_id`, `POST /api/run/step`, `GET /api/run/status`, `POST /api/run/stop`
- Produces: model-aware plan cards and bounded execution controls

- [ ] **Step 1: Write failing tests for plan payload and VLA status mapping**

```javascript
assert.deepEqual(
  buildRunStepPayload({
    method_id: 'vla_model',
    model_id: 'model_a',
    description: 'pick up the bow',
    target_bbox: null,
  }, 100),
  {
    method_id: 'vla_model',
    model_id: 'model_a',
    description: 'pick up the bow',
    target_bbox: null,
    max_steps: 100,
  },
);

assert.equal(runStateLabel({state: 'running', model_id: 'model_a'}), 'Running model_a');
assert.equal(runStateLabel({state: 'failed', exit_code: 1}), 'Failed (exit 1)');
```

- [ ] **Step 2: Run UI tests and verify RED**

Run: `node tests/test_multi_model_ui.js`

Expected: FAIL because `buildRunStepPayload` and `runStateLabel` do not exist.

- [ ] **Step 3: Build planning requests from saved planner config**

Run Plan sends Gemini model, instruction, and image. The server selects the correct prompt and selected policies from saved planner config. Remove client reconstruction using legacy `available_methods` and `prompt_template` for this flow.

- [ ] **Step 4: Render model identity on every VLA step**

Plan, history, and Run cards show `model_id` below `vla_model`. IK cards show only `ik_reach_object_v1`. Escaping must use the existing `esc` helper for descriptions and IDs.

- [ ] **Step 5: Poll bounded VLA execution and expose Stop**

After a VLA start response, poll `/api/run/status` every 500 ms. Mark the step done only when state is `completed`. On `failed`, keep it selected and show exit code plus the last log lines. While `running`, show a Stop button that calls `/api/run/stop`; stopped steps are not marked complete.

- [ ] **Step 6: Run UI tests and full inline-script syntax check GREEN**

Run: `node tests/test_multi_model_ui.js && node tests/test_camera_websocket.js`

Run: `perl -0777 -ne 'while (/<script(?:\s[^>]*)?>(.*?)<\/script>/sg) { print "$1\n" }' static/index.html > /tmp/hybridge-inline.js && node --check /tmp/hybridge-inline.js`

Expected: both behavior tests PASS and syntax check exits 0.

- [ ] **Step 7: Commit Run UI changes without staging the prior WebSocket fix**

```bash
git add tests/test_multi_model_ui.js
git add -p static/index.html
git commit -m "Run explicit VLA model plan steps"
```

---

### Task 7: Full Migration and Verification

**Files:**
- Modify: `static/index.html:692-750`
- Modify: `tests/test_planner_api.py`
- Modify: `tests/test_multi_model_ui.js`

**Interfaces:**
- Consumes: all prior task interfaces
- Produces: updated in-app documentation and end-to-end regression coverage

- [ ] **Step 1: Add an API integration test covering saved config to executable plan**

```python
def test_saved_no_ik_model_reaches_vla_run_validation(client, registry_fixture):
    saved = client.post("/api/config/save", json={"planner": {
        "use_ik": False,
        "selected_models": ["model_a"],
        "prompt_templates": {"use_ik": "{instruction}\n{available_models}", "no_ik": "{instruction}\n{available_models}"},
    }})
    assert saved.status_code == 200
    response = client.post("/api/run/step", json={
        "method_id": "vla_model",
        "model_id": "model_a",
        "description": "pick up the bow",
        "target_bbox": None,
        "max_steps": 100,
    })
    assert response.status_code == 200
    assert response.json()["model_id"] == "model_a"
```

Patch the process manager start boundary so this integration test validates application behavior without opening cameras or robot ports.

- [ ] **Step 2: Run the integration test and fix only uncovered contract gaps**

Run: `/opt/miniconda3/envs/lerobot/bin/python -m pytest tests/test_planner_api.py::test_saved_no_ik_model_reaches_vla_run_validation -v`

Expected: PASS after prior tasks; if it fails, change the smallest mismatched interface and rerun all affected tests.

- [ ] **Step 3: Update in-app documentation**

Document model capability sources, disabled/download behavior, Use IK versus Not use IK planning, explicit `model_id`, exact-task safety validation, bounded execution, status polling, and Stop semantics. Remove documentation that describes free-form Methods as the active configuration.

- [ ] **Step 4: Run the complete automated verification suite**

Run: `/opt/miniconda3/envs/lerobot/bin/python -m pytest tests/test_model_registry.py tests/test_planner_config.py tests/test_planner_api.py tests/test_vla_execution.py -v`

Run: `node tests/test_multi_model_ui.js && node tests/test_camera_websocket.js`

Run: `python3 -m py_compile main.py model_registry.py planner_config.py vla_execution.py infer_python.py`

Run: `python3 -m json.tool fakecam_params.json >/dev/null && git diff --check`

Expected: zero pytest failures, both Node tests PASS, Python compilation exits 0, JSON validation exits 0, and no whitespace errors.

- [ ] **Step 5: Perform a read-only runtime smoke test**

Start the app with the LeRobot interpreter, request `/api/config` and `/api/models/registry`, and confirm at least one known local model reports exact tasks without connecting robot hardware:

```bash
/opt/miniconda3/envs/lerobot/bin/python main.py
curl -sS http://127.0.0.1:8007/api/models/registry | python3 -m json.tool
```

Stop the smoke-test server with Ctrl+C. Do not call `/api/run/step` against physical hardware as part of automated verification.

- [ ] **Step 6: Commit documentation and final integration coverage**

```bash
git add static/index.html tests/test_planner_api.py tests/test_multi_model_ui.js
git commit -m "Document and verify multi-model planning flow"
```

- [ ] **Step 7: Verify final branch state and preserve unrelated work**

Run: `git status --short --branch && git log --oneline -8`

Expected: feature commits are present; only the previously uncommitted Camera & Teleop WebSocket fix may remain outside feature commits unless the user separately asks to include it.
