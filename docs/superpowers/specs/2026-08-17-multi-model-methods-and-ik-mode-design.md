# Multi-Model Methods and IK Mode Design

## Goal

Replace the free-form Methods configuration with a model-aware registry that lets an operator select multiple trained robot policies, understand the text instructions each policy was trained on, choose whether planning uses IK, and execute the model selected for each plan step.

The complete flow is:

1. Discover local and Hugging Face models.
2. Resolve each model to its training dataset and exact training tasks.
3. Let the operator select multiple downloaded models.
4. Generate a plan using either the Use IK or Not use IK prompt.
5. Record the selected checkpoint explicitly on each VLA step.
6. Execute that checkpoint with the step's text instruction.

## Current-State Problems

- `available_methods` is a free-form list of strings and does not identify a checkpoint.
- The planner sees `smolVLA_v1` but cannot distinguish models trained for different tasks.
- Training instructions are available in dataset metadata but are not exposed during model selection or planning.
- The single prompt always requires IK and cannot represent an end-to-end VLA-only plan.
- `/api/run/step` supports IK and placeholder action methods but does not execute a selected VLA checkpoint.

## Model Registry

### Sources

The registry merges:

- Models returned by the configured user's Hugging Face account.
- Downloaded model directories under `models/`.

For a downloaded model, `train_config.json` is authoritative for:

- Policy type.
- Dataset `repo_id` and local `root`.
- Input camera features.
- Local checkpoint path.

The registry resolves the training task list through the model's dataset configuration and reads the exact task strings from `meta/tasks.parquet`. Dataset-name matching is a fallback only when the configured local root is missing. A Hugging Face dataset may supply task metadata for a remote model when the metadata is available through the existing authenticated integration.

### Registry Record

Each API record has a stable shape:

```json
{
  "id": "smolvla_DS4_2B2O_FF_FULL",
  "repo_id": "phawitbinabik/smolvla_DS4_2B2O_FF_FULL",
  "display_name": "smolvla_DS4_2B2O_FF_FULL",
  "policy_type": "smolvla",
  "downloaded": true,
  "local_path": "models/smolvla_DS4_2B2O_FF_FULL",
  "dataset_id": "phawitbinabik/DS4_2B2O_FF_FULL",
  "tasks": [
    "pick up the start to the green bowl",
    "pick up the bow to the white bowl"
  ],
  "camera_features": [
    "observation.images.top",
    "observation.images.wrist"
  ],
  "selectable": true,
  "unavailable_reason": null
}
```

The registry ID is stable and separate from the display name. Paths are resolved and validated server-side; the browser never supplies an arbitrary checkpoint path.

### Selectability Rules

A model is selectable only when:

- It is downloaded locally.
- Its policy type is supported by the inference launcher.
- Its training instructions are known and non-empty.
- Its required input features can be supplied by the configured robot cameras.

Remote-only models remain visible but disabled and expose a Download action. A model with missing task metadata displays `Training instructions unavailable` and remains disabled so the planner cannot guess its capabilities.

## Config UI

### Model Methods

The existing free-form Methods chip editor is replaced by a Model Registry panel. It provides:

- Refresh action.
- Multi-select checkboxes.
- Download action for remote-only models.
- Local/remote status.
- Policy type and source dataset.
- Required cameras.
- Expandable exact training instructions.

At least one model must be selected before the configuration can be saved.

### IK Mode

A required radio group provides:

- `Use IK`
- `Not use IK`

`Use IK` is the migration default for existing configurations.

### Prompt Editors

Config stores and edits two independent prompt templates:

- Use IK prompt.
- Not use IK prompt.

The visible editor follows the selected IK radio mode. Switching modes does not overwrite either template. Both templates support:

- `{instruction}`
- `{available_models}`

`{available_models}` expands to selected model IDs plus policy type, exact training tasks, and camera requirements.

## Configuration Schema

The new configuration shape is:

```yaml
planner:
  use_ik: true
  selected_models:
    - smolvla_DS4_2B2O_FF_FULL
  prompt_templates:
    use_ik: "Configured Use IK planner prompt"
    no_ik: "Configured Not use IK planner prompt"
```

The API continues returning compatibility fields required by existing screens during migration. Existing `available_methods` and `prompt_template` values are read as legacy input but are replaced by the planner fields after the first successful save. Existing installations default to `use_ik: true`.

## Planner Contracts

### Available Model Description

The planner receives only selected models. Each entry includes the exact commands from the training dataset. The planner must not choose an unselected model or invent a checkpoint ID.

### Use IK Prompt

The default Use IK prompt instructs the planner to:

- Use `ik_reach_object_v1` only to move the arm to the next interaction target.
- Put an IK step immediately before every VLA interaction step.
- Choose a selected model whose training instructions cover the required interaction.
- Keep the VLA `description` equal or semantically as close as possible to a training instruction.
- Never assign a task outside the chosen model's declared capabilities.
- Return a structured planning error when no selected model can perform a required interaction.
- Preserve normalized bounding boxes for IK targets.

### Not use IK Prompt

The default Not use IK prompt instructs the planner to:

- Never emit `ik_reach_object_v1` or any other reach-only step.
- Treat a VLA step as an end-to-end trained task.
- Prefer one complete trained instruction over decomposing it into motion primitives.
- Choose only a selected model whose training instruction covers the requested task.
- Use the closest exact training instruction as the VLA `description`.
- Return a structured planning error when no selected model covers the task.

In Not use IK mode, `target_bbox` may be returned for visualization but is not consumed for motion execution.

### Plan Step Schema

VLA step:

```json
{
  "step_index": 1,
  "description": "pick up the bow to the green bowl",
  "target_bbox": null,
  "method_id": "vla_model",
  "model_id": "smolvla_DS4_2B2O_FF_FULL"
}
```

IK step:

```json
{
  "step_index": 1,
  "description": "Reach the pink bow",
  "target_bbox": [0.42, 0.55, 0.06, 0.06],
  "method_id": "ik_reach_object_v1",
  "model_id": null
}
```

Separating `method_id` from `model_id` keeps execution semantics stable across SmolVLA, ACT, Pi0.5, and future policy types.

### Verification

The verifier receives the same IK mode and selected model capability list. It rejects plans when:

- A model is not selected.
- A model does not declare a compatible training task.
- Use IK mode has a VLA step without its required IK predecessor.
- Not use IK mode contains an IK step.
- A plan invents a model ID or unsupported method.

## VLA Step Execution

### Request Validation

`/api/run/step` validates a VLA step before launching anything:

1. `method_id` is `vla_model`.
2. `model_id` is in the saved selected-model list.
3. The registry reports that the model is downloaded and selectable.
4. The requested description is compatible with one of the model's declared training tasks.
5. Configured cameras satisfy the checkpoint's input features.
6. No conflicting robot, teleop, collection, or inference process owns the hardware.

The server resolves the local path from the registry and ignores client-supplied paths.

### Inference Lifecycle

Execution uses a policy-type adapter boundary. Each supported adapter builds the appropriate LeRobot inference command while sharing lifecycle management:

- Start the selected local checkpoint.
- Pass the step description as the task instruction where the policy supports language.
- Supply only the camera keys required by the checkpoint.
- Capture stdout and stderr in a bounded log buffer.
- Expose running, success, failure, and stopped states.
- Support explicit stop with terminate followed by kill timeout fallback.
- Always release cameras, ports, and process state.

The first implementation supports policy types for which this repository already has a reliable evaluation command. Unsupported policy types remain visible but not selectable and explain why.

### UI Feedback

Run Step displays:

- Selected model and policy type.
- Exact instruction being sent.
- Starting/running/completed/failed/stopped state.
- Recent inference log output.
- Stop action while inference is active.

The UI does not mark a step complete until the backend process reports success.

## API Changes

- `GET /api/models/registry` returns merged model capability records.
- Existing model download endpoints remain the download mechanism and trigger registry refresh after completion.
- `GET /api/config` includes planner mode, selected model IDs, and both prompt templates.
- `POST /api/config/save` validates and persists the planner configuration.
- `POST /api/infer` receives IK mode and selected model IDs or resolves them from the saved configuration, then validates the returned plan.
- `POST /api/run/step` accepts `model_id` for `vla_model` steps and starts the matching adapter.
- `GET /api/run/status` reports VLA inference lifecycle and logs.
- `POST /api/run/stop` stops active VLA inference safely.

## Error Handling

Errors are returned as structured JSON with a stable code and readable message. Important cases include:

- No selected model.
- Model not downloaded.
- Missing training metadata.
- Unsupported policy type.
- Required camera unavailable.
- Task outside model capabilities.
- Model ID not selected.
- Hardware busy.
- Inference launch failure.
- Inference process exit failure.

Planner capability failures are shown before physical execution. Execution validation fails closed and never substitutes another checkpoint automatically.

## Testing Strategy

### Registry Tests

- Resolve dataset root and repository ID from representative `train_config.json` files.
- Read exact task strings from `tasks.parquet`.
- Extract policy type and camera features.
- Merge local and remote records without duplicates.
- Disable remote-only, unsupported, missing-task, and camera-incompatible records.

### Config and Prompt Tests

- Migrate legacy config to Use IK.
- Require at least one selected model.
- Preserve two templates independently.
- Expand only selected model capabilities.
- Use IK prompt requires the IK/VLA sequence.
- Not use IK prompt forbids IK steps.

### Plan Validation Tests

- Reject unselected or invented model IDs.
- Reject a model/task mismatch.
- Reject missing IK in Use IK mode.
- Reject any IK step in Not use IK mode.
- Accept valid VLA and IK plan schemas.

### Execution Tests

- Resolve model paths server-side.
- Build the correct adapter command with checkpoint, cameras, and instruction.
- Reject invalid or busy execution requests before process launch.
- Track successful, failed, and stopped processes.
- Clean up process and hardware state on every exit path.

### UI Tests

- Multi-select downloaded models.
- Disable and download remote models.
- Show exact training instructions.
- Switch IK mode and prompt editor without losing edits.
- Render model ID on VLA plan steps.
- Display execution progress, logs, errors, and stop state.

## Scope Boundaries

This feature does not train models, infer missing training capabilities from model names, ensemble multiple policies on one step, or silently fall back to a different model. It selects one declared-capable checkpoint per VLA step and executes it explicitly.
