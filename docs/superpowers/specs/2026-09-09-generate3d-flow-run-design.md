# Generate 3D Flow Run Design

**Date:** 2026-09-09  
**Status:** Approved design, implementation pending  
**Branch:** `codex/generate3d-flow-run`

## Goal

Add a `Flow Run` subtab under Generate 3D for long, ordered pick-and-place instructions. The system decomposes one instruction into atomic subtasks, detects the required source and target from a fresh frame before every subtask, previews and executes the motion, captures a verification frame, and advances only after a successful verification. The interface must make the current phase, inputs, outputs, evidence, and failures easy to understand, while retaining the controls available in `3D Simulation`.

Example input:

> Pick up the pink bow to the green bowl and pick up the star to the green bowl.

This becomes two ordered blocks:

1. Pick up the pink bow and place it in the green bowl.
2. Pick up the star and place it in the green bowl.

## Scope

### Included

- A third Generate 3D subtab named `Flow Run`.
- Gemini planning of up to 10 atomic pick-and-place subtasks.
- A backend-owned, single-active-flow orchestrator.
- Fresh capture, instruction-guided detection, motion planning, execution, capture, and verification for every block.
- `Run All`, `Run This Block`, `Retry Block`, and `Stop Flow` controls.
- A flowchart-style progress display and a detailed selected-block inspector.
- Simulation and real-robot execution modes.
- Persistent flow journal and image artifacts.
- Reuse of 3D Simulation capabilities: full read-only Gemini prompt preview, capture/upload, editable detections, 3D/path preview, waypoint/smooth motion choice, pick/place height controls, calibrated-workspace safety option defaulting to unchecked, rest-position controls, raw Gemini output, stop control, and live robot state.
- Automatic return to the saved Rest Position after a successful `Run All`, enabled by default and configurable before the run.

### Excluded

- General-purpose actions other than pick-and-place.
- Parallel robot actions or multiple active flows.
- Automatic continuation after `failed`, `uncertain`, or `interrupted` verification.
- Automatic retry, autonomous replanning, or silently skipping a failed block.
- Automatic resumption of physical motion after server restart.
- Merging this branch into `main`.

## Architecture Decision

### Selected: backend flow orchestrator over the existing Generate 3D pipeline

The server owns flow state, phase transitions, persistence, hardware exclusion, and background execution. Shared Generate 3D helpers perform capture, detection, geometric planning, and robot execution without making internal HTTP requests. The browser renders state and submits commands but is not the source of truth.

This approach survives page refreshes, keeps safety decisions close to hardware control, and makes phase transitions testable independently of the UI.

### Alternatives considered

1. **Browser-only orchestration.** This is simpler initially, but a refresh or lost connection can strand an active physical action and lose the authoritative phase. Rejected for safety and reliability.
2. **Reuse the generic VLA Run session directly.** It already has verified looping, but its action model and prompts are VLA-oriented rather than Generate 3D geometric pick/place. Adapting it would couple two different execution semantics. Rejected; only its proven verification and UI patterns will be borrowed.
3. **One monolithic long Gemini/robot call.** This hides intermediate evidence and cannot enforce verification between subtasks. Rejected.

## User Experience

### Subtab layout

`Generate 3D` contains:

- `Camera Calibrate`
- `3D Simulation`
- `Flow Run`

The Flow Run screen is organized in this order:

1. **Long Task & Gemini Input** — instruction field, active Gemini model, and the complete read-only system prompt shown before `Detect & Generate Flow`.
2. **Source Image** — camera capture and image upload controls, preview, and capture status.
3. **Flow controls** — `Detect & Generate Flow`, execution mode (`Simulation` or `Real Robot`), motion mode (`Natural Smooth` or `Waypoint`), pick/place heights, workspace-safety checkbox (unchecked by default), return-to-rest checkbox (checked by default), `Run All`, and `Stop Flow`.
4. **Flowchart** — horizontally or vertically responsive block cards connected in sequence.
5. **Selected block details** — editable detected source/target objects, 3D/path preview, inputs, outputs, images, verification evidence, errors, and per-block actions.
6. **Robot & diagnostics** — live robot state, rest-position save/return controls, and raw Gemini output.

### Block appearance and states

Each block displays its sequence number, short instruction, source → target, current phase, and final outcome. Color conveys status but text and icons remain the authoritative accessibility cues:

- gray: `pending`
- blue: `capturing`, `detecting`, or `planning`
- orange: `executing`
- purple: `verifying`
- green: `success`
- red: `failed`, `uncertain`, or `interrupted`

Clicking a block selects it without running it. The detail pane shows:

- atomic instruction and source/target names;
- detected bounding boxes, confidence, estimated dimensions, and robot positions;
- pick and place heights actually used;
- simulation/real and smooth/waypoint modes;
- before-execution and verification images;
- complete Gemini input and raw output for planning, detection, and verification;
- verification status, visible evidence, reason, timestamps, and errors;
- `Run This Block` for an unstarted block or `Retry Block` for a terminal unsuccessful block.

### Control behavior

- `Detect & Generate Flow` plans the flow from the long instruction and source image. It does not move the robot.
- `Run All` starts at the first non-successful block and proceeds in order. Existing successful blocks are preserved; a user must explicitly retry one to overwrite its result.
- `Run This Block` executes only the selected block and never advances to another block.
- Both run modes acquire a fresh frame and repeat detection. Preview detections from initial flow generation are explanatory, not trusted for physical execution.
- `Retry Block` clears that block's execution-phase artifacts and status, then runs only that block from a fresh frame. It does not clear successful earlier or later blocks.
- `Stop Flow` requests cooperative cancellation, prevents the next phase from starting, and invokes the existing safe robot stop for a physical execution in progress.
- Editing the long instruction and regenerating creates a new flow ID and retains prior flow artifacts on disk.
- Editing object boxes or heights applies to the selected preview. A real run still refreshes detections; user-adjusted values are only reused when they remain associated with the same fresh captured image.

## Planning Contract

Gemini receives the complete read-only planning prompt, the user's long instruction, and the initial image. It must return strict JSON:

```json
{
  "subtasks": [
    {
      "instruction": "Pick up the pink bow and place it in the green bowl.",
      "source_name": "pink bow",
      "target_name": "green bowl"
    }
  ]
}
```

Validation rules:

- `subtasks` contains 1–10 items.
- Every item contains non-empty `instruction`, `source_name`, and `target_name` strings.
- Every item is one source-to-target pick-and-place operation.
- Order matches the user's instruction.
- Duplicate operations remain separate when the instruction explicitly repeats them.
- Markdown fences and explanatory prose are rejected unless the existing JSON extraction utility can unambiguously isolate one JSON object.
- Invalid output returns a user-readable planning error and performs no execution.

The planner does not recommend heights. Existing deterministic rules remain authoritative:

- Pick Height defaults to `0 cm`, relative to the detected source object's base.
- Place Height defaults to target estimated height plus `2 cm`, measured from the workspace floor.
- The user may override both values before execution.

## Per-Block State Machine

The valid phase sequence is:

```text
pending
  → capturing
  → detecting
  → planning
  → executing
  → capturing_verification
  → verifying
  → success
```

Any active phase may terminate as `failed`, `uncertain`, `stopped`, or `interrupted`:

- `failed`: a definitive operation, validation, detection, planning, execution, or verification failure.
- `uncertain`: Gemini cannot establish success from visible evidence.
- `stopped`: the user requested a stop and the phase exited cooperatively.
- `interrupted`: persisted state shows the server stopped while the block was active.

Only `success` allows `Run All` to advance. `failed`, `uncertain`, `stopped`, and `interrupted` stop the flow and require an explicit retry or block run.

### Execution sequence

For each block:

1. Capture the current camera frame in real mode. In simulation mode, snapshot the current simulated scene and retain the user's uploaded/captured camera frame as audit context.
2. Detect only the block's named source and target using the existing normalized `box_2d` contract.
3. Require both objects, valid bounding boxes, and usable calibrated robot positions for real mode. Missing objects are shown with detection evidence and end the block as `failed` without movement.
4. Build the 3D objects and geometric motion path with current height, safety, and motion-mode settings.
5. Persist the planned path and expose it in the selected-block preview.
6. Execute through the existing Generate 3D task manager. The flow orchestrator waits on task status rather than duplicating robot motion code.
7. Capture the post-execution frame or simulated-scene snapshot.
8. Verify outcome according to the mode.
9. Persist the terminal result. Advance only when running all and the result is `success`.

## Verification

### Real Robot mode

Gemini receives the atomic instruction, source/target identities, the before frame, and the post-execution frame. It returns:

```json
{
  "status": "success",
  "reason": "The pink bow is now visibly inside the green bowl.",
  "visible_evidence": [
    "The bow is absent from its original location.",
    "The bow is visible within the bowl boundary."
  ]
}
```

`status` must be exactly `success`, `failed`, or `uncertain`. Invalid JSON, missing fields, API errors, or non-decisive output become `uncertain`, never `success`. Verification checks visible task outcome, not merely whether robot commands completed.

### Simulation mode

The system still creates before/after audit snapshots, but the result is determined from authoritative simulated state rather than asking Gemini to infer a change that did not occur in the physical camera frame. Success requires the simulated source to end at the selected target according to the completed motion model. Execution errors produce `failed`; missing state produces `uncertain`.

## Data Model and Persistence

Only one flow may be active at a time. The persisted journal is `data/generate3d_flow_state.json` and is written atomically after every phase transition.

Top-level flow shape:

```json
{
  "schema_version": 1,
  "flow_id": "20260909T120000Z-a1b2c3d4",
  "instruction": "...",
  "status": "ready",
  "active_block_index": null,
  "run_scope": null,
  "config": {
    "execution_mode": "simulation",
    "motion_mode": "smooth",
    "pick_height_cm": 0.0,
    "place_height_cm": 8.0,
    "use_calibrated_workspace_safety": false,
    "return_to_rest_on_success": true
  },
  "blocks": [],
  "created_at": "...",
  "updated_at": "..."
}
```

Each block stores:

- stable block ID and zero-based index;
- instruction, source name, and target name;
- phase and terminal outcome;
- per-phase timestamps;
- configuration snapshot actually used;
- detection objects and coordinate mappings;
- planned motion/path metadata;
- linked Generate 3D task ID;
- before and verification artifact paths;
- Gemini prompt and raw output records;
- verification status, reason, and evidence;
- user-facing error and machine-readable error code.

Artifacts are stored under `data/generate3d_flows/<flow_id>/` with block-specific filenames. API responses expose artifact URLs through a controlled endpoint and never accept arbitrary filesystem paths.

At application startup:

- a persisted terminal or ready flow is restored for viewing;
- if the persisted flow or block was active, it is marked `interrupted`;
- no camera capture or robot motion restarts automatically;
- the user must select `Retry Block` or start an eligible block explicitly.

Persistence failure prevents a new physical phase from starting and reports a `failed` result. In-memory state may remain visible for diagnosis.

## Backend Components

### `Generate3DFlowManager`

A dedicated manager will:

- validate and store planned flows;
- enforce one active execution worker;
- own the cancellation event and phase-transition rules;
- persist journals and artifacts;
- invoke shared capture, detection, planning, execution, and verification services;
- expose immutable status snapshots to request handlers;
- coordinate with existing Generate 3D task execution so two robot tasks cannot run concurrently.

The existing `Generate3DTaskManager` remains responsible for one geometric robot task. Shared hardware exclusion will reject a new Flow Run or 3D Simulation physical execution while the other owns the robot.

### Shared services

Endpoint logic currently embedded in `main.py` will be extracted only where Flow Run needs identical behavior:

- instruction-guided Gemini object detection;
- camera capture and image normalization;
- deterministic height calculation;
- geometric task validation and launch;
- Gemini JSON parsing and prompt-record construction.

This is a targeted extraction, not a broad rewrite of the existing application.

## API Contract

### `POST /api/generate3d/flow/plan`

Input includes the long instruction, selected Gemini model, and initial image source. Returns the complete new flow snapshot. Rejects empty instructions, invalid planner output, more than 10 blocks, and attempts to replace an actively executing flow.

### `GET /api/generate3d/flow/status`

Returns the persisted/current flow snapshot, manager activity, and whether stop has been requested. If no flow exists, returns a stable empty-state payload rather than an error.

### `POST /api/generate3d/flow/start`

Input includes `scope` (`all` or `block`), optional `block_index`, and the current execution configuration. It validates preflight, freezes the configuration snapshot for each block as it starts, and launches the background worker. `block_index` is required for block scope.

### `POST /api/generate3d/flow/stop`

Requests cooperative cancellation and safe robot stop. Repeated calls are idempotent.

### `GET /api/generate3d/flow/artifact/{flow_id}/{filename}`

Returns an artifact only after resolving and validating it inside that flow's artifact directory. Path traversal and unknown flow IDs are rejected.

No endpoint accepts a custom system prompt. Full prompts shown in the UI are generated by the server and read-only.

## Safety and Failure Handling

Before a real block starts, preflight requires:

- valid current camera source;
- both required objects detected in the fresh frame;
- calibrated robot coordinates for source and target;
- a connected and available robot;
- valid joint and gripper state;
- no other Generate 3D or Flow Run physical task owning execution;
- a saved, valid Rest Position when automatic return is enabled;
- workspace bounds only when `Use calibrated workspace safety` is checked.

The workspace-safety checkbox defaults to unchecked, matching 3D Simulation, but all inherent joint, timeout, and emergency-stop protections remain active.

Failure behavior:

- Never proceed to a later block after any unsuccessful terminal status.
- Preserve the phase, error code, human-readable message, inputs, outputs, and available artifacts.
- A timeout invokes safe stop and marks the block `failed`.
- A stop between blocks prevents the next capture.
- A stop during physical execution delegates to the task manager's safe-stop path, then marks the block `stopped` after the worker exits.
- Automatic return to Rest Position happens only after every block in a `Run All` succeeds. It does not run automatically after failure or stop; the existing manual Rest Position button remains available.

## Frontend State and Refresh Behavior

The browser keeps only presentation state such as the selected block and expanded panels. It polls the status endpoint while a flow is active and renders the server snapshot. A refresh retrieves the same persisted flow and artifacts. Client-side controls are disabled according to server capabilities, including active execution, missing detections, missing calibration, and terminal state.

The implementation should reuse existing Generate 3D rendering and control helpers where practical. Because the current frontend is monolithic, the work will add focused Flow Run functions and extract small shared helpers only when this reduces duplicated behavior; it will not introduce a new frontend framework.

## Testing Strategy

### Python unit and API tests

- Planner accepts a valid multi-step response and rejects empty, malformed, unsupported, or over-limit plans.
- State-machine transitions permit only defined edges.
- `Run All` advances after success and stops on failed/uncertain/stopped/interrupted.
- A block-only run does not advance.
- Every block captures and detects fresh inputs.
- Real verification maps invalid/ambiguous output to `uncertain`.
- Simulation verification uses simulated state.
- Deterministic height defaults remain pick `0 cm` and place target height plus `2 cm`.
- Cancellation is idempotent and safe-stop is invoked for active physical work.
- Persistence is atomic, restart marks active state interrupted, and no work resumes automatically.
- Artifact routing prevents path traversal.
- Hardware exclusion prevents concurrent real tasks.
- API validation and empty status payloads are stable.

### JavaScript tests

- New subtab navigation and initial state.
- Flowchart rendering for every phase/status.
- Selected-block details show inputs, outputs, images, evidence, and errors.
- Correct enable/disable behavior for Run All, Run This Block, Retry, and Stop.
- Polling updates progress and stops at terminal flow states.
- Read-only full Gemini prompts appear before flow generation.
- Workspace safety defaults unchecked and return-to-rest defaults checked.
- Existing 3D Simulation behavior remains unchanged.

### Full regression verification

```bash
uv run --with pytest --with-requirements requirements.txt pytest -q
for test_file in tests/*.js; do node "$test_file" || exit 1; done
```

Manual browser verification will cover responsive layout, block selection, image previews, path preview, refresh recovery, and both simulated success and stopped/failure states. Real-robot code paths will be verified with mocks unless suitable hardware is explicitly available.

## Acceptance Criteria

1. A long instruction can be converted into 1–10 ordered atomic pick/place blocks.
2. The user can see each block's current phase, input, output, and evidence and can select any block for details.
3. `Run All` captures and detects a fresh frame for each block, executes it, captures a verification frame, and advances only after success.
4. Failure or uncertainty stops the flow without skipping or moving to a later block.
5. The user can execute or retry one selected block without automatically running others.
6. Simulation verification is based on simulated state; real verification uses before/after camera evidence and Gemini.
7. Flow state and artifacts remain viewable after browser refresh and application restart; interrupted motion never resumes automatically.
8. All relevant 3D Simulation controls and diagnostics are available in Flow Run with the same defaults and semantics.
9. Existing Generate 3D and generic Run workflows continue to pass their regression tests.
10. The implementation remains only on `codex/generate3d-flow-run` until the user separately requests integration.
