# Verified VLA Execution Loop Design

## Goal

Replace the current assumption that a VLA step is complete after 100 control steps with a bounded execute-observe-verify loop. A plan advances only when Gemini verifies the task from a fresh camera image. After five unsuccessful execution cycles for one task, Gemini creates a new plan for only the remaining work.

## Execution State Machine

Each `vla_model` plan step starts with cycle number one. The selected model is loaded once into a persistent inference process. One cycle sends the configured number of control steps (default 100). At the cycle boundary the process stops sending new actions but keeps the model, robot connection, camera connection, and motor torque alive, holding the last commanded pose while Gemini verifies the task.

The inference process writes its latest configured-camera observation to a session snapshot and reports `waiting_for_verification` over its control channel. The server reads that snapshot without opening the camera a second time. It then asks Gemini to classify the requested task as one of:

- `success`: visible evidence shows the requested task is complete.
- `continue`: visible evidence shows the task is incomplete and another execution cycle is appropriate.
- `uncertain`: the image does not provide enough evidence; treat this like `continue`.

On `success`, the server tells the persistent process to stop, allowing a controlled disconnect, marks the current plan step complete, and advances to the next step. On `continue` or `uncertain`, the server sends `continue` to the same process, which immediately performs another configured action cycle without reloading the model or reconnecting hardware.

If the next plan step uses a different task or model, the old process is stopped cleanly before the new process starts. A verification wait has a 120-second safety timeout; if the server disappears or sends no command, inference disconnects the robot rather than holding torque indefinitely.

The Stop action cancels the active model process and the surrounding execution loop. A stopped run must never restart itself or advance the plan.

## Configurable Re-plan Threshold

If one task reaches the configured cycle limit (default five) without a verified success, execution stops repeating that task and requests a new plan from Gemini using:

- the original user instruction;
- a fresh scene image;
- the list of completed steps;
- the failed step, model, and five verification histories;
- the currently selected models and their exact training instructions;
- the active IK mode.

The re-plan prompt explicitly requires a plan for remaining work only and forbids repeating completed steps. The returned plan passes the same local model/task/IK validation as the initial plan. A valid replacement becomes the active remaining plan and its first step is ready to execute. An invalid replacement or Gemini API failure changes the run to `needs_human_review`; it does not resume the failed model automatically.

Each newly planned task receives its own independent cycle budget. To prevent an endless sequence of re-plans, one Run session permits the configured maximum number of automatic re-plans (default three). Reaching that limit changes the run to `needs_human_review`.

## Config & Test Settings

The `Config & Test` page adds an **Execution Loop** settings group with three numeric fields:

- **Actions per cycle**: default `100`, allowed range `1–1000`.
- **Cycles before re-plan**: default `5`, allowed range `1–20`.
- **Max automatic re-plans**: default `3`, allowed range `1–10`.

These values are stored in `config.yaml` under `execution_loop.actions_per_cycle`, `execution_loop.cycles_before_replan`, and `execution_loop.max_replans`. The existing config read/save API returns and validates all three fields. Missing settings migrate at read time to the defaults without rewriting the file until the user explicitly saves Config & Test.

The Run page displays the active limits in its status text, such as `Executing cycle 2/5 · 100 actions`. A Run session snapshots the three settings when it starts, so editing Config & Test does not change an already-running robot operation mid-cycle.

## API and UI

The server owns orchestration so execution continues reliably even if browser polling pauses. Run-session state includes the original instruction, active plan, completed steps, current step, cycle number, verification history, re-plan count, state, and Stop flag.

The Run UI displays the current execution phase, for example `Executing cycle 2/5`, `Verifying`, `Re-planning remaining work`, `Completed`, or `Needs human review`. Logs remain visible. The existing Run Step button starts the verified loop for the selected VLA step; Stop cancels the full loop. When a replacement plan is accepted, the flow chart updates without re-inserting completed steps.

## Completion Verification

Gemini receives the task text and the latest observation captured by the persistent inference process at the cycle boundary. The verification prompt requires JSON with `status`, `reason`, and `visible_evidence`. Responses outside the allowed schema are classified as `uncertain`, retained in history, and consume one cycle. Verification is conservative: absence of clear visible evidence cannot produce `success`.

For pick-and-place, success requires visible evidence that the target object has been released at the requested destination; merely holding the object above or near the destination is not success.

## Safety and Failure Handling

- The configured action limit is enforced for every cycle; no unbounded inference process is introduced.
- The configured cycle and automatic re-plan limits are enforced for every Run session.
- Process failure, camera capture failure, invalid re-plan, or API failure results in `needs_human_review` with a specific error.
- Hardware ownership remains exclusively inside the persistent inference process while a task is active; the server verifies the exported observation snapshot.
- Motor torque remains enabled only while waiting for verification and is released on success, Stop, process failure, task/model change, or the 120-second command timeout.
- Stop is checked before every launch, verification, retry, and re-plan transition.
- The existing manual Stop endpoint remains idempotent.

## Tests

- Unit tests cover state transitions for success, continue, uncertain, configured-cycle re-plan, re-plan limit, Stop, and failures.
- Config tests cover defaults, persistence, range validation, and per-session setting snapshots.
- API tests cover starting, polling, stopping, fresh-image verification, accepted replacement plans, rejected replacement plans, and completed-step context.
- Process tests retain the exact 100-step command bound.
- Process tests cover cycle-boundary pause, snapshot publication, `continue`, controlled `stop`, model reuse, and verification-command timeout.
- UI tests cover phase labels, updated plans, automatic advancement, `needs_human_review`, and Stop behavior.
- A no-hardware integration test uses fake process, camera, and Gemini adapters to exercise the complete cycle and re-plan flow deterministically.
