# Verified VLA Execution Loop Design

## Goal

Replace the current assumption that a VLA step is complete after 100 control steps with a bounded execute-observe-verify loop. A plan advances only when Gemini verifies the task from a fresh camera image. After five unsuccessful execution cycles for one task, Gemini creates a new plan for only the remaining work.

## Execution State Machine

Each `vla_model` plan step starts with cycle number one. One cycle launches the selected model for exactly 100 control steps. When the inference process exits successfully, the server reconnects only the cameras needed for verification, captures a fresh frame, and asks Gemini to classify the requested task as one of:

- `success`: visible evidence shows the requested task is complete.
- `continue`: visible evidence shows the task is incomplete and another execution cycle is appropriate.
- `uncertain`: the image does not provide enough evidence; treat this like `continue`.

On `success`, the current plan step is marked complete and execution advances to the next step. On `continue` or `uncertain`, the same model and exact training-task instruction run for another 100 steps. The model process is disconnected between cycles so camera ownership can safely transfer to the verifier.

The Stop action cancels the active model process and the surrounding execution loop. A stopped run must never restart itself or advance the plan.

## Five-Cycle Re-plan

If one task reaches five cycles without a verified success, execution stops repeating that task and requests a new plan from Gemini using:

- the original user instruction;
- a fresh scene image;
- the list of completed steps;
- the failed step, model, and five verification histories;
- the currently selected models and their exact training instructions;
- the active IK mode.

The re-plan prompt explicitly requires a plan for remaining work only and forbids repeating completed steps. The returned plan passes the same local model/task/IK validation as the initial plan. A valid replacement becomes the active remaining plan and its first step is ready to execute. An invalid replacement or Gemini API failure changes the run to `needs_human_review`; it does not resume the failed model automatically.

Each newly planned task receives its own independent five-cycle budget. To prevent an endless sequence of re-plans, one Run session permits at most three automatic re-plans. Reaching that limit changes the run to `needs_human_review`.

## API and UI

The server owns orchestration so execution continues reliably even if browser polling pauses. Run-session state includes the original instruction, active plan, completed steps, current step, cycle number, verification history, re-plan count, state, and Stop flag.

The Run UI displays the current execution phase, for example `Executing cycle 2/5`, `Verifying`, `Re-planning remaining work`, `Completed`, or `Needs human review`. Logs remain visible. The existing Run Step button starts the verified loop for the selected VLA step; Stop cancels the full loop. When a replacement plan is accepted, the flow chart updates without re-inserting completed steps.

## Completion Verification

Gemini receives the task text and a fresh image from the model's configured camera. The verification prompt requires JSON with `status`, `reason`, and `visible_evidence`. Responses outside the allowed schema are classified as `uncertain`, retained in history, and consume one cycle. Verification is conservative: absence of clear visible evidence cannot produce `success`.

For pick-and-place, success requires visible evidence that the target object has been released at the requested destination; merely holding the object above or near the destination is not success.

## Safety and Failure Handling

- Exactly 100 control steps are allowed per cycle; no unbounded inference process is introduced.
- Maximum five cycles per task and three automatic re-plans per Run session.
- Process failure, camera capture failure, invalid re-plan, or API failure results in `needs_human_review` with a specific error.
- Hardware ownership remains exclusive between inference and verification camera capture.
- Stop is checked before every launch, verification, retry, and re-plan transition.
- The existing manual Stop endpoint remains idempotent.

## Tests

- Unit tests cover state transitions for success, continue, uncertain, five-cycle re-plan, re-plan limit, Stop, and failures.
- API tests cover starting, polling, stopping, fresh-image verification, accepted replacement plans, rejected replacement plans, and completed-step context.
- Process tests retain the exact 100-step command bound.
- UI tests cover phase labels, updated plans, automatic advancement, `needs_human_review`, and Stop behavior.
- A no-hardware integration test uses fake process, camera, and Gemini adapters to exercise the complete cycle and re-plan flow deterministically.
