# Persistent VLA Worker and Outcome Verification Design

## Problem

The verified Run flow currently launches `infer_python.py` with one fixed task. The subprocess loads the VLA checkpoint, connects to the robot and cameras, executes one or more action cycles, and exits when the step succeeds. This causes three problems:

1. The first executable step begins before the selected VLA checkpoint is guaranteed to be loaded.
2. Switching from VLA execution to IK requires stopping the subprocess, which unloads the checkpoint and makes the next VLA step reload it.
3. A new task using the same model cannot reuse the process safely because the task text and SmolVLA language cache are fixed at startup.

The completion verifier also misclassifies some successful pick-and-place outcomes. In the observed failure, it reported that the bow was visibly in the requested bowl but returned `continue` because the arm remained above the bowl.

## Goals

- Load the first required VLA model completely before step 1 starts.
- Keep the selected model and preprocessing objects in memory across IK steps, VLA steps, and completed Run sessions.
- Ensure only one process owns the robot serial port and cameras at a time.
- Reuse the warm worker for a different declared task when the model is unchanged.
- Reload only when the requested model changes, the user explicitly stops the worker, the server shuts down, or the worker fails.
- Make completion decisions follow the visible requested final state.
- Preserve existing action-cycle verification and re-planning behavior.

## Non-goals

- Keeping multiple VLA checkpoints loaded simultaneously.
- Sharing an active robot or camera connection between processes.
- Changing model training-task capability validation.
- Replacing the subprocess architecture with in-process inference.
- Inferring success from hidden history when the final frame is ambiguous.

## Architecture

`infer_python.py` becomes a persistent, command-driven VLA worker. It loads one checkpoint and its pre/post processors, emits a readiness event, then waits for newline-delimited JSON commands on stdin. The model remains resident while hardware connections are opened and closed around execution.

`VlaProcessManager` becomes the owner of the worker protocol. It tracks model readiness, hardware state, active task, action-cycle state, and protocol acknowledgements. It provides synchronous, timeout-bounded operations to preload a model, run or continue a task, release hardware, and stop the worker.

`VerifiedExecutionManager` gains a preparation phase that runs before the first plan step. The Run API preloads the first VLA model and does not begin the step loop until `MODEL_READY` is observed. Step execution then performs explicit hardware handoffs through `VlaProcessManager`.

Only one warm VLA worker exists. A request for the same model reuses it. A request for a different model stops the old worker after releasing hardware, starts a new worker, and waits for readiness.

## Worker Protocol

Worker output uses one-line events with a stable prefix and JSON payload:

- `MODEL_READY {"model_id":"..."}`: checkpoint and processors are ready; hardware is not connected.
- `HARDWARE_READY {"task":"..."}`: robot and cameras are connected for the active task.
- `CYCLE_READY {"step":100,"image":"..."}`: one action cycle finished and verification may run.
- `HARDWARE_RELEASED {"task":"..."}`: robot and cameras are disconnected while the model remains loaded.
- `WORKER_ERROR {"code":"...","error":"..."}`: command or runtime failure.

Controller input is newline-delimited JSON:

- `{"command":"run_task","task":"...","max_steps":100,"verification_image":"...","verification_timeout":120}`
- `{"command":"continue"}`
- `{"command":"release_hardware"}`
- `{"command":"shutdown"}`

Malformed or invalid-state commands produce `WORKER_ERROR` and do not silently change ownership.

## Worker Lifecycle

### Startup and preload

The worker loads model configuration, policy weights, and pre/post processors before touching the robot or cameras. It then emits `MODEL_READY` and waits. A startup timeout or early process exit fails the Run session before any plan step moves the robot.

### Starting a VLA task

On `run_task`, the worker:

1. Verifies that hardware is currently released.
2. Resets the policy action queue.
3. Replaces the active task text.
4. Invalidates all cached language tokens, masks, and embeddings.
5. Connects the configured robot and cameras.
6. Emits `HARDWARE_READY`.
7. Starts the action loop.

Changing task text with the same model therefore reuses weights and processors but never reuses task-specific actions or language state.

### Verification boundary

At an action-cycle boundary, the worker saves the verification image, emits `CYCLE_READY`, and stops sending robot actions. A `continue` command resumes the same task and cycle counter. A `release_hardware` command disconnects robot and cameras, emits `HARDWARE_RELEASED`, and returns to the ready state without unloading the model.

### IK handoff

Before every IK step, the controller requests `release_hardware` when necessary and waits for `HARDWARE_RELEASED`. Only then may the main process connect to the robot and execute IK. Before the next VLA task, the main process disconnects its robot connection, then sends `run_task` to the warm worker.

At no time may the main process and worker both report hardware ownership. A missing acknowledgement or timeout stops automatic execution and enters human review.

### Completion and shutdown

When a VLA step succeeds, the controller releases worker hardware instead of stopping the process. On normal session completion, worker hardware remains released and the model remains warm for a later Run session.

The worker is fully stopped when:

- the user explicitly presses Stop;
- a different model must be loaded;
- the worker exits or enters an unrecoverable protocol state;
- the server shuts down.

## Verified Run Flow

1. Validate the plan against selected models and IK mode.
2. Find the first VLA step at or after the selected start index.
3. Preload its model and wait for `MODEL_READY`.
4. Begin step 1 only after preload succeeds.
5. For IK, obtain an acknowledged hardware release, connect in the main process, move, then retain or release the main connection according to the next step.
6. For VLA, disconnect the main robot connection, activate the warm worker task, and execute an action cycle.
7. Verify the resulting frame.
8. On `continue`, resume the same task without reconnecting or reloading.
9. On `success`, release worker hardware and advance.
10. On re-plan, reuse the worker when the replacement plan uses the same model; otherwise perform a controlled model switch.

The status API exposes preparation and handoff phases such as `loading_model`, `model_ready`, `releasing_hardware`, and `connecting_vla_hardware`. The UI displays these phases so model-loading latency is not mistaken for a stalled first step.

## Completion Verification Prompt

The verifier must derive the requested final spatial state from the complete VLA task description and judge the current image against that state.

For pick-and-place tasks:

- Return `success` when the named object is visibly inside or on the named destination and is not visibly held by the gripper.
- A successful final state does not require visual evidence of the earlier pickup action.
- The arm being above, near, or moving away from the destination does not override a visibly correct released-object state.
- Return `continue` only when the object is clearly outside the requested destination or clearly still held.
- Return `uncertain` when object identity, destination identity, containment, or release state cannot be determined reliably.
- A generic statement such as “in a bowl” is insufficient when the task names a specific bowl; the evidence must identify the requested destination.
- `status`, `reason`, and `visible_evidence` must be internally consistent. Evidence that directly satisfies the requested final state cannot accompany `continue`.

The response remains strict JSON with `status`, `reason`, and `visible_evidence`.

## Error Handling

- Model load failure: fail before step execution and retain no hardware connection.
- Worker startup timeout: terminate the worker and enter human review.
- Hardware-release timeout: stop automatic execution; do not let the main process connect speculatively.
- Robot/camera connection failure: release any partially acquired resources and report the failing boundary.
- Worker death: clear readiness and ownership state; require a fresh preload.
- Model switch failure: do not fall back to the previous model for a task it was not selected to perform.
- Invalid control message: emit a protocol error and preserve the last known safe hardware state.

## Testing

### Completion prompt

- A prompt regression test requires final-state dominance for an object visibly released in the named destination.
- Tests cover `success`, `continue`, and `uncertain` decision boundaries in the prompt contract.

### Worker protocol

- Model readiness is emitted before any hardware connection.
- The same model is not relaunched for repeated preload requests.
- A different model stops the old worker and waits for the new worker.
- `release_hardware` disconnects robot/cameras while keeping the process alive.
- A new task resets the action queue and language caches.
- Continuing the same task retains its process, task state, and hardware connection.
- Invalid commands and timeouts produce safe failures.

### Verified execution

- Step 1 cannot execute before model readiness.
- IK waits for worker hardware release.
- VLA waits for the main-process robot disconnect.
- A successful VLA step releases hardware without unloading the model.
- IK-to-VLA and VLA-to-IK sequences reuse the same worker PID for the same model.
- Re-plans reuse the worker for the same model and reload for a changed model.
- Stop and server shutdown terminate the worker and release hardware.

The full existing test suite must remain green. Tests use fake subprocesses and fake hardware boundaries; they do not require a physical robot or model checkpoint.

## Acceptance Criteria

- The Run UI does not begin step 1 until the first needed VLA model is ready.
- A plan alternating IK and the same VLA model loads that model exactly once.
- During IK, the VLA worker remains alive but owns neither robot nor cameras.
- A second task using the same model runs without process restart and without stale language/action state.
- Hardware ownership is acknowledged at every process boundary and never overlaps.
- The observed bow-in-green-bowl frame is classified as `success` when the bow is visibly released in the green bowl.
- Existing planning, verification-cycle, re-planning, manual Stop, and safety behavior continue to work.
