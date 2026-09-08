# Generate 3D Simulation Mode Design

## Scope

Change only **Generate 3D > 3D Simulation**. The feature must support manually drawn objects without Gemini detection, execute the same pick-and-place plan in either simulation or on the real robot, animate the 3D robot during both modes, and improve real-robot waypoint timeout diagnostics. No VLA flow or other tab is changed.

## User interface

- `+ Add Object` becomes available after a top-view image is captured or uploaded, even when `Detect & Generate` has not run.
- Add a mutually exclusive execution-mode control beside the task controls:
  - `Simulation only` is selected by default.
  - `Use real robot` must be selected explicitly before physical motion is possible.
- Existing task instruction, workspace-safety checkbox, Run Task, Stop, and live status remain in the same panel.
- In both modes, every published waypoint updates the robot model in 3D Workspace.

## Manual object workflow

Drawing the first valid bounding box creates a manual detection scene on the server. The request contains the current image dimensions and the complete client-side object list. The server validates each box and name, derives its center pixel, predicts its 3D position and joints from the existing Generate 3D calibration, assigns a new detection ID, and records the current calibration revision.

Subsequent add, rename, move, resize, and delete operations continue through the existing detection-update flow. A manual scene is task-ready once it contains at least two valid, uniquely named objects and its current 3D scene has rendered. Gemini raw output is not required for task execution.

## Shared task planning and execution

The backend resolves source and target names and builds one ordered set of phase waypoints from the calibration model. Predicted poses, requested heights, optional calibrated-workspace enforcement, and joint limits are validated before execution begins.

- In `simulation` mode, no robot connection or hardware ownership is required. The task manager publishes interpolated waypoints on a timer; the browser's existing status polling applies those joints to the 3D robot.
- In `real` mode, the existing exclusive hardware ownership checks remain. The same phase targets are sent to the physical robot, and measured joints are published so 3D Workspace follows the real state.
- Stop interrupts either executor. Only real mode can call robot read/write functions.

## Real-robot timeout behavior

The current fixed two-second timeout for every interpolation waypoint is too short and reports no convergence detail. Real execution will use a bounded adaptive timeout based on the largest commanded joint delta, with a conservative minimum and maximum. Arm-motion phases check arm joints only; gripper-specific phases keep their existing grip behavior.

If convergence still fails, the error identifies the phase and each joint still outside tolerance, including target, measured value, and remaining error. The executor never treats a timeout as success or continues to the next waypoint.

## API behavior

- The task-start payload adds `execution_mode`, accepting only `simulation` or `real`; omitted values default to `simulation`.
- Robot connection and hardware-busy checks apply only to `real`.
- Task status includes the execution mode and published joints.
- A manual-scene endpoint creates/replaces the current Generate 3D detection state from image size plus objects.
- Existing detection IDs and calibration-revision checks protect both detected and manual scenes from stale edits.

## Safety

- Simulation is the default and cannot send hardware commands.
- Real mode retains calibrated-workspace enforcement, joint-limit validation, safe-height motion, operation ownership, and cancellation.
- Disabling calibrated-workspace safety continues to bypass only that boundary check.
- Invalid manual boxes, coordinates, names, image sizes, or predictions are rejected before task readiness.

## Verification

Automated tests cover manual scene creation without Gemini, editing the resulting scene, simulation without a connected robot or hardware calls, real-mode ownership and measured-state publication, default simulation payload/UI state, 3D animation updates, adaptive timeout success, and detailed timeout failure. The full Python and JavaScript suites must pass before commit and push.
