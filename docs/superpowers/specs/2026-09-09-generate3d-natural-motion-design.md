# Generate 3D Natural Motion Design

## Goal

Add a selectable motion style to Generate 3D so the 3D preview and the real robot can use either the existing segmented waypoint motion or a smoother, human-like pick-and-place trajectory. Smooth Natural is the default. Safety constraints and the existing Waypoint behavior remain available.

## User Experience

The Generate 3D > 3D Simulation task controls add a motion-style radio group:

- **Smooth Natural** (default): continuous eased motion with vertical approach and departure near objects and a smooth transfer arc.
- **Waypoint**: the current raise, translate, and descend behavior.

The existing Pick/Place Height and Safety/Transfer Height inputs apply to both modes. Changing the instruction, motion mode, or either height refreshes the path preview immediately.

The preview uses the selected mode:

- Waypoint renders the existing segmented path.
- Smooth Natural renders a sampled curve with visible Pick, Arc peak, and Place markers.

The selected mode is included in the task request as `motion_mode`, using `smooth` or `waypoint`. The server defaults a missing value to `smooth` and rejects unknown values before acquiring the robot.

## Motion Model

### Shared anchors

Both modes derive the same semantic anchors from the resolved source, target, object dimensions, requested heights, and calibration snapshot:

1. Source approach above the source.
2. Source pick at Pick/Place Height.
3. Source departure at or above Safety/Transfer Height.
4. Target approach at or above Safety/Transfer Height.
5. Target release above the target using the existing target-height clearance rule.
6. Final departure above the target.

The transfer height is never lower than the release height required to clear the target object.

### Waypoint mode

Waypoint mode keeps the current plan and executor unchanged. The robot converges at the end of each existing phase. This provides a predictable fallback and protects existing workflows.

### Smooth Natural mode

Smooth mode builds a Cartesian-style trajectory in image/height space because the current calibration already maps a pixel and height to robot joints. It then predicts and validates joint targets for sampled positions along the curve.

The trajectory has three motion characteristics:

- Vertical approach and departure zones near the source and target prevent sideways contact with objects.
- Cubic smoothstep easing produces zero-slope starts and stops for lift and descent.
- The transfer uses a smooth cubic curve whose height remains at or above the configured safety height, with a modest rounded crown rather than a sharp corner.

The planner samples the curve densely enough for continuous servo commands. Every sample is converted through the captured calibration snapshot and validated against the existing joint limits before execution begins. Planning is all-or-nothing: one invalid sample rejects the task without moving hardware.

## Plan Representation and Execution

The existing semantic phases remain visible in task status. Smooth movement phases carry a precomputed trajectory—a sequence of validated joint targets—instead of only one final joint target.

The real executor streams intermediate trajectory targets at the existing bounded command cadence. It does not wait for full convergence at each intermediate sample, which would recreate stop-and-go motion. It performs measured convergence only at safety-critical final anchors such as pick, release, and the end of lift/approach phases.

Simulation consumes the same trajectory samples and publishes them through the existing task status channel. This keeps simulated joint movement consistent with real execution. The frontend preview is derived from the same curve rules and semantic anchors; it is a workspace visualization rather than a claim of millimeter-perfect forward kinematics.

The existing loaded-lift shoulder tolerance remains scoped to lifting phases. Other arm joints and gripper convergence retain their stricter limits.

## Components

### Frontend

- Add the motion-style radio group and default Smooth Natural selection.
- Include `motion_mode` in `g3dBuildTaskPayload`.
- Extend path-point generation to produce either segmented waypoint points or sampled smooth-curve points.
- Re-render the path when the mode, instruction, or height changes.
- Keep path visuals separate from detected-object visuals so either can refresh independently.

### Backend planner

- Validate and normalize `motion_mode` at the API boundary.
- Preserve `_g3d_build_pick_place_plan` behavior for Waypoint mode.
- Add a focused smooth-trajectory builder that creates anchors, samples curves, predicts joints, and validates the complete trajectory.
- Keep source/target resolution, workspace enforcement, calibration snapshots, joint limits, clearance, and hardware ownership shared between modes.

### Executors

- Extend real and simulation execution to consume trajectory-bearing phases.
- Stream intermediate samples without per-sample convergence waits.
- Retain stop-event checks throughout streaming.
- Require convergence at the final target of each safety-critical phase and report existing detailed residual errors on failure.

## Safety and Error Handling

- No hardware command is sent until every trajectory sample has been predicted and validated.
- Smooth trajectories never dip below the effective transfer height between source departure and target approach.
- Approach and release remain vertical near objects.
- Existing calibrated-workspace enforcement remains unchanged.
- Joint limits apply to every generated sample.
- Stop requests interrupt both streamed and converging motion promptly.
- Invalid `motion_mode` returns `INVALID_MOTION_MODE`.
- An unreachable curve returns a planning error that identifies the affected motion phase.
- Waypoint remains available as a user-selectable fallback.

## Testing

Frontend tests verify:

- Smooth Natural is selected by default.
- Both radio controls exist and payloads contain the selected mode.
- Smooth preview points start at Pick, finish at Place, retain vertical near-object segments, and never fall below the effective transfer height during transfer.
- Waypoint preview preserves the segmented path.
- Changing mode or height triggers path regeneration.

Backend tests verify:

- Missing mode defaults to Smooth Natural and both supported values are accepted.
- Unknown modes are rejected before hardware acquisition.
- Waypoint mode preserves the existing plan.
- Smooth samples use easing, preserve vertical approach/departure, and respect effective safety height.
- Every sample is joint-limit validated before execution.
- A failed sample prevents all hardware writes.
- Real execution streams intermediate samples and converges at required final anchors.
- Simulation publishes the same planned joint sequence.
- Stop requests work during trajectory streaming.
- Loaded-lift shoulder residuals up to 4 degrees are accepted only where configured, while values over 4 degrees still fail.

Full JavaScript and Python suites must pass after implementation.

## Scope Boundaries

This change does not add obstacle avoidance, dynamic replanning, collision sensing, force control, or learned human-motion imitation. “Natural” means deterministic smooth curves and easing within the existing calibrated pick-and-place workflow.
