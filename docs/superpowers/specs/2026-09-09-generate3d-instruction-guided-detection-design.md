# Generate 3D Instruction-Guided Detection Design

## Goal

Require a task instruction before Generate 3D detection, return only instruction-relevant objects that Gemini can find, and automatically populate absolute Pick and Place Height controls from Gemini recommendations.

## User Experience

The Task Instruction input moves into the Top View Image control card, before Detect & Generate. Detect & Generate remains disabled until both an image and a non-empty instruction are available.

After detection:

- The object list contains only instruction-relevant objects Gemini found.
- A partial result is valid and visible. If only source or target is found, the user can inspect or edit that object, but Run Task remains disabled.
- When both roles are present, Run Task becomes available using the original instruction.
- Gemini's valid Pick and Place recommendations replace the current height values and refresh the path preview.
- Invalid or missing recommendations leave the existing values unchanged. Initial defaults remain Pick 0 cm and Place 5 cm.
- Users may edit either suggested height before running.

Capturing or uploading a new image clears detection state but preserves the instruction so the user can retry detection against another frame. Editing the instruction after detection invalidates the existing detection and requires Detect & Generate again, preventing a stale object-role mapping.

## Detection Request

The browser sends multipart form fields:

- `image`: the captured or uploaded image.
- `instruction`: required trimmed task text.
- `model`: optional existing Gemini model override.

The server returns HTTP 400 with `INSTRUCTION_REQUIRED` when instruction is empty. The frontend also prevents this request and presents an inline message.

## Gemini Prompt and Schema

The server builds the Gemini prompt; browser-supplied text is treated only as task data inside explicit delimiters, not as a replacement system prompt.

Gemini is asked to locate only visible objects needed by the instruction and return:

```json
{
  "objects": [
    {
      "name": "short visible name",
      "task_role": "source",
      "bbox": [100, 120, 220, 260],
      "color_hex": "#d32f2f",
      "shape_3d": "box",
      "estimated_size_cm": [4, 4, 2],
      "confidence": 0.94
    },
    {
      "name": "short visible name",
      "task_role": "target",
      "bbox": [420, 100, 580, 300],
      "color_hex": "#00897b",
      "shape_3d": "cylinder",
      "estimated_size_cm": [12, 12, 5],
      "confidence": 0.91
    }
  ],
  "recommended_pick_height_cm": 0,
  "recommended_place_height_cm": 5
}
```

Rules communicated to Gemini:

- Return at most one `source` and one `target`.
- Omit a role when its object is not visible or cannot be located confidently.
- Never return unrelated workspace objects.
- Heights are absolute centimeters from the calibrated floor.
- Pick Height is the gripper target height for grasping the source.
- Place Height is the gripper release height from the floor, with no implicit target-height or clearance addition.
- Return JSON only.

## Server Validation and Filtering

The server does not trust Gemini output directly.

For each candidate object it:

- Accepts only `task_role` values `source` and `target`.
- Keeps at most the first valid object per role.
- Applies existing bbox normalization, position prediction, size normalization, and joint prediction.
- Drops malformed or unpositionable candidates without failing other valid candidates.
- Returns zero, one, or two validated objects.
- Stores the original instruction and role-tagged objects in `g3d_detection_state`.

Recommended heights are independently parsed as finite numbers and clamped to 0–30 cm. Invalid or missing values are omitted from the response rather than replaced server-side, allowing the browser to preserve its current values.

## Task Resolution

Task start first checks for exactly one validated `source` and one validated `target` in the active detection state. When present, those roles determine source and target regardless of whether Gemini's display names exactly match the instruction text.

Legacy and manually edited detections without roles retain the existing exact-name resolver. This preserves manual object editing and compatibility with older clients.

If roles are incomplete, task start returns `OBJECT_MATCH_REQUIRED`. The message identifies which roles are missing.

Manual edits retain an existing object's `task_role`. New manually drawn objects have no role and therefore use the legacy name-based path unless roles are assigned by a future feature.

## Frontend State and Data Flow

1. User captures/uploads an image and enters instruction.
2. Detect & Generate becomes enabled.
3. Browser posts image and instruction.
4. Server builds the constrained prompt, validates Gemini JSON, saves role-tagged detections, and returns validated height recommendations.
5. Browser displays zero to two found objects.
6. Browser applies each valid recommendation to its corresponding height control.
7. Browser refreshes the 3D objects and selected motion path.
8. Run availability requires either a complete source/target role pair or a valid legacy two-name detection.

The instruction input is no longer auto-generated from detected names. It remains the user's task statement and is preserved through detection.

## Error Handling

- Empty instruction: frontend inline error and server `INSTRUCTION_REQUIRED`.
- Gemini finds nothing: successful detection with an empty list and a clear “No requested objects found” status.
- Gemini finds one role: successful partial detection with “Found source; target not found” or the inverse.
- Duplicate role candidates: server keeps the first valid candidate and drops later duplicates.
- Invalid height recommendation: omit it and keep the current UI value.
- Instruction changed after detection: invalidate detection ID, clear current objects/path, and require detection again.
- Existing Gemini transport and parse failures retain current error handling.

## Testing

Frontend tests cover:

- Instruction is positioned before Detect & Generate.
- Detection is disabled without instruction or image.
- Multipart request contains the trimmed instruction.
- Detection no longer overwrites the user's instruction.
- Valid recommendations populate Pick/Place Height and regenerate the preview.
- Missing recommendations preserve current values.
- Partial object results render while Run remains disabled.
- Editing instruction invalidates a completed detection.

Backend tests cover:

- Empty instruction is rejected before Gemini is called.
- Prompt contains the task inside explicit delimiters and requests only source/target roles.
- Unrelated and invalid-role candidates are filtered.
- Duplicate role candidates are limited to one per role.
- Zero- and one-object results succeed.
- Valid recommendations are returned as absolute, clamped heights.
- Invalid recommendations are omitted.
- Role-based task start resolves source and target without exact name matching.
- Incomplete role sets return `OBJECT_MATCH_REQUIRED`.
- Legacy role-free detection still uses exact-name resolution.

Full JavaScript and Python test suites must pass.

## Scope Boundaries

This change does not add open-vocabulary multi-step planning, role selection controls, automatic retries, a second Gemini request, obstacle avoidance, or height recommendations based on robot force feedback.

