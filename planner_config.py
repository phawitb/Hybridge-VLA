"""Planner configuration, prompt rendering, and executable-plan validation."""

from __future__ import annotations

import json


DEFAULT_USE_IK_PROMPT = """You are a robotic task planner for a single-arm robot.

## Instruction
{instruction}

## Selected VLA Models and Exact Training Tasks
{available_models}

## Planning Rules
- Output only valid JSON using the schema below.
- Use only `ik_reach_object_v1` and `vla_model`.
- Every `vla_model` step MUST be immediately preceded by one `ik_reach_object_v1` step targeting the same interaction location.
- An IK step only moves the arm. It does not grasp, release, open, close, push, or pull.
- For each VLA step, choose exactly one selected model and set `model_id` to its exact ID.
- The VLA description MUST exactly match one training task declared for that model.
- Never invent a model ID or assign a task not listed for that model.
- Use normalized YOLO bounding boxes [x_center, y_center, width, height] for IK targets.
- If no selected model supports a required action, output {"error":"NO_CAPABLE_MODEL","reason":"brief explanation"}.

## Output Schema
{"task_id":"descriptive_id","total_steps":2,"steps":[{"step_index":1,"description":"Reach the target","target_bbox":[0.5,0.5,0.2,0.2],"method_id":"ik_reach_object_v1","model_id":null},{"step_index":2,"description":"exact training task","target_bbox":null,"method_id":"vla_model","model_id":"selected_model_id"}]}"""


DEFAULT_NO_IK_PROMPT = """You are a robotic task planner for end-to-end VLA policies.

## Instruction
{instruction}

## Selected VLA Models and Exact Training Tasks
{available_models}

## Planning Rules
- Output only valid JSON using the schema below.
- NEVER emit `ik_reach_object_v1`, reach-only steps, or hand-written motion primitives.
- Every step MUST use `method_id` equal to `vla_model`.
- Treat each step as one complete end-to-end task that a selected model was trained to perform.
- Set `model_id` to the exact selected model ID.
- The step description MUST exactly match one training task declared for that model.
- Prefer a single complete trained task instead of decomposing it.
- Never invent a model ID or assign a task not listed for that model.
- If no selected model supports the instruction, output {"error":"NO_CAPABLE_MODEL","reason":"brief explanation"}.

## Output Schema
{"task_id":"descriptive_id","total_steps":1,"steps":[{"step_index":1,"description":"exact training task","target_bbox":null,"method_id":"vla_model","model_id":"selected_model_id"}]}"""


def planner_settings(config: dict) -> dict:
    planner = config.get("planner")
    if not isinstance(planner, dict):
        planner = {}
    prompts = planner.get("prompt_templates")
    if not isinstance(prompts, dict):
        prompts = {}
    selected = planner.get("selected_models", [])
    if not isinstance(selected, list):
        selected = []
    return {
        "use_ik": bool(planner.get("use_ik", True)),
        "selected_models": list(dict.fromkeys(str(item) for item in selected if str(item).strip())),
        "prompt_templates": {
            "use_ik": _prompt_with_schema(prompts.get("use_ik"), DEFAULT_USE_IK_PROMPT),
            "no_ik": _prompt_with_schema(prompts.get("no_ik"), DEFAULT_NO_IK_PROMPT),
        },
    }


def _prompt_with_schema(value: object, fallback: str) -> str:
    prompt = str(value or "")
    required = ("{instruction}", "{available_models}", '"steps"')
    return prompt if all(marker in prompt for marker in required) else fallback


def _selected_records(records: list[dict], selected_ids: list[str]) -> list[dict]:
    by_id = {record.get("id"): record for record in records}
    return [by_id[model_id] for model_id in selected_ids if model_id in by_id]


def render_available_models(records: list[dict], selected_ids: list[str]) -> str:
    sections = []
    for record in _selected_records(records, selected_ids):
        tasks = "\n".join(f"  - {task}" for task in record.get("tasks", [])) or "  - No declared tasks"
        cameras = ", ".join(record.get("camera_features", [])) or "none"
        sections.append(
            f"MODEL_ID: {record['id']}\n"
            f"POLICY_TYPE: {record.get('policy_type') or 'unknown'}\n"
            f"CAMERAS: {cameras}\n"
            f"TRAINING_TASKS:\n{tasks}"
        )
    return "\n\n".join(sections)


def render_planner_prompt(config: dict, instruction: str, records: list[dict]) -> tuple[str, list[str]]:
    settings = planner_settings(config)
    selected = _selected_records(records, settings["selected_models"])
    key = "use_ik" if settings["use_ik"] else "no_ik"
    prompt = settings["prompt_templates"][key]
    prompt = prompt.replace("{instruction}", instruction)
    prompt = prompt.replace(
        "{available_models}", render_available_models(records, settings["selected_models"])
    )
    return prompt, [record["id"] for record in selected]


def _error(code: str, message: str, step_index: int | None) -> dict:
    return {"code": code, "message": message, "step_index": step_index}


def _normalize_task(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def validate_plan(plan: dict, use_ik: bool, selected_records: list[dict]) -> list[dict]:
    errors = []
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    if not isinstance(steps, list):
        return [_error("INVALID_STEPS", "Plan steps must be a list", None)]
    selected = {record.get("id"): record for record in selected_records}
    for position, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(_error("INVALID_STEP", "Each plan step must be an object", None))
            continue
        index = step.get("step_index", position + 1)
        method = step.get("method_id")
        if method == "ik_reach_object_v1":
            if not use_ik:
                errors.append(_error("IK_FORBIDDEN", f"IK step {index} is forbidden in Not use IK mode", index))
            bbox = step.get("target_bbox")
            if use_ik and (
                not isinstance(bbox, list) or len(bbox) != 4
                or any(not isinstance(value, (int, float)) or value < 0 or value > 1 for value in bbox)
            ):
                errors.append(_error("INVALID_IK_BBOX", f"IK step {index} requires a normalized bbox", index))
            continue
        if method != "vla_model":
            errors.append(_error("METHOD_NOT_SUPPORTED", f"Unsupported method on step {index}: {method}", index))
            continue
        model_id = step.get("model_id")
        record = selected.get(model_id)
        if record is None:
            errors.append(_error("MODEL_NOT_SELECTED", f"Model on step {index} is not selected: {model_id}", index))
            continue
        declared = {_normalize_task(task) for task in record.get("tasks", [])}
        if _normalize_task(step.get("description")) not in declared:
            errors.append(_error(
                "TASK_NOT_SUPPORTED",
                f"Task on step {index} is not declared for model {model_id}",
                index,
            ))
        if use_ik:
            previous = steps[position - 1] if position else None
            if not isinstance(previous, dict) or previous.get("method_id") != "ik_reach_object_v1":
                errors.append(_error(
                    "IK_REQUIRED",
                    f"VLA step {index} must be immediately preceded by an IK step",
                    index,
                ))
    return errors


def validation_feedback(errors: list[dict]) -> str:
    return json.dumps({"validation_errors": errors}, ensure_ascii=False)
