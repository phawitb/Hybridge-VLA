from planner_config import (
    DEFAULT_NO_IK_PROMPT,
    DEFAULT_USE_IK_PROMPT,
    planner_settings,
    render_available_models,
    render_planner_prompt,
    validate_plan,
)


def model_record(model_id: str, tasks: list[str], selectable: bool = True) -> dict:
    return {
        "id": model_id,
        "display_name": model_id,
        "policy_type": "smolvla",
        "downloaded": True,
        "local_path": f"models/{model_id}",
        "dataset_id": f"phawitbinabik/{model_id}_dataset",
        "tasks": tasks,
        "camera_features": ["observation.images.top"],
        "selectable": selectable,
        "unavailable_reason": None if selectable else "Unavailable",
    }


def vla_step(index: int, model_id: str, description: str) -> dict:
    return {
        "step_index": index,
        "description": description,
        "target_bbox": None,
        "method_id": "vla_model",
        "model_id": model_id,
    }


def ik_step(index: int) -> dict:
    return {
        "step_index": index,
        "description": "Reach the bow",
        "target_bbox": [0.5, 0.5, 0.2, 0.2],
        "method_id": "ik_reach_object_v1",
        "model_id": None,
    }


def test_planner_settings_migrates_legacy_config_to_use_ik():
    settings = planner_settings({
        "available_methods": ["smolVLA_v1"],
        "prompt_template": "legacy",
    })

    assert settings["use_ik"] is True
    assert settings["selected_models"] == []
    assert settings["prompt_templates"]["use_ik"] == DEFAULT_USE_IK_PROMPT
    assert settings["prompt_templates"]["no_ik"] == DEFAULT_NO_IK_PROMPT


def test_planner_settings_replaces_saved_template_without_steps_schema():
    settings = planner_settings({
        "planner": {
            "use_ik": False,
            "prompt_templates": {
                "no_ik": "Instruction {instruction}; models {available_models}; each step has fields",
            },
        }
    })

    assert settings["prompt_templates"]["no_ik"] == DEFAULT_NO_IK_PROMPT


def test_planner_settings_preserves_both_schema_complete_custom_templates():
    use_ik = 'IK {instruction} {available_models} {"steps": []}'
    no_ik = 'NO IK {instruction} {available_models} {"steps": []}'
    settings = planner_settings({
        "planner": {
            "use_ik": False,
            "selected_models": ["model_a"],
            "prompt_templates": {"use_ik": use_ik, "no_ik": no_ik},
        }
    })

    assert settings == {
        "use_ik": False,
        "selected_models": ["model_a"],
        "prompt_templates": {"use_ik": use_ik, "no_ik": no_ik},
    }


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


def test_render_planner_prompt_selects_no_ik_template():
    config = {
        "planner": {
            "use_ik": False,
            "selected_models": ["model_a"],
            "prompt_templates": {
                "use_ik": 'USE {instruction} {available_models} {"steps": []}',
                "no_ik": 'NO {instruction} {available_models} {"steps": []}',
            },
        }
    }

    prompt, selected = render_planner_prompt(
        config, "pick up the bow", [model_record("model_a", ["pick up the bow"])]
    )

    assert prompt.startswith("NO pick up the bow")
    assert "pick up the bow" in prompt
    assert selected == ["model_a"]


def test_validate_plan_requires_ik_before_vla_in_use_ik_mode():
    plan = {"steps": [vla_step(1, "model_a", "pick up the bow")]}

    errors = validate_plan(plan, True, [model_record("model_a", ["pick up the bow"])])

    assert errors == [{
        "code": "IK_REQUIRED",
        "message": "VLA step 1 must be immediately preceded by an IK step",
        "step_index": 1,
    }]


def test_validate_plan_accepts_ik_then_matching_vla():
    plan = {"steps": [ik_step(1), vla_step(2, "model_a", "pick up the bow")]}

    assert validate_plan(plan, True, [model_record("model_a", ["pick up the bow"])]) == []


def test_validate_plan_rejects_ik_in_no_ik_mode():
    errors = validate_plan(
        {"steps": [ik_step(1)]}, False, [model_record("model_a", ["pick up the bow"])]
    )

    assert errors[0]["code"] == "IK_FORBIDDEN"


def test_validate_plan_rejects_unselected_model():
    errors = validate_plan(
        {"steps": [vla_step(1, "model_b", "open the drawer")]},
        False,
        [model_record("model_a", ["pick up the bow"])],
    )

    assert errors[0]["code"] == "MODEL_NOT_SELECTED"


def test_validate_plan_rejects_task_not_declared_by_model():
    errors = validate_plan(
        {"steps": [vla_step(1, "model_a", "open the drawer")]},
        False,
        [model_record("model_a", ["pick up the bow"])],
    )

    assert errors[0]["code"] == "TASK_NOT_SUPPORTED"


def test_validate_plan_rejects_unknown_method():
    errors = validate_plan(
        {"steps": [{"step_index": 1, "method_id": "invented", "description": "x"}]},
        False,
        [model_record("model_a", ["pick up the bow"])],
    )

    assert errors[0]["code"] == "METHOD_NOT_SUPPORTED"
