import main


def test_pick_place_completion_prompt_prioritizes_requested_final_state():
    prompt = main._verified_completion_prompt({
        "description": "pick up the bow to the green bowl",
        "method_id": "vla_model",
        "model_id": "model_a",
    })

    assert "inside or on the named destination" in prompt
    assert "does not require seeing the earlier pickup" in prompt
    assert "arm is above or near the destination" in prompt
    assert "status, reason, and visible_evidence must agree" in prompt


def test_pick_place_completion_prompt_distinguishes_continue_from_uncertain():
    prompt = main._verified_completion_prompt({"description": "move cube to green bowl"})

    assert "continue only when" in prompt
    assert "uncertain when" in prompt
    assert "specific named destination" in prompt
