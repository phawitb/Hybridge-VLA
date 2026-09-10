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


def test_success_releases_worker_hardware_without_stopping_model(monkeypatch):
    events = []
    monkeypatch.setattr(main, "_capture_verified_frame", lambda step: (b"jpeg", "image/jpeg"))
    monkeypatch.setattr(main, "_gemini_image_json", lambda prompt, image, mime: {
        "raw": '{"status":"success","reason":"bow released","visible_evidence":"bow inside green bowl"}'
    })
    monkeypatch.setattr(main.vla_manager, "release_hardware", lambda timeout=10.0: events.append("release") or {"ok": True})
    monkeypatch.setattr(main.vla_manager, "stop", lambda: events.append("stop") or {"ok": True})

    result = main._verified_completion({"description": "pick up the bow to the green bowl"})

    assert result["status"] == "success"
    assert events == ["release"]
