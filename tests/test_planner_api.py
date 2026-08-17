from copy import deepcopy
import io
import json

import yaml
from fastapi.testclient import TestClient
from PIL import Image

import main


SELECTABLE_MODEL = {
    "id": "model_a",
    "repo_id": "phawitbinabik/model_a",
    "display_name": "model_a",
    "policy_type": "smolvla",
    "downloaded": True,
    "local_path": "models/model_a",
    "dataset_id": "phawitbinabik/dataset_a",
    "tasks": ["pick up the bow"],
    "camera_features": ["observation.images.top"],
    "selectable": True,
    "unavailable_reason": None,
}


def setup_config(monkeypatch, tmp_path):
    config = {
        "gemini": {"default_model": "gemini-test"},
        "models": [{"id": "gemini-test", "name": "Gemini Test", "group": "Test", "input_limit": 1000}],
        "storage": {"results_dir": "data/results", "images_dir": "data/images"},
        "robot": {"port": "/dev/test", "id": "arm", "cameras": {"top": {"index": 0}}},
        "planner": {
            "use_ik": True,
            "selected_models": ["model_a"],
            "prompt_templates": {"use_ik": "IK {instruction} {available_models}", "no_ik": "NO {instruction} {available_models}"},
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (tmp_path / "models" / "model_a").mkdir(parents=True)
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "_load_model_registry", lambda cfg=None: [deepcopy(SELECTABLE_MODEL)])


def test_config_returns_planner_settings(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json()["planner"]["selected_models"] == ["model_a"]
    assert response.json()["planner"]["use_ik"] is True
    assert response.json()["execution_loop"] == {
        "actions_per_cycle": 100,
        "cycles_before_replan": 5,
        "max_replans": 3,
    }


def test_config_save_persists_execution_loop_settings(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post("/api/config/save", json={"execution_loop": {
        "actions_per_cycle": 250,
        "cycles_before_replan": 7,
        "max_replans": 4,
    }})

    assert response.status_code == 200
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert saved["execution_loop"] == {
        "actions_per_cycle": 250,
        "cycles_before_replan": 7,
        "max_replans": 4,
    }


def test_config_save_rejects_execution_loop_values_outside_ranges(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post("/api/config/save", json={"execution_loop": {
        "actions_per_cycle": 0,
        "cycles_before_replan": 21,
        "max_replans": 11,
    }})

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_EXECUTION_LOOP"


def test_config_save_rejects_empty_selected_models(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post("/api/config/save", json={
        "planner": {
            "use_ik": False,
            "selected_models": [],
            "prompt_templates": {"use_ik": "A", "no_ik": "B"},
        }
    })

    assert response.status_code == 400
    assert response.json()["code"] == "NO_SELECTED_MODELS"


def test_config_save_rejects_unselectable_model(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post("/api/config/save", json={
        "planner": {
            "use_ik": False,
            "selected_models": ["missing"],
            "prompt_templates": {"use_ik": "A", "no_ik": "B"},
        }
    })

    assert response.status_code == 400
    assert response.json()["code"] == "MODEL_NOT_SELECTABLE"


def test_config_save_persists_both_prompts_without_dropping_other_fields(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post("/api/config/save", json={
        "planner": {
            "use_ik": False,
            "selected_models": ["model_a"],
            "prompt_templates": {"use_ik": "USE", "no_ik": "NO"},
        }
    })

    assert response.status_code == 200
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert saved["planner"] == {
        "use_ik": False,
        "selected_models": ["model_a"],
        "prompt_templates": {"use_ik": "USE", "no_ik": "NO"},
    }
    assert saved["gemini"]["default_model"] == "gemini-test"


def test_model_registry_returns_workspace_relative_paths(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.get("/api/models/registry")

    assert response.status_code == 200
    assert response.json()["models"] == [SELECTABLE_MODEL]
    assert all(
        not item["local_path"].startswith("/")
        for item in response.json()["models"] if item["local_path"]
    )


def image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(output, format="JPEG")
    return output.getvalue()


def setup_infer(monkeypatch, tmp_path, raw_plan: str):
    setup_config(monkeypatch, tmp_path)
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    config["planner"]["use_ik"] = False
    config["verify"] = {"enabled": False, "max_retries": 1}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    images = tmp_path / "images"
    results = tmp_path / "results"
    images.mkdir()
    results.mkdir()
    monkeypatch.setattr(main, "IMAGES_DIR", images)
    monkeypatch.setattr(main, "RESULTS_DIR", results)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    async def fake_call_gemini(client, url, b64, mime, prompt):
        assert "MODEL_ID: model_a" in prompt
        assert "pick up the bow" in prompt
        return ({
            "candidates": [{"content": {"parts": [{"text": raw_plan}]}}],
            "usageMetadata": {},
        }, 0.01, 200)

    monkeypatch.setattr(main, "call_gemini", fake_call_gemini)


def test_infer_rejects_plan_with_task_outside_model_capabilities(monkeypatch, tmp_path):
    setup_infer(monkeypatch, tmp_path, '{"task_id":"bad","total_steps":1,"steps":[{"step_index":1,"description":"open the drawer","target_bbox":null,"method_id":"vla_model","model_id":"model_a"}]}')
    client = TestClient(main.app)

    response = client.post(
        "/api/infer",
        data={"model": "gemini-test", "instruction": "open the drawer"},
        files={"image": ("capture.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["error_code"] == "INVALID_PLAN"
    assert response.json()["validation_errors"][0]["code"] == "TASK_NOT_SUPPORTED"


def test_infer_accepts_selected_model_exact_training_task(monkeypatch, tmp_path):
    setup_infer(monkeypatch, tmp_path, '{"task_id":"good","total_steps":1,"steps":[{"step_index":1,"description":"pick up the bow","target_bbox":null,"method_id":"vla_model","model_id":"model_a"}]}')
    client = TestClient(main.app)

    response = client.post(
        "/api/infer",
        data={"model": "gemini-test", "instruction": "pick up the bow"},
        files={"image": ("capture.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["plan"]["steps"][0]["model_id"] == "model_a"
    assert response.json()["methods"] == ["vla_model:model_a"]


def test_infer_reports_invalid_plan_format_after_all_attempts(monkeypatch, tmp_path):
    raw_response = '{"step":{"description":"pick up the bow"}}'
    setup_infer(monkeypatch, tmp_path, raw_response)
    client = TestClient(main.app)

    response = client.post(
        "/api/infer",
        data={"model": "gemini-test", "instruction": "pick up the bow"},
        files={"image": ("capture.jpg", image_bytes(), "image/jpeg")},
    )

    body = response.json()
    assert body["error_code"] == "INVALID_PLAN_FORMAT"
    assert "steps" in body["error"]
    assert body["raw_response"] == raw_response
    assert body["plan"] is None


class FakeVlaManager:
    def __init__(self):
        self.started = None

    def start(self, command, model_id, task):
        self.started = (command, model_id, task)
        return {"ok": True, "pid": 123, "model_id": model_id, "task": task}

    def status(self):
        running = self.started is not None
        return {"ok": True, "state": "running" if running else "idle", "running": running, "model_id": "model_a" if running else None, "task": "pick up the bow" if running else None, "exit_code": None, "lines": []}

    def stop(self):
        return {"ok": True, "state": "stopped"}


class FakeVerifiedManager:
    def __init__(self):
        self.started = None
        self.stopped = False

    def start(self, *args):
        self.started = args
        return {"ok": True, "state": "running"}

    def status(self):
        return {"ok": True, "state": "stopped" if self.stopped else ("running" if self.started else "idle"), "running": bool(self.started) and not self.stopped}

    def stop(self):
        self.stopped = True
        return {"ok": True, "state": "stopped"}


def test_run_session_start_snapshots_config_and_full_plan(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    config["planner"]["use_ik"] = False
    config["execution_loop"] = {"actions_per_cycle": 250, "cycles_before_replan": 4, "max_replans": 2}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manager = FakeVerifiedManager()
    monkeypatch.setattr(main, "verified_manager", manager)
    client = TestClient(main.app)
    plan = {"task_id": "pick", "total_steps": 1, "steps": [{
        "step_index": 1, "description": "pick up the bow", "target_bbox": None,
        "method_id": "vla_model", "model_id": "model_a",
    }]}

    response = client.post("/api/run/session/start", json={
        "original_instruction": "pick up the bow",
        "plan": plan,
        "start_index": 0,
    })

    assert response.status_code == 200
    assert manager.started[0:4] == (
        "pick up the bow", plan, 0,
        {"actions_per_cycle": 250, "cycles_before_replan": 4, "max_replans": 2},
    )
    assert all(callable(adapter) for adapter in manager.started[4:])


def test_run_session_start_rejects_invalid_plan(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    manager = FakeVerifiedManager()
    monkeypatch.setattr(main, "verified_manager", manager)
    client = TestClient(main.app)

    response = client.post("/api/run/session/start", json={
        "original_instruction": "open drawer",
        "plan": {"steps": [{
            "step_index": 1, "description": "open drawer", "target_bbox": None,
            "method_id": "vla_model", "model_id": "model_a",
        }]},
        "start_index": 0,
    })

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_PLAN"
    assert manager.started is None


def test_completion_verifier_treats_invalid_gemini_schema_as_uncertain(monkeypatch):
    monkeypatch.setattr(main, "_capture_verified_frame", lambda step: (b"jpg", "image/jpeg"))
    monkeypatch.setattr(main, "_gemini_image_json", lambda prompt, image, mime: {"raw": '{"done":true}'})

    result = main._verified_completion({"description": "pick up the bow"})

    assert result["ok"] is True
    assert result["status"] == "uncertain"


def test_replan_rejects_completed_task_repetition(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    config["planner"]["use_ik"] = False
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(main, "_capture_verified_frame", lambda step: (b"jpg", "image/jpeg"))
    monkeypatch.setattr(main, "_gemini_image_json", lambda prompt, image, mime: {"raw": json.dumps({
        "task_id": "repeat", "total_steps": 1, "steps": [{
            "step_index": 1, "description": "pick up the bow", "target_bbox": None,
            "method_id": "vla_model", "model_id": "model_a",
        }],
    })})

    result = main._verified_replan({
        "original_instruction": "pick up the bow",
        "completed_steps": [{"step_index": 1, "description": "pick up the bow"}],
        "failed_step": {"step_index": 2, "description": "pick up the bow", "model_id": "model_a"},
        "verification_history": [],
    })

    assert result["ok"] is False
    assert result["validation_errors"][0]["code"] == "COMPLETED_STEP_REPEATED"


def test_run_step_starts_selected_model_with_exact_declared_task(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    manager = FakeVlaManager()
    monkeypatch.setattr(main, "vla_manager", manager)
    monkeypatch.setattr(main, "build_infer_command", lambda **kwargs: ["safe-command"])
    client = TestClient(main.app)

    response = client.post("/api/run/step", json={
        "method_id": "vla_model",
        "model_id": "model_a",
        "description": "pick up the bow",
        "target_bbox": None,
        "max_steps": 100,
    })

    assert response.status_code == 200
    assert response.json()["model_id"] == "model_a"
    assert manager.started == (["safe-command"], "model_a", "pick up the bow")


def test_run_step_rejects_task_outside_selected_model_capabilities(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "vla_manager", FakeVlaManager())
    client = TestClient(main.app)

    response = client.post("/api/run/step", json={
        "method_id": "vla_model",
        "model_id": "model_a",
        "description": "open the drawer",
        "max_steps": 100,
    })

    assert response.status_code == 400
    assert response.json()["code"] == "TASK_NOT_SUPPORTED"


def test_run_status_and_stop_delegate_to_process_manager(monkeypatch, tmp_path):
    setup_config(monkeypatch, tmp_path)
    manager = FakeVlaManager()
    manager.started = (["safe-command"], "model_a", "pick up the bow")
    monkeypatch.setattr(main, "vla_manager", manager)
    client = TestClient(main.app)

    assert client.get("/api/run/status").json()["state"] == "running"
    assert client.post("/api/run/stop").json()["state"] == "stopped"
