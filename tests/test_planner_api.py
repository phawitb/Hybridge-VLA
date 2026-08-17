from copy import deepcopy
import io

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
