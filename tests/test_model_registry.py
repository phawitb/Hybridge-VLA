import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from model_registry import (
    get_model_record,
    load_local_models,
    load_remote_models,
    merge_model_records,
)


def make_model_fixture(
    tmp_path: Path,
    model_name: str,
    policy_type: str,
    dataset_name: str,
    tasks: list[str],
    cameras: list[str],
) -> Path:
    root = tmp_path / "models"
    model_dir = root / model_name
    dataset_dir = tmp_path / "data" / dataset_name
    model_dir.mkdir(parents=True)
    (dataset_dir / "meta").mkdir(parents=True)
    (model_dir / "train_config.json").write_text(json.dumps({
        "dataset": {
            "repo_id": f"phawitbinabik/{dataset_name}",
            "root": f"./data/{dataset_name}",
        }
    }))
    (model_dir / "config.json").write_text(json.dumps({
        "type": policy_type,
        "input_features": {
            "observation.state": {"type": "STATE"},
            **{camera: {"type": "VISUAL"} for camera in cameras},
        },
    }))
    pq.write_table(
        pa.table({"task_index": list(range(len(tasks))), "task": tasks}),
        dataset_dir / "meta" / "tasks.parquet",
    )
    return root


def make_model_fixture_without_tasks(tmp_path: Path, model_name: str, policy_type: str) -> Path:
    root = tmp_path / "models"
    model_dir = root / model_name
    model_dir.mkdir(parents=True)
    (model_dir / "train_config.json").write_text(json.dumps({
        "dataset": {"repo_id": "phawitbinabik/missing", "root": "./data/missing"}
    }))
    (model_dir / "config.json").write_text(json.dumps({
        "type": policy_type,
        "input_features": {"observation.images.top": {"type": "VISUAL"}},
    }))
    return root


def test_load_local_models_reads_exact_training_tasks(tmp_path):
    root = make_model_fixture(
        tmp_path,
        model_name="smolvla_demo",
        policy_type="smolvla",
        dataset_name="demo_ds",
        tasks=["pick up the bow", "place the bow in the bowl", "pick up the bow"],
        cameras=["observation.images.top", "observation.images.wrist"],
    )

    [record] = load_local_models(root, {"top", "wrist"})

    assert record["tasks"] == ["pick up the bow", "place the bow in the bowl"]
    assert record["camera_features"] == [
        "observation.images.top",
        "observation.images.wrist",
    ]
    assert record["policy_type"] == "smolvla"
    assert record["dataset_id"] == "phawitbinabik/demo_ds"
    assert record["selectable"] is True


def test_load_local_models_disables_missing_task_metadata(tmp_path):
    root = make_model_fixture_without_tasks(tmp_path, "act_unknown", "act")

    [record] = load_local_models(root, {"top", "wrist"})

    assert record["selectable"] is False
    assert record["unavailable_reason"] == "Training instructions unavailable"


def test_load_local_models_disables_missing_camera(tmp_path):
    root = make_model_fixture(
        tmp_path,
        model_name="smolvla_two_cam",
        policy_type="smolvla",
        dataset_name="demo_ds",
        tasks=["pick up the bow"],
        cameras=["observation.images.top", "observation.images.wrist"],
    )

    [record] = load_local_models(root, {"top"})

    assert record["selectable"] is False
    assert record["unavailable_reason"] == "Missing configured cameras: wrist"


def test_load_local_models_disables_unsupported_policy(tmp_path):
    root = make_model_fixture(
        tmp_path,
        model_name="diffusion_demo",
        policy_type="diffusion",
        dataset_name="demo_ds",
        tasks=["pick up the bow"],
        cameras=["observation.images.top"],
    )

    [record] = load_local_models(root, {"top"})

    assert record["selectable"] is False
    assert record["unavailable_reason"] == "Unsupported policy type: diffusion"


def test_load_remote_models_reads_tasks_through_injected_hub_loader():
    records = load_remote_models(
        [{"id": "phawitbinabik/smolvla_demo", "name": "smolvla_demo"}],
        ["phawitbinabik/demo"],
        task_loader=lambda dataset_id: ["pick up the bow"]
        if dataset_id == "phawitbinabik/demo" else [],
    )

    assert records[0]["id"] == "smolvla_demo"
    assert records[0]["dataset_id"] == "phawitbinabik/demo"
    assert records[0]["tasks"] == ["pick up the bow"]
    assert records[0]["selectable"] is False
    assert records[0]["unavailable_reason"] == "Download required"


def test_merge_model_records_prefers_local_capabilities():
    local = {
        "id": "smolvla_demo", "repo_id": None, "display_name": "smolvla_demo",
        "policy_type": "smolvla", "downloaded": True, "local_path": "models/smolvla_demo",
        "dataset_id": "phawitbinabik/demo", "tasks": ["pick up the bow"],
        "camera_features": ["observation.images.top"], "selectable": True,
        "unavailable_reason": None,
    }
    remote = [
        {
            "id": "smolvla_demo", "repo_id": "phawitbinabik/smolvla_demo",
            "display_name": "smolvla_demo", "policy_type": None, "downloaded": False,
            "local_path": None, "dataset_id": "phawitbinabik/demo", "tasks": ["pick up the bow"],
            "camera_features": [], "selectable": False, "unavailable_reason": "Download required",
        },
        {
            "id": "act_remote", "repo_id": "phawitbinabik/act_remote",
            "display_name": "act_remote", "policy_type": None, "downloaded": False,
            "local_path": None, "dataset_id": None, "tasks": [], "camera_features": [],
            "selectable": False, "unavailable_reason": "Download required",
        },
    ]

    merged = merge_model_records([local], remote)

    assert [item["id"] for item in merged] == ["act_remote", "smolvla_demo"]
    assert merged[1]["repo_id"] == "phawitbinabik/smolvla_demo"
    assert merged[1]["downloaded"] is True
    assert merged[1]["tasks"] == ["pick up the bow"]
    assert get_model_record(merged, "smolvla_demo") == merged[1]
    assert get_model_record(merged, "missing") is None
