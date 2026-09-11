import json

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

import main


def _write_dataset(root, name="demo"):
    dataset = root / "data" / name
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    (dataset / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dataset / "videos" / "observation.images.custom" / "chunk-000").mkdir(parents=True)
    (dataset / "videos" / "observation.images.custom" / "chunk-000" / "file-000.mp4").write_bytes(b"video")
    (dataset / "meta" / "info.json").write_text(json.dumps({
        "fps": 2,
        "total_episodes": 2,
        "total_frames": 7,
        "splits": {"train": "0:2"},
        "features": {"observation.images.custom": {"dtype": "video"}},
    }))
    rows = []
    for episode, count in [(0, 4), (1, 3)]:
        for frame in range(count):
            rows.append({
                "episode_index": episode,
                "frame_index": frame,
                "index": len(rows),
                "timestamp": frame / 2,
                "task_index": 0,
                "observation.state": [float(episode), float(frame)],
                "action": [float(frame)],
            })
    pq.write_table(pa.Table.from_pylist(rows), dataset / "data" / "chunk-000" / "file-000.parquet")
    episodes = [
        {"episode_index": 0, "length": 4, "dataset_from_index": 0, "dataset_to_index": 4,
         "videos/observation.images.custom/from_timestamp": 10.0,
         "videos/observation.images.custom/to_timestamp": 12.0,
         "videos/observation.images.custom/chunk_index": 0,
         "videos/observation.images.custom/file_index": 0},
        {"episode_index": 1, "length": 3, "dataset_from_index": 4, "dataset_to_index": 7,
         "videos/observation.images.custom/from_timestamp": 20.0,
         "videos/observation.images.custom/to_timestamp": 21.5,
         "videos/observation.images.custom/chunk_index": 0,
         "videos/observation.images.custom/file_index": 0},
    ]
    pq.write_table(pa.Table.from_pylist(episodes), dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return dataset


def test_remove_beginning_creates_reindexed_dataset_and_advances_video_offsets(monkeypatch, tmp_path):
    _write_dataset(tmp_path)
    monkeypatch.setattr(main, "ROOT", tmp_path)

    response = TestClient(main.app).post("/api/datasets/demo/remove-beginning", json={
        "seconds": 1,
        "output_name": "demo_remove_1s",
    })

    assert response.status_code == 200
    assert response.json() == {
        "ok": True, "name": "demo_remove_1s", "episodes": 2,
        "frames": 3, "removed_frames": 4, "seconds": 1.0,
    }
    output = tmp_path / "data" / "demo_remove_1s"
    rows = pq.read_table(output / "data" / "chunk-000" / "file-000.parquet").to_pylist()
    assert [(r["episode_index"], r["frame_index"], r["index"], r["timestamp"]) for r in rows] == [
        (0, 0, 0, 0.0), (0, 1, 1, 0.5), (1, 0, 2, 0.0),
    ]
    episodes = pq.read_table(output / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
    assert [(e["length"], e["dataset_from_index"], e["dataset_to_index"]) for e in episodes] == [
        (2, 0, 2), (1, 2, 3),
    ]
    assert [e["videos/observation.images.custom/from_timestamp"] for e in episodes] == [11.0, 21.0]
    assert (output / "videos" / "observation.images.custom" / "chunk-000" / "file-000.mp4").read_bytes() == b"video"
    assert json.loads((output / "meta" / "info.json").read_text())["total_frames"] == 3


def test_remove_beginning_rejects_duration_that_would_empty_an_episode(monkeypatch, tmp_path):
    _write_dataset(tmp_path)
    monkeypatch.setattr(main, "ROOT", tmp_path)

    response = TestClient(main.app).post("/api/datasets/demo/remove-beginning", json={
        "seconds": 2,
        "output_name": "bad",
    })

    assert response.json()["ok"] is False
    assert "episode 0" in response.json()["error"].lower()
    assert not (tmp_path / "data" / "bad").exists()

