"""Safe subprocess lifecycle for bounded VLA plan-step execution."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from pathlib import Path


def build_infer_command(
    python: str,
    script: Path,
    model: dict,
    task: str,
    robot: dict,
    cameras: dict,
    max_steps: int,
) -> list[str]:
    required_names = [
        feature.rsplit(".", 1)[-1]
        for feature in model.get("camera_features", [])
    ]
    missing = [name for name in required_names if name not in cameras]
    if missing:
        raise ValueError(f"Missing configured cameras: {', '.join(missing)}")
    camera_indices = {
        name: int(cameras[name].get("index", 0))
        for name in required_names
    }
    first_camera = cameras[required_names[0]] if required_names else {}
    command = [
        python,
        str(script),
        f"--model-path={model['local_path']}",
        f"--task={task}",
        f"--robot-port={robot.get('port', '')}",
        f"--robot-id={robot.get('id', 'my_awesome_follower_arm')}",
        f"--cameras={json.dumps(camera_indices, separators=(',', ':'))}",
        f"--fps={int(first_camera.get('fps', 30))}",
        f"--width={int(first_camera.get('w', first_camera.get('width', 320)))}",
        f"--height={int(first_camera.get('h', first_camera.get('height', 240)))}",
        f"--max-steps={int(max_steps)}",
    ]
    if model.get("policy_type") == "smolvla":
        command.append("--cache-language")
    return command


class VlaProcessManager:
    def __init__(self, max_log_lines: int = 500):
        self.max_log_lines = max_log_lines
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._state = "idle"
        self._model_id: str | None = None
        self._task: str | None = None
        self._exit_code: int | None = None
        self._lines: list[str] = []

    def start(self, command: list[str], model_id: str, task: str) -> dict:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return {"ok": False, "code": "VLA_ALREADY_RUNNING", "error": "VLA inference is already running"}
            self._state = "starting"
            self._model_id = model_id
            self._task = task
            self._exit_code = None
            self._lines = []
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._exit_code = -1
                self._lines = [str(exc)]
            return {"ok": False, "code": "VLA_LAUNCH_FAILED", "error": str(exc)}
        with self._lock:
            self._process = process
            self._state = "running"
        threading.Thread(target=self._reader, args=(process,), daemon=True).start()
        return {"ok": True, "pid": process.pid, "model_id": model_id, "task": task}

    def _reader(self, process: subprocess.Popen) -> None:
        stream = process.stdout
        if stream is not None:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                with self._lock:
                    self._lines.append(line)
                    if len(self._lines) > self.max_log_lines:
                        self._lines = self._lines[-self.max_log_lines:]
        exit_code = process.wait()
        with self._lock:
            self._exit_code = exit_code
            if self._state != "stopped":
                self._state = "completed" if exit_code == 0 else "failed"

    def status(self) -> dict:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return {
                "ok": True,
                "state": self._state,
                "running": running,
                "model_id": self._model_id,
                "task": self._task,
                "exit_code": self._exit_code,
                "lines": list(self._lines),
            }

    def stop(self, timeout: float = 5.0) -> dict:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return {"ok": True, "state": self.status()["state"]}
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            try:
                process.send_signal(signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                process.kill()
            process.wait(timeout=timeout)
        with self._lock:
            self._state = "stopped"
            self._exit_code = process.returncode
        return {"ok": True, "state": "stopped"}
