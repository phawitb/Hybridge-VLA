"""Pick-and-place task resolution and background lifecycle for Generate 3D."""

from __future__ import annotations

import copy
import re
import threading
from typing import Callable


class TaskResolutionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(value).casefold()).split())


def resolve_pick_place_objects(instruction: str, objects: list[dict]) -> tuple[dict, dict]:
    normalized_instruction = _normalized(instruction)
    matches = []
    names = []
    normalized_names = [_normalized(obj.get("name", "")) for obj in objects if isinstance(obj, dict)]
    if any(name and normalized_names.count(name) > 1 for name in normalized_names):
        available = ", ".join(str(obj.get("name", "")) for obj in objects if isinstance(obj, dict)) or "none"
        raise TaskResolutionError(
            "OBJECT_MATCH_REQUIRED",
            f"Detected object names must be unique. Available: {available}",
        )
    for obj in objects:
        name = str(obj.get("name", "")).strip()
        normalized_name = _normalized(name)
        if not normalized_name:
            continue
        names.append(name)
        match = re.search(rf"(?:^| ){re.escape(normalized_name)}(?: |$)", normalized_instruction)
        if match:
            matches.append((match.start(), obj))
    matches.sort(key=lambda item: item[0])
    if len(matches) != 2 or matches[0][1] is matches[1][1]:
        available = ", ".join(names) or "none"
        raise TaskResolutionError(
            "OBJECT_MATCH_REQUIRED",
            f"Instruction must mention exactly two detected object names. Available: {available}",
        )
    return copy.deepcopy(matches[0][1]), copy.deepcopy(matches[1][1])


class Generate3DTaskManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = self._empty_state()

    @staticmethod
    def _empty_state() -> dict:
        return {
            "ok": True,
            "state": "idle",
            "phase": "idle",
            "running": False,
            "instruction": "",
            "source": None,
            "target": None,
            "joints": None,
            "error": None,
        }

    def status(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._state)

    def _publish(self, phase: str, joints: dict | None = None) -> None:
        with self._lock:
            self._state["phase"] = str(phase)
            if joints is not None:
                self._state["joints"] = copy.deepcopy(joints)

    def start(
        self,
        instruction: str,
        source: dict,
        target: dict,
        execute: Callable,
    ) -> dict:
        with self._lock:
            if self._state["running"]:
                return {
                    "ok": False,
                    "code": "TASK_ALREADY_RUNNING",
                    "error": "A Generate 3D task is already running",
                }
            self._stop_event = threading.Event()
            self._state = {
                **self._empty_state(),
                "state": "running",
                "phase": "starting",
                "running": True,
                "instruction": str(instruction),
                "source": copy.deepcopy(source),
                "target": copy.deepcopy(target),
            }
        self._thread = threading.Thread(
            target=self._worker,
            args=(execute,),
            daemon=True,
        )
        self._thread.start()
        return {"ok": True, **self.status()}

    def _worker(self, execute: Callable) -> None:
        state = self.status()
        try:
            execute(
                state["source"],
                state["target"],
                self._stop_event,
                self._publish,
            )
            with self._lock:
                if self._stop_event.is_set():
                    self._state.update(state="stopped", phase="stopped", running=False)
                else:
                    self._state.update(state="completed", phase="completed", running=False)
        except Exception as exc:
            with self._lock:
                self._state.update(
                    state="failed",
                    phase="failed",
                    running=False,
                    error=str(exc),
                )

    def stop(self, timeout: float = 2.0) -> dict:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            if self._state["running"]:
                self._state.update(state="stopped", phase="stopped")
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))
        with self._lock:
            if thread is None or not thread.is_alive():
                self._state["running"] = False
        return self.status()
