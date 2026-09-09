"""Persistent sequential orchestration for Generate 3D Flow Run."""

from __future__ import annotations

import copy
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


TERMINAL = {"success", "failed", "uncertain", "stopped", "interrupted"}
ACTIVE = {"capturing", "detecting", "planning", "executing", "capturing_verification", "verifying"}


class FlowValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_flow_plan(raw: object) -> list[dict]:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        try:
            raw = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise FlowValidationError("INVALID_FLOW_PLAN", "Gemini returned invalid flow JSON") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("subtasks"), list):
        raise FlowValidationError("INVALID_FLOW_PLAN", "Flow plan must contain a subtasks array")
    subtasks = raw["subtasks"]
    if not 1 <= len(subtasks) <= 10:
        raise FlowValidationError("INVALID_FLOW_PLAN", "Flow must contain between 1 and 10 subtasks")
    result = []
    for index, item in enumerate(subtasks):
        if not isinstance(item, dict):
            raise FlowValidationError("INVALID_FLOW_PLAN", f"Subtask {index + 1} is invalid")
        clean = {}
        for field in ("instruction", "source_name", "target_name"):
            value = " ".join(str(item.get(field, "")).strip().split())
            if not value or len(value) > 500:
                raise FlowValidationError("INVALID_FLOW_PLAN", f"Subtask {index + 1} needs {field}")
            clean[field] = value
        result.append(clean)
    return result


class Generate3DFlowManager:
    def __init__(self, state_file: Path, artifact_root: Path):
        self.state_file = Path(state_file)
        self.artifact_root = Path(artifact_root)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stop_active: Callable | None = None
        self._state = self._empty_state()
        self._restore()

    @staticmethod
    def _empty_state() -> dict:
        return {"ok": True, "schema_version": 1, "flow_id": None, "instruction": "", "status": "empty",
                "running": False, "active_block_index": None, "run_scope": None, "config": {}, "planning": {},
                "blocks": [], "created_at": None, "updated_at": None, "error": None}

    def _restore(self) -> None:
        try:
            loaded = json.loads(self.state_file.read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            return
        if not isinstance(loaded, dict) or loaded.get("schema_version") != 1:
            return
        if loaded.get("running") or loaded.get("status") == "running":
            index = loaded.get("active_block_index")
            if isinstance(index, int) and 0 <= index < len(loaded.get("blocks", [])):
                loaded["blocks"][index].update(phase="interrupted", outcome="interrupted")
            loaded.update(status="interrupted", running=False, active_block_index=None, run_scope=None,
                          error="Flow was interrupted by application restart")
            self._state = loaded
            self._persist()
        else:
            self._state = loaded

    def _persist(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.state_file)

    def status(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._state)

    def create_flow(self, instruction: str, subtasks: object, planning: dict | None = None) -> dict:
        clean_instruction = " ".join(str(instruction).strip().split())
        if not clean_instruction:
            raise FlowValidationError("INSTRUCTION_REQUIRED", "Enter a long task instruction")
        blocks = parse_flow_plan({"subtasks": subtasks})
        with self._lock:
            if self._state.get("running"):
                raise FlowValidationError("FLOW_RUNNING", "Stop the active flow before creating another")
            stamp = _now()
            flow_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
            self._state = {**self._empty_state(), "flow_id": flow_id, "instruction": clean_instruction,
                "status": "ready", "planning": copy.deepcopy(planning or {}), "created_at": stamp, "updated_at": stamp,
                "blocks": [{**item, "id": f"block-{index + 1}", "index": index, "phase": "pending", "outcome": None,
                    "timestamps": {"pending": stamp}, "config": {}, "objects": [], "path": [], "task_id": None,
                    "artifacts": {}, "prompts": {}, "verification": None, "error": None, "error_code": None}
                    for index, item in enumerate(blocks)]}
            (self.artifact_root / flow_id).mkdir(parents=True, exist_ok=True)
            self._persist()
            return self.status()

    def _update_block(self, index: int, phase: str, updates: dict | None = None) -> None:
        with self._lock:
            block = self._state["blocks"][index]
            block["phase"] = phase
            block.setdefault("timestamps", {})[phase] = _now()
            if updates:
                block.update(copy.deepcopy(updates))
            self._state["updated_at"] = _now()
            self._persist()

    def start(self, scope: str, block_index: int | None, config: dict, runner: Callable,
              stop_active: Callable | None = None, on_all_success: Callable | None = None) -> dict:
        with self._lock:
            if self._state.get("running"):
                return {"ok": False, "code": "FLOW_RUNNING", "error": "A flow is already running"}
            if not self._state.get("blocks"):
                return {"ok": False, "code": "FLOW_REQUIRED", "error": "Generate a flow first"}
            if scope not in {"all", "block"}:
                return {"ok": False, "code": "INVALID_SCOPE", "error": "Scope must be all or block"}
            if scope == "block" and (not isinstance(block_index, int) or not 0 <= block_index < len(self._state["blocks"])):
                return {"ok": False, "code": "INVALID_BLOCK", "error": "Select a valid flow block"}
            indices = [block_index] if scope == "block" else [b["index"] for b in self._state["blocks"] if b["phase"] != "success"]
            if not indices:
                return {"ok": False, "code": "FLOW_COMPLETE", "error": "All flow blocks are already complete"}
            self._stop_event = threading.Event()
            self._stop_active = stop_active
            self._state.update(status="running", running=True, run_scope=scope, config=copy.deepcopy(config), error=None)
            self._persist()
            self._thread = threading.Thread(target=self._worker,
                args=(indices, scope, copy.deepcopy(config), runner, on_all_success), daemon=True)
            self._thread.start()
            return {"ok": True, **self.status()}

    def _worker(self, indices, scope, config, runner, on_all_success) -> None:
        final = "success"
        try:
            for index in indices:
                if self._stop_event.is_set():
                    final = "stopped"
                    break
                with self._lock:
                    self._state["active_block_index"] = index
                    self._state["blocks"][index]["config"] = copy.deepcopy(config)
                    self._persist()
                    block = copy.deepcopy(self._state["blocks"][index])
                def transition(phase, updates=None, _index=index):
                    self._update_block(_index, phase, updates)
                result = runner(block, copy.deepcopy(config), transition, self._stop_event.is_set) or {}
                outcome = result.get("status", "uncertain")
                if outcome not in TERMINAL:
                    outcome = "uncertain"
                result = {key: value for key, value in result.items() if key != "status"}
                result["outcome"] = outcome
                self._update_block(index, outcome, result)
                if outcome != "success":
                    final = outcome
                    break
            if final == "success" and scope == "all" and on_all_success:
                on_all_success(config)
        except Exception as exc:
            final = "failed"
            with self._lock:
                index = self._state.get("active_block_index")
            if isinstance(index, int):
                self._update_block(index, "failed", {"outcome": "failed", "error": str(exc), "error_code": "FLOW_EXECUTION_FAILED"})
        finally:
            with self._lock:
                self._state.update(status=final, running=False, active_block_index=None, run_scope=None,
                                   updated_at=_now(), error=None if final == "success" else self._state.get("error"))
                self._stop_active = None
                self._persist()

    def stop(self, stop_active: Callable | None = None) -> dict:
        self._stop_event.set()
        callback = stop_active or self._stop_active
        if callback and self.status().get("running"):
            callback()
        return {"ok": True, **self.status()}

    def artifact_path(self, flow_id: str, filename: str) -> Path:
        if flow_id != self._state.get("flow_id") or not filename or Path(filename).name != filename:
            raise FlowValidationError("INVALID_ARTIFACT", "Unknown flow artifact")
        root = (self.artifact_root / flow_id).resolve()
        candidate = (root / filename).resolve()
        if candidate.parent != root:
            raise FlowValidationError("INVALID_ARTIFACT", "Unknown flow artifact")
        return candidate
