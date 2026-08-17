"""Server-owned execute, observe, verify, and re-plan state machine."""

from __future__ import annotations

import copy
import threading
from typing import Callable


TERMINAL_STATES = {"completed", "needs_human_review", "stopped", "awaiting_step_selection"}


class VerifiedExecutionManager:
    def __init__(self, on_terminal: Callable | None = None):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = self._empty_state()
        self._on_terminal = on_terminal

    @staticmethod
    def _empty_state() -> dict:
        return {
            "ok": True,
            "state": "idle",
            "phase": "idle",
            "running": False,
            "original_instruction": "",
            "run_mode": "all",
            "plan": {"steps": []},
            "current_step_index": 0,
            "cycle": 0,
            "actions_per_cycle": 100,
            "cycles_before_replan": 5,
            "max_replans": 3,
            "replan_count": 0,
            "completed_steps": [],
            "verification_history": [],
            "error": None,
        }

    def _update(self, **values) -> None:
        with self._lock:
            self._state.update(values)

    def status(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._state)

    def start(
        self,
        original_instruction: str,
        plan: dict,
        start_index: int,
        settings: dict,
        execute_step: Callable,
        verify_step: Callable,
        replan: Callable,
        run_mode: str = "all",
    ) -> dict:
        with self._lock:
            if self._state["running"]:
                return {"ok": False, "code": "RUN_ALREADY_ACTIVE", "error": "A verified run is already active"}
            steps = plan.get("steps", []) if isinstance(plan, dict) else []
            if not isinstance(steps, list) or not steps or not 0 <= start_index < len(steps):
                return {"ok": False, "code": "INVALID_RUN_PLAN", "error": "Plan and start index are required"}
            if run_mode not in {"all", "step"}:
                return {"ok": False, "code": "INVALID_RUN_MODE", "error": "Run mode must be all or step"}
            self._stop_event = threading.Event()
            self._state = {
                **self._empty_state(),
                "state": "running",
                "phase": "starting",
                "running": True,
                "original_instruction": str(original_instruction),
                "run_mode": run_mode,
                "plan": copy.deepcopy(plan),
                "current_step_index": int(start_index),
                "actions_per_cycle": int(settings["actions_per_cycle"]),
                "cycles_before_replan": int(settings["cycles_before_replan"]),
                "max_replans": int(settings["max_replans"]),
            }
        self._thread = threading.Thread(
            target=self._worker,
            args=(execute_step, verify_step, replan),
            daemon=True,
        )
        self._thread.start()
        return {"ok": True, **self.status()}

    def _stop_requested(self) -> bool:
        if not self._stop_event.is_set():
            return False
        self._update(state="stopped", phase="stopped", running=False)
        return True

    def _review(self, error: str) -> None:
        self._update(state="needs_human_review", phase="needs_human_review", running=False, error=error)
        if self._on_terminal:
            self._on_terminal()

    def _worker(self, execute_step: Callable, verify_step: Callable, replan: Callable) -> None:
        while True:
            if self._stop_requested():
                return
            state = self.status()
            steps = state["plan"].get("steps", [])
            index = state["current_step_index"]
            if index >= len(steps):
                self._update(state="completed", phase="completed", running=False)
                if self._on_terminal:
                    self._on_terminal()
                return
            current = copy.deepcopy(steps[index])
            self._update(cycle=0)

            if current.get("method_id") != "vla_model":
                self._update(phase="executing")
                result = execute_step(current, state["actions_per_cycle"], self._stop_event)
                if self._stop_requested():
                    return
                if not result.get("ok"):
                    self._review(result.get("error", "Step execution failed"))
                    return
                completed = state["completed_steps"] + [current]
                self._update(completed_steps=completed, current_step_index=index + 1)
                if state["run_mode"] == "step":
                    self._update(state="completed", phase="completed", running=False)
                    if self._on_terminal:
                        self._on_terminal()
                    return
                continue

            task_history = []
            while True:
                if self._stop_requested():
                    return
                state = self.status()
                cycle = state["cycle"] + 1
                self._update(phase="executing", cycle=cycle)
                result = execute_step(current, state["actions_per_cycle"], self._stop_event)
                if self._stop_requested():
                    return
                if not result.get("ok"):
                    self._review(result.get("error", "VLA execution failed"))
                    return

                self._update(phase="verifying")
                verdict = verify_step(current)
                if self._stop_requested():
                    return
                if not verdict.get("ok", True):
                    self._review(verdict.get("error", "Completion verification failed"))
                    return
                status = verdict.get("status")
                if status not in {"success", "continue", "uncertain"}:
                    status = "uncertain"
                record = {
                    "step": copy.deepcopy(current),
                    "cycle": cycle,
                    "status": status,
                    "reason": str(verdict.get("reason", "")),
                    "visible_evidence": str(verdict.get("visible_evidence", "")),
                }
                task_history.append(record)
                history = self.status()["verification_history"] + [record]
                self._update(verification_history=history)

                if status == "success":
                    completed = self.status()["completed_steps"] + [current]
                    self._update(completed_steps=completed, current_step_index=index + 1)
                    if state["run_mode"] == "step":
                        self._update(state="completed", phase="completed", running=False)
                        if self._on_terminal:
                            self._on_terminal()
                        return
                    break
                if cycle < state["cycles_before_replan"]:
                    continue
                if state["replan_count"] >= state["max_replans"]:
                    self._review("Automatic re-plan limit reached")
                    return
                if self._stop_requested():
                    return

                self._update(phase="replanning", replan_count=state["replan_count"] + 1)
                context = {
                    "original_instruction": state["original_instruction"],
                    "completed_steps": self.status()["completed_steps"],
                    "failed_step": current,
                    "verification_history": task_history,
                    "replan_count": state["replan_count"] + 1,
                }
                replacement = replan(context)
                if self._stop_requested():
                    return
                replacement_plan = replacement.get("plan") if replacement.get("ok") else None
                replacement_steps = replacement_plan.get("steps", []) if isinstance(replacement_plan, dict) else []
                if not replacement.get("ok") or not isinstance(replacement_steps, list) or not replacement_steps:
                    self._review(replacement.get("error", "Re-plan did not return executable remaining steps"))
                    return
                self._update(plan=copy.deepcopy(replacement_plan), current_step_index=0, cycle=0)
                if state["run_mode"] == "step":
                    self._update(state="awaiting_step_selection", phase="awaiting_step_selection", running=False)
                    if self._on_terminal:
                        self._on_terminal()
                    return
                break

    def stop(self) -> dict:
        self._stop_event.set()
        with self._lock:
            if self._state["state"] not in TERMINAL_STATES:
                self._state.update(state="stopped", phase="stopped", running=False)
        if self._on_terminal:
            self._on_terminal()
        return {"ok": True, "state": self.status()["state"]}
