# Persistent VLA Cycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pause a loaded VLA process at each action-cycle boundary for Gemini verification, then resume the same process without model reload or robot torque release.

**Architecture:** `infer_python.py` owns robot/model/camera lifetime and publishes the latest observation snapshot at each boundary. `VlaProcessManager` provides a line-based stdin/stdout control channel and exposes `waiting_for_verification`. The verified execution adapter resumes a compatible process, verifies its snapshot, and stops it only on success, task/model change, Stop, failure, or timeout.

**Tech Stack:** Python 3.12, subprocess pipes, threading, OpenCV, pytest

## Global Constraints

- Preserve the configured action count per cycle.
- Keep torque, model, and camera loaded while waiting for Gemini.
- Never open the same camera from the server during an active inference process.
- Disconnect after 120 seconds without a verification command.
- Stop remains immediate and idempotent.
- Preserve the user's `config.yaml` changes.

### Task 1: Inference Cycle Protocol

- [ ] Add failing tests for command parsing, boundary decisions, snapshot extraction, and timeout behavior.
- [ ] Add `--verification-image` and `--verification-timeout` arguments.
- [ ] Save the latest observation image atomically at each boundary.
- [ ] Emit a machine-readable `CYCLE_READY` event and wait for `continue` or `stop` while keeping the robot connected.
- [ ] Run tests and commit.

### Task 2: Persistent Process Control

- [ ] Add failing manager tests for `waiting_for_verification`, snapshot metadata, `continue_cycle`, and Stop.
- [ ] Change subprocess stdin to a pipe and parse cycle-ready events.
- [ ] Expose process compatibility by model and task.
- [ ] Run tests and commit.

### Task 3: Verified Execution Integration

- [ ] Add failing adapter tests proving the second cycle resumes the same PID/process.
- [ ] Resume compatible waiting processes; stop incompatible processes before launch.
- [ ] Verify the exported snapshot instead of opening a camera.
- [ ] Stop the persistent process on Gemini success and terminal failure paths.
- [ ] Run all tests, syntax checks, and a fake-process end-to-end cycle.
- [ ] Merge to `v3`, verify again, and clean the worktree.
