# Run All and Step UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate whole-plan and single-step execution while showing Gemini verification inside each plan block and moving terminal logs to a rightmost column.

**Architecture:** Add a snapshotted `run_mode` to `VerifiedExecutionManager`. In `all` mode the state machine keeps its existing automatic advancement; in `step` mode it stops after the selected step succeeds or after a replacement plan is accepted. The Run UI sends mode-specific start payloads and renders server verification history per block.

**Tech Stack:** Python 3.12, FastAPI, pytest, browser JavaScript, Node.js assertions

## Constraints

- `Run All` starts at index zero and executes every task in the current plan.
- `Run Step` executes only the selected task.
- Stop cancels both modes.
- Gemini status and latest evidence/reason live inside the matching block.
- Terminal log is the third/rightmost column.
- Preserve persistent VLA model reuse and configured retry/re-plan limits.

### Task 1: Execution Modes

- [ ] Add failing state-machine tests for Run All and Run Step success/re-plan boundaries.
- [ ] Add `run_mode` to session state and start API validation.
- [ ] Stop single-step sessions after success or accepted re-plan.
- [ ] Run focused tests and commit.

### Task 2: Run Controls and Block Verification

- [ ] Add failing Node assertions for Run All/Run Step payloads and block verification content.
- [ ] Add Run All and Run Step controls with distinct mode payloads.
- [ ] Render cycle, phase, latest verdict, reason, and evidence in each matching plan block.
- [ ] Keep re-plan updates mode-aware.
- [ ] Run UI tests and commit.

### Task 3: Three-Column Layout and Verification

- [ ] Move terminal markup into a third/rightmost Run column.
- [ ] Add responsive fallback for narrow screens.
- [ ] Run all Python/Node/syntax checks.
- [ ] Merge to `v3`, verify merged output, and remove temporary worktree/branch.
