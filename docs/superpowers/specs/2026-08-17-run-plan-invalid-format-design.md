# Run Plan Invalid-Format Recovery Design

## Problem

`POST /api/infer` can receive a successful Gemini response that is valid JSON but does not contain the required top-level `steps` array. After all retries, the endpoint currently returns HTTP 200 with `plan: null` and no `error`. The Run UI consequently reports only `No plan returned`, hiding the actual failure. Existing saved planner templates may predate the explicit output schema and make this response shape more likely.

## Design

1. Planner settings will recognize a saved legacy template that lacks the required output-schema markers and fall back to the current built-in template for the selected IK mode. Valid customized templates that declare the required `steps` schema remain unchanged.
2. If every inference attempt fails to produce an object with a `steps` array, `/api/infer` will return a structured `INVALID_PLAN_FORMAT` error. The response will include a concise message, retry metadata, elapsed time, and the final raw model response for diagnostics.
3. The Run UI will continue using the endpoint's existing `error` field, so users see the specific format failure. It will retain `No plan returned` only as a defensive fallback for malformed or obsolete server responses.

## Compatibility and Data Handling

- The migration is read-time fallback only; it does not overwrite `config.yaml` or discard user changes.
- Model selection, exact-task validation, IK validation, verification, and successful response payloads remain unchanged.
- A custom template is considered current only when it explicitly describes a top-level `steps` array/schema; otherwise the safe built-in template is used.

## Tests

- Planner unit test: legacy no-IK template without an output schema resolves to the built-in no-IK template.
- Planner unit test: a schema-complete custom template is preserved.
- API test: repeated valid JSON responses without `steps` return `INVALID_PLAN_FORMAT`, not `plan: null` without an error.
- UI test: Run Plan surfaces the backend error and retains the defensive fallback.
- Existing planner, API, model registry, VLA execution, UI, and Camera WebSocket tests must remain green.
