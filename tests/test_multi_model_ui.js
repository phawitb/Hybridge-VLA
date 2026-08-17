const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(new URL('../static/index.html', `file://${__dirname}/`), 'utf8');

function extractFunction(name) {
  const start = html.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${name} must exist`);
  const openParen = html.indexOf('(', start);
  let parenDepth = 0;
  let closeParen = -1;
  for (let index = openParen; index < html.length; index += 1) {
    if (html[index] === '(') parenDepth += 1;
    if (html[index] === ')') {
      parenDepth -= 1;
      if (parenDepth === 0) { closeParen = index; break; }
    }
  }
  const brace = html.indexOf('{', closeParen);
  let depth = 0;
  let quote = null;
  let escaped = false;
  for (let index = brace; index < html.length; index += 1) {
    const char = html[index];
    if (quote) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === quote) quote = null;
      continue;
    }
    if (char === "'" || char === '"' || char === '`') { quote = char; continue; }
    if (char === '{') depth += 1;
    if (char === '}') {
      depth -= 1;
      if (depth === 0) return html.slice(start, index + 1);
    }
  }
  throw new Error(`unterminated function ${name}`);
}

const context = { console };
vm.createContext(context);
for (const name of ['selectableModelIds', 'buildPlannerPayload', 'buildRunStepPayload', 'runStateLabel']) {
  vm.runInContext(`${extractFunction(name)}; this.${name} = ${name};`, context);
}

assert.deepEqual(
  Array.from(context.selectableModelIds([
    {id: 'a', selectable: true},
    {id: 'b', selectable: false},
  ])),
  ['a'],
);

assert.throws(
  () => context.buildPlannerPayload({
    selectedIds: [], useIk: false, useIkPrompt: 'A', noIkPrompt: 'B',
  }),
  /Select at least one downloaded model/,
);

const payload = context.buildPlannerPayload({
  selectedIds: ['a', 'b'], useIk: false, useIkPrompt: 'A', noIkPrompt: 'B',
});
assert.deepEqual(JSON.parse(JSON.stringify(payload)), {
  use_ik: false,
  selected_models: ['a', 'b'],
  prompt_templates: {use_ik: 'A', no_ik: 'B'},
});

assert.deepEqual(JSON.parse(JSON.stringify(context.buildRunStepPayload({
  method_id: 'vla_model',
  model_id: 'model_a',
  description: 'pick up the bow',
  target_bbox: null,
}, 100))), {
  method_id: 'vla_model',
  model_id: 'model_a',
  description: 'pick up the bow',
  target_bbox: null,
  max_steps: 100,
});

assert.equal(context.runStateLabel({state: 'running', model_id: 'model_a'}), 'Running model_a');
assert.equal(context.runStateLabel({state: 'failed', exit_code: 1}), 'Failed (exit 1)');
assert.equal(context.runStateLabel({state: 'completed'}), 'Completed');

const runPlanSource = extractFunction('runPlan');
assert.ok(
  runPlanSource.indexOf('if (d.error)') < runPlanSource.indexOf('if (!d.plan || !d.plan.steps)'),
  'runPlan must surface the backend error before its defensive missing-plan fallback',
);
assert.match(runPlanSource, /throw new Error\(d\.error\)/);

console.log('multi-model config UI behavior passes');
