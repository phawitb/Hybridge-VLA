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
for (const name of ['selectableModelIds', 'buildPlannerPayload', 'buildRunStepPayload', 'runStateLabel', 'buildExecutionLoopPayload', 'buildRunSessionPayload', 'verifiedRunLabel', 'stepMatches', 'verificationForStep']) {
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

assert.deepEqual(JSON.parse(JSON.stringify(context.buildExecutionLoopPayload({
  actionsPerCycle: 100, cyclesBeforeReplan: 5, maxReplans: 3,
}))), {actions_per_cycle: 100, cycles_before_replan: 5, max_replans: 3});
assert.throws(() => context.buildExecutionLoopPayload({
  actionsPerCycle: 0, cyclesBeforeReplan: 5, maxReplans: 3,
}), /Actions per cycle/);

assert.deepEqual(JSON.parse(JSON.stringify(context.buildRunSessionPayload('pick', {steps: []}, 2, 'step'))), {
  original_instruction: 'pick', plan: {steps: []}, start_index: 2, run_mode: 'step',
});
assert.equal(context.buildRunSessionPayload('pick', {steps: []}, 2, 'all').start_index, 0);
assert.throws(() => context.buildRunSessionPayload('pick', {steps: []}, 0, 'bad'), /Run mode/);

const latestCheck = context.verificationForStep([
  {step: {step_index: 1, description: 'pick', model_id: 'a'}, cycle: 1, status: 'continue'},
  {step: {step_index: 1, description: 'pick', model_id: 'a'}, cycle: 2, status: 'success', reason: 'released', visible_evidence: 'inside bowl'},
], {step_index: 1, description: 'pick', model_id: 'a'});
assert.equal(latestCheck.status, 'success');
assert.equal(latestCheck.cycle, 2);
assert.equal(context.verifiedRunLabel({phase: 'executing', cycle: 2, cycles_before_replan: 5, actions_per_cycle: 100}), 'Executing cycle 2/5 · 100 actions');
assert.equal(context.verifiedRunLabel({phase: 'replanning', replan_count: 1, max_replans: 3}), 'Re-planning remaining work 1/3');
assert.equal(context.verifiedRunLabel({state: 'needs_human_review', error: 'limit'}), 'Needs human review: limit');

for (const id of ['cfgActionsPerCycle', 'cfgCyclesBeforeReplan', 'cfgMaxReplans']) {
  assert.match(html, new RegExp(`id=["']${id}["']`));
}
assert.match(extractFunction('runStep'), /startRunSession\('step'\)/);
assert.match(extractFunction('startRunSession'), /\/api\/run\/session\/start/);
assert.match(extractFunction('runAll'), /startRunSession\('all'\)/);
assert.match(html, /id=["']runAllBtn["']/);
assert.match(extractFunction('runBuildFlow'), /visible_evidence/);
assert.match(html, /class=["'][^"']*run-three-column/);
assert.ok(html.indexOf('id="runBboxCanvas"') < html.indexOf('id="runFlowChart"'));
assert.ok(html.indexOf('id="runFlowChart"') < html.indexOf('id="runStepLog"'));

const runPlanSource = extractFunction('runPlan');
assert.ok(
  runPlanSource.indexOf('if (d.error)') < runPlanSource.indexOf('if (!d.plan || !d.plan.steps)'),
  'runPlan must surface the backend error before its defensive missing-plan fallback',
);
assert.match(runPlanSource, /throw new Error\(d\.error\)/);

console.log('multi-model config UI behavior passes');
