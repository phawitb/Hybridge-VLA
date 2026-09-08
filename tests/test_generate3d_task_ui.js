const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(new URL('../static/index.html', `file://${__dirname}/`), 'utf8');

function extractFunction(name) {
  const start = html.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${name} must exist`);
  const brace = html.indexOf('{', start);
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
    if (char === '}' && --depth === 0) return html.slice(start, index + 1);
  }
  throw new Error(`unterminated function ${name}`);
}

for (const id of ['g3dTaskControls', 'g3dTaskInstruction', 'g3dRunTaskBtn', 'g3dStopTaskBtn', 'g3dTaskStatus']) {
  assert.match(html, new RegExp(`id=["']${id}["']`));
}

const elements = {
  g3dTaskInstruction: {value: 'pick up white star to teal bowl'},
  g3dTargetHeight: {value: '1'},
  g3dSafetyHeight: {value: '10'},
  g3dTaskStatus: {textContent: '', style: {}},
  g3dRunTaskBtn: {disabled: false},
  g3dStopTaskBtn: {style: {display: 'none'}},
  g3dTaskControls: {style: {display: 'none'}},
  g3dTaskHint: {textContent: ''},
  g3dSceneInfo: {textContent: ''},
};
const applied = [];
const context = {
  G3D: {
    objects: [
      {name: 'white star', center_pixel: [392.5, 169]},
      {name: 'teal bowl', center_pixel: [509, 91]},
    ],
    imageSize: [800, 600],
    detectionId: 'det-1',
    robot: {},
  },
  document: {getElementById: id => elements[id]},
  g3dInitScene() {},
  g3dJointArray: joints => Object.values(joints),
  simApplyJoints: (robot, joints) => applied.push({robot, joints}),
  console,
};
vm.createContext(context);
for (const name of ['g3dSetTaskReady', 'g3dBuildTaskPayload', 'g3dTaskPhaseLabel', 'g3dApplyTaskStatus']) {
  vm.runInContext(`${extractFunction(name)}; this.${name} = ${name};`, context);
}

context.g3dSetTaskReady(true);
assert.equal(elements.g3dTaskControls.style.display, 'block');
assert.equal(elements.g3dRunTaskBtn.disabled, false);
assert.match(elements.g3dTaskHint.textContent, /white star.*teal bowl/);

assert.deepEqual(JSON.parse(JSON.stringify(context.g3dBuildTaskPayload())), {
  instruction: 'pick up white star to teal bowl',
  detection_id: 'det-1',
  target_height_cm: 1,
  safety_height_cm: 10,
});

context.g3dApplyTaskStatus({
  state: 'running',
  phase: 'moving_to_target',
  running: true,
  source: {name: 'white star'},
  target: {name: 'teal bowl'},
  joints: {shoulder_pan: 8.5, gripper: 0},
});
assert.equal(applied.length, 1);
assert.equal(elements.g3dRunTaskBtn.disabled, true);
assert.equal(elements.g3dStopTaskBtn.style.display, '');
assert.match(elements.g3dTaskStatus.textContent, /Moving to target/);
assert.match(elements.g3dSceneInfo.textContent, /white star.*teal bowl/);

context.g3dApplyTaskStatus({state: 'completed', phase: 'completed', running: false, joints: null});
assert.equal(elements.g3dRunTaskBtn.disabled, false);
assert.equal(elements.g3dStopTaskBtn.style.display, 'none');
assert.equal(elements.g3dTaskStatus.textContent, 'Task completed');

assert.match(extractFunction('g3dResumeTaskStatus'), /g3dPollTaskStatus/);

assert.match(extractFunction('g3dDetectObjects'), /g3dSetTaskReady\(/);
assert.match(extractFunction('g3dRunTask'), /\/api\/generate3d\/task\/start/);
assert.match(extractFunction('g3dStopTask'), /\/api\/generate3d\/task\/stop/);

console.log('Generate 3D task controls and live robot-state rendering pass');
