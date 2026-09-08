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

for (const id of ['g3dTaskControls', 'g3dTaskInstruction', 'g3dEnforceWorkspace', 'g3dRunTaskBtn', 'g3dStopTaskBtn', 'g3dTaskStatus', 'g3dAddObjectBtn']) {
  assert.match(html, new RegExp(`id=["']${id}["']`));
}

const elements = {
  g3dTaskInstruction: {value: ''},
  g3dEnforceWorkspace: {checked: true},
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
    instructionAuto: true,
    sceneGeneration: 4,
    renderedGeneration: 4,
    renderedDetectionId: 'det-1',
    hasRawDetection: true,
    sceneBusy: false,
    editSyncing: false,
    editsValid: true,
  },
  document: {getElementById: id => elements[id]},
  g3dInitScene() {},
  g3dJointArray: joints => Object.values(joints),
  simApplyJoints: (robot, joints) => applied.push({robot, joints}),
  console,
};
vm.createContext(context);
for (const name of ['g3dSceneIsCurrent', 'g3dRenderedSceneIsCurrent', 'g3dCanAddObject', 'g3dCanRunTask', 'g3dSetSceneBusy', 'g3dDefaultTaskInstruction', 'g3dUpdateDefaultInstruction', 'g3dNormalizeEditedBbox', 'g3dHitObject', 'g3dSetTaskReady', 'g3dBuildTaskPayload', 'g3dTaskPhaseLabel', 'g3dApplyTaskStatus']) {
  vm.runInContext(`${extractFunction(name)}; this.${name} = ${name};`, context);
}

assert.equal(context.g3dSceneIsCurrent(4, 'det-1'), true);
assert.equal(context.g3dSceneIsCurrent(3, 'det-1'), false);
assert.equal(context.g3dSceneIsCurrent(4, 'older'), false);
assert.equal(context.g3dRenderedSceneIsCurrent(), true);
context.G3D.renderedDetectionId = 'older';
assert.equal(context.g3dRenderedSceneIsCurrent(), false);
context.G3D.renderedDetectionId = 'det-1';
context.G3D.imageElement = {};
context.G3D.detectionId = null;
context.G3D.renderedDetectionId = null;
assert.equal(context.g3dCanAddObject(), true);
context.G3D.detectionId = 'det-1';
context.G3D.renderedDetectionId = 'det-1';
assert.equal(context.g3dCanRunTask(), true);
context.G3D.sceneBusy = true;
assert.equal(context.g3dCanRunTask(), false);
context.G3D.sceneBusy = false;

context.g3dSetTaskReady(true);
assert.equal(elements.g3dTaskControls.style.display, 'block');
assert.equal(elements.g3dRunTaskBtn.disabled, false);
assert.match(elements.g3dTaskHint.textContent, /white star.*teal bowl/);
assert.equal(elements.g3dTaskInstruction.value, 'pick up white star to teal bowl');

assert.deepEqual(JSON.parse(JSON.stringify(context.g3dBuildTaskPayload())), {
  instruction: 'pick up white star to teal bowl',
  detection_id: 'det-1',
  target_height_cm: 1,
  safety_height_cm: 10,
  enforce_workspace: true,
});
elements.g3dEnforceWorkspace.checked = false;
assert.equal(context.g3dBuildTaskPayload().enforce_workspace, false);

elements.g3dTaskInstruction.value = 'custom instruction';
context.G3D.instructionAuto = false;
context.G3D.objects[0].name = 'yellow star';
context.g3dUpdateDefaultInstruction();
assert.equal(elements.g3dTaskInstruction.value, 'custom instruction');

context.G3D.objects[0].bbox = [100, 120, 300, 320];
assert.deepEqual(JSON.parse(JSON.stringify(context.g3dNormalizeEditedBbox([300, 320, 100, 120], 800, 600))), [100, 120, 300, 320]);
assert.deepEqual(JSON.parse(JSON.stringify(context.g3dHitObject([200, 200]))), {index: 0, mode: 'move'});
assert.deepEqual(JSON.parse(JSON.stringify(context.g3dHitObject([100, 120]))), {index: 0, mode: 'resize', corner: 'nw'});

context.G3D.editsValid = false;
context.g3dSetTaskReady(true);
assert.equal(elements.g3dRunTaskBtn.disabled, true);
context.G3D.editsValid = true;

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
assert.match(extractFunction('g3dSyncEditedObjects'), /\/api\/generate3d\/detection\/manual/);
assert.match(extractFunction('g3dRunTask'), /\/api\/generate3d\/task\/start/);
assert.match(extractFunction('g3dStopTask'), /\/api\/generate3d\/task\/stop/);

console.log('Generate 3D task controls and live robot-state rendering pass');
