const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(new URL('../static/index.html', `file://${__dirname}/`), 'utf8');

function extractFunction(name) {
  const start = html.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${name} must exist`);
  const brace = html.indexOf('{', start);
  let depth = 0, quote = null, escaped = false;
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

const flowStart = html.indexOf('id="inner-g3d-flow"');
const detected = html.indexOf('<h3>Detected Objects</h3>', flowStart);
const controls = html.indexOf('<h3>Flow Controls</h3>', flowStart);
const workspace = html.indexOf('<h3>3D Workspace</h3>', flowStart);
assert.ok(detected < controls && controls < workspace, 'Flow Controls must be directly below Detected Objects in the left column');
assert.match(html.match(/<input[^>]*id="g3dFlowExecutionReal"[^>]*>/)[0], /\schecked(?:\s|>)/);
assert.doesNotMatch(html.match(/<input[^>]*id="g3dFlowExecutionSimulation"[^>]*>/)[0], /\schecked(?:\s|>)/);
assert.doesNotMatch(html.match(/<input[^>]*id="g3dFlowSkipVerification"[^>]*>/)[0], /\schecked(?:\s|>)/);

const configElements = {
  g3dFlowExecutionReal: {checked: true}, g3dFlowMotionWaypoint: {checked: false},
  g3dFlowModel: {value: ''}, g3dFlowPickHeight: {value: '0'},
  g3dFlowPlaceHeight: {value: ''}, g3dFlowSafetyHeight: {value: '10'},
  g3dFlowEnforceWorkspace: {checked: false}, g3dFlowReturnRest: {checked: true},
  g3dFlowSkipVerification: {checked: true},
};
const configContext = {
  G3D: {currentJoints: {shoulder_pan: 12}},
  document: {getElementById: id => configElements[id]},
};
vm.createContext(configContext);
vm.runInContext(`${extractFunction('g3dFlowConfig')}; this.g3dFlowConfig = g3dFlowConfig;`, configContext);
assert.equal(configContext.g3dFlowConfig().skip_verification, true);

const elements = {
  g3dFlowPlanBtn: {}, g3dFlowRunAllBtn: {}, g3dFlowRunBlockBtn: {},
  g3dFlowRetryBtn: {}, g3dFlowStopBtn: {}, g3dFlowInstruction: {value: 'task'},
};
const context = {
  G3D_FLOW: {imageB64: 'image', selectedIndex: 0, state: {running: false, blocks: [{phase: 'success'}]}},
  document: {getElementById: id => elements[id]},
};
vm.createContext(context);
vm.runInContext(`${extractFunction('g3dFlowUpdateControls')}; this.g3dFlowUpdateControls = g3dFlowUpdateControls;`, context);
context.g3dFlowUpdateControls();
assert.equal(elements.g3dFlowRunBlockBtn.disabled, false, 'a successful block can be run again');

const objectContext = {};
vm.createContext(objectContext);
vm.runInContext(`${extractFunction('g3dFlowAllObjects')}; this.g3dFlowAllObjects = g3dFlowAllObjects;`, objectContext);
const allObjects = JSON.parse(JSON.stringify(objectContext.g3dFlowAllObjects({blocks: [
  {objects: [
    {name: 'blue star', task_role: 'source', bbox: [1, 2, 3, 4]},
    {name: 'green bowl', task_role: 'target', bbox: [5, 6, 7, 8]},
  ]},
  {objects: [
    {name: 'pink bow', task_role: 'source', bbox: [9, 10, 11, 12]},
    {name: 'green bowl', task_role: 'target', bbox: [13, 14, 15, 16]},
  ]},
]}, 0)));
assert.deepEqual(allObjects.map(object => object.name), ['blue star', 'green bowl', 'pink bow']);
assert.equal(allObjects.filter(object => object.name === 'green bowl').length, 1);
assert.deepEqual(allObjects.find(object => object.name === 'green bowl').bbox, [5, 6, 7, 8]);
assert.deepEqual(allObjects.map(object => object.selected), [true, true, false]);
assert.match(extractFunction('g3dFlowDrawDetections'), /g3dFlowAllObjects/);
assert.match(extractFunction('g3dFlowRenderPath'), /g3dFlowAllObjects/);
assert.match(extractFunction('g3dFlowRender'), /g3dFlowAllObjects/);

const poseContext = {
  G3D: {currentJoints: {}, robot: {}},
  document: {getElementById: () => ({textContent: ''})},
  fetch: async () => ({json: async () => ({ok: true, joints: {shoulder_pan: 12, gripper: 44}, position_3d: [0.1, 0, 0.2]})}),
  g3dJointArray: joints => Object.values(joints),
  simApplyJoints: (_robot, joints) => { poseContext.applied = joints; },
};
vm.createContext(poseContext);
vm.runInContext(`async ${extractFunction('g3dLoadCurrentRobotPose')}; this.g3dLoadCurrentRobotPose = g3dLoadCurrentRobotPose;`, poseContext);
poseContext.g3dLoadCurrentRobotPose().then(() => {
  assert.deepEqual(JSON.parse(JSON.stringify(poseContext.G3D.currentJoints)), {shoulder_pan: 12, gripper: 44});
  assert.deepEqual(poseContext.applied, [12, 44]);
  assert.match(extractFunction('g3dInitScene'), /g3dLoadCurrentRobotPose\(/);
  console.log('Generate 3D flow UI behavior passes');
});
