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

for (const id of ['g3dTaskControls', 'g3dTaskInstruction', 'g3dTaskPickHeight', 'g3dTaskPlaceHeight', 'g3dTaskSafetyHeight', 'g3dMotionSmooth', 'g3dMotionWaypoint', 'g3dExecutionSimulation', 'g3dExecutionReal', 'g3dEnforceWorkspace', 'g3dRunTaskBtn', 'g3dStopTaskBtn', 'g3dTaskStatus', 'g3dAddObjectBtn', 'g3dDetectBtn']) {
  assert.match(html, new RegExp(`id=["']${id}["']`));
}
assert.ok(
  html.indexOf('id="g3dTaskInstruction"') < html.indexOf('id="g3dDetectBtn"'),
  'Task instruction must appear before Detect & Generate',
);

const elements = {
  g3dTaskInstruction: {value: 'pick up white star to teal bowl'},
  g3dExecutionSimulation: {checked: true},
  g3dExecutionReal: {checked: false},
  g3dEnforceWorkspace: {checked: true},
  g3dTargetHeight: {value: '1'},
  g3dSafetyHeight: {value: '10'},
  g3dTaskPickHeight: {value: '0'},
  g3dTaskPlaceHeight: {value: '5'},
  g3dTaskSafetyHeight: {value: '18'},
  g3dMotionSmooth: {checked: true},
  g3dMotionWaypoint: {checked: false},
  g3dTaskStatus: {textContent: '', style: {}},
  g3dRunTaskBtn: {disabled: false},
  g3dStopTaskBtn: {style: {display: 'none'}},
  g3dTaskControls: {style: {display: 'none'}},
  g3dTaskHint: {textContent: ''},
  g3dPathHint: {textContent: ''},
  g3dSceneInfo: {textContent: ''},
  g3dImageCanvas: {width: 0, height: 0, style: {}},
  g3dImagePlaceholder: {style: {}},
  g3dAddObjectBtn: {disabled: true},
  g3dDetectBtn: {disabled: true},
  g3dCaptureSceneBtn: {disabled: false},
  g3dImageInput: {disabled: false},
  g3dEditHint: {style: {}},
};
const applied = [];
const context = {
  G3D: {
    objects: [
      {name: 'white star', center_pixel: [392.5, 169], position_3d: [0.12, 0, 0.08], estimated_size_cm: [3, 3, 3]},
      {name: 'teal bowl', center_pixel: [509, 91], position_3d: [-0.09, 0, 0.14], estimated_size_cm: [6, 6, 6]},
    ],
    imageSize: [800, 600],
    currentJoints: {
      shoulder_pan: 4,
      shoulder_lift: -12,
      elbow_flex: 18,
      wrist_flex: 25,
      wrist_roll: 3,
      gripper: 35,
    },
    detectionId: 'det-1',
    robot: {},
    objectVisuals: [
      {mesh: {position: {x: 0.12, y: 0.015, z: 0.08}}, sprite: {position: {x: 0.12, y: 0.047, z: 0.08}}, height: 0.03},
      {mesh: {position: {x: -0.09, y: 0.03, z: 0.14}}, sprite: {position: {x: -0.09, y: 0.077, z: 0.14}}, height: 0.06},
    ],
    instructionAuto: true,
    sceneGeneration: 4,
    renderedGeneration: 4,
    renderedDetectionId: 'det-1',
    hasRawDetection: true,
    sceneBusy: false,
    editSyncing: false,
    editsValid: true,
    imageFile: {},
    taskRunning: false,
  },
  document: {getElementById: id => elements[id]},
  g3dInitScene() {},
  g3dPreviewTaskPath() {},
  g3dJointArray: joints => Object.values(joints),
  simApplyJoints: (robot, joints) => applied.push({robot, joints}),
  g3dRedrawImage() {},
  Image: class {
    set src(_value) {
      this.width = 1024;
      this.height = 768;
      this.naturalWidth = 1024;
      this.naturalHeight = 768;
      this.onload();
    }
  },
  console,
};
vm.createContext(context);
for (const name of ['g3dSceneIsCurrent', 'g3dRenderedSceneIsCurrent', 'g3dCanAddObject', 'g3dCanDetect', 'g3dHasCompleteTaskObjects', 'g3dCanRunTask', 'g3dSetSceneBusy', 'g3dDefaultTaskInstruction', 'g3dUpdateDefaultInstruction', 'g3dNormalizeEditedBbox', 'g3dHitObject', 'g3dSetTaskReady', 'g3dResolveTaskObjects', 'g3dSelectedMotionMode', 'g3dSmoothstep', 'g3dSampleSegment', 'g3dBuildTaskPathPoints', 'g3dBuildTaskPayload', 'g3dTaskPhaseLabel', 'g3dPlaceSourceAtTarget', 'g3dApplyTaskStatus']) {
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
elements.g3dTaskInstruction.value = '   ';
assert.equal(context.g3dCanDetect(), false);
elements.g3dTaskInstruction.value = '  pick up white star to teal bowl  ';
assert.equal(context.g3dCanDetect(), true);
assert.equal(context.g3dCanRunTask(), true);
context.G3D.sceneBusy = true;
assert.equal(context.g3dCanRunTask(), false);
context.G3D.sceneBusy = false;

context.g3dSetTaskReady(true);
assert.equal(elements.g3dTaskControls.style.display, 'block');
assert.equal(elements.g3dRunTaskBtn.disabled, false);
assert.match(elements.g3dTaskHint.textContent, /white star.*teal bowl/);
assert.equal(elements.g3dTaskInstruction.value, '  pick up white star to teal bowl  ');

assert.deepEqual(JSON.parse(JSON.stringify(context.g3dBuildTaskPayload())), {
  instruction: 'pick up white star to teal bowl',
  detection_id: 'det-1',
  pick_height_cm: 0,
  place_height_cm: 5,
  safety_height_cm: 18,
  enforce_workspace: true,
  execution_mode: 'simulation',
  motion_mode: 'smooth',
  initial_joints: {
    shoulder_pan: 4,
    shoulder_lift: -12,
    elbow_flex: 18,
    wrist_flex: 25,
    wrist_roll: 3,
    gripper: 35,
  },
});
elements.g3dMotionSmooth.checked = false;
elements.g3dMotionWaypoint.checked = true;
assert.equal(context.g3dBuildTaskPayload().motion_mode, 'waypoint');
elements.g3dMotionSmooth.checked = true;
elements.g3dMotionWaypoint.checked = false;

assert.deepEqual(JSON.parse(JSON.stringify(context.g3dResolveTaskObjects('pick up white star to teal bowl'))), {
  source: context.G3D.objects[0],
  target: context.G3D.objects[1],
});
context.G3D.objects[0].task_role = 'source';
assert.equal(context.g3dHasCompleteTaskObjects(), false);
context.G3D.objects[1].task_role = 'target';
assert.equal(context.g3dHasCompleteTaskObjects(), true);
assert.deepEqual(JSON.parse(JSON.stringify(context.g3dResolveTaskObjects('names do not match'))), {
  source: context.G3D.objects[0],
  target: context.G3D.objects[1],
});
context.G3D.objects[0].position_valid = false;
assert.equal(context.g3dHasCompleteTaskObjects(), false);
delete context.G3D.objects[0].position_valid;
delete context.G3D.objects[0].task_role;
delete context.G3D.objects[1].task_role;
assert.deepEqual(JSON.parse(JSON.stringify(context.g3dBuildTaskPathPoints(
  context.G3D.objects[0], context.G3D.objects[1], 0, 5, 18, 'waypoint',
))), [
  [0.12, 0, 0.08],
  [0.12, 0.18, 0.08],
  [-0.09, 0.18, 0.14],
  [-0.09, 0.05, 0.14],
]);
const smoothPath = JSON.parse(JSON.stringify(context.g3dBuildTaskPathPoints(
  context.G3D.objects[0], context.G3D.objects[1], 0, 5, 18, 'smooth',
)));
assert.deepEqual(smoothPath[0], [0.12, 0, 0.08]);
assert.deepEqual(smoothPath.at(-1), [-0.09, 0.05, 0.14]);
assert.equal(smoothPath.length, 25);
assert.ok(smoothPath.slice(0, 7).every(point => point[0] === 0.12 && point[2] === 0.08));
assert.ok(smoothPath.slice(7, 19).every(point => point[1] >= 0.18));
assert.ok(smoothPath.slice(19).every(point => point[0] === -0.09 && point[2] === 0.14));

const addedPathVisuals = [];
context.G3D.inited = true;
context.G3D.pathVisuals = [];
context.G3D.scene = {
  add: object => addedPathVisuals.push(object),
  remove: object => addedPathVisuals.splice(addedPathVisuals.indexOf(object), 1),
};
context.THREE = {
  Vector3: class { constructor(...values) { this.values = values; } },
  BufferGeometry: class { setFromPoints(points) { this.points = points; return this; } },
  LineDashedMaterial: class { constructor(options) { this.options = options; } },
  Line: class { constructor(geometry, material) { this.geometry = geometry; this.material = material; } computeLineDistances() {} },
  SphereGeometry: class { constructor(...values) { this.values = values; } },
  MeshBasicMaterial: class { constructor(options) { this.options = options; } },
  Mesh: class { constructor(geometry, material) { this.geometry = geometry; this.material = material; this.position = {set(...values) { this.values = values; }}; } },
};
vm.runInContext(`${extractFunction('g3dClearTaskPath')}; this.g3dClearTaskPath = g3dClearTaskPath;`, context);
vm.runInContext(`${extractFunction('g3dPreviewTaskPath')}; this.g3dPreviewTaskPath = g3dPreviewTaskPath;`, context);
assert.equal(context.g3dPreviewTaskPath(), true);
assert.equal(context.G3D.pathVisuals.length, 4);
assert.match(elements.g3dPathHint.textContent, /pick 0 cm.*place 5 cm.*transfer 18 cm/);
elements.g3dExecutionReal.checked = true;
assert.equal(context.g3dBuildTaskPayload().execution_mode, 'real');
elements.g3dExecutionReal.checked = false;
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
context.G3D.objects[0].name = 'white star';
elements.g3dTaskInstruction.value = 'pick up white star to teal bowl';

context.g3dApplyTaskStatus({
  state: 'running',
  phase: 'moving_to_target',
  running: true,
  source: {name: 'white star'},
  target: {name: 'teal bowl'},
  joints: {shoulder_pan: 8.5, gripper: 0},
});
assert.equal(applied.length, 1);
assert.deepEqual(JSON.parse(JSON.stringify(context.G3D.currentJoints)), {shoulder_pan: 8.5, gripper: 0});
assert.equal(elements.g3dRunTaskBtn.disabled, true);
assert.equal(elements.g3dStopTaskBtn.style.display, '');
assert.match(elements.g3dTaskStatus.textContent, /Moving to target/);
assert.match(elements.g3dSceneInfo.textContent, /white star.*teal bowl/);

context.g3dApplyTaskStatus({
  state: 'running',
  phase: 'releasing',
  running: true,
  execution_mode: 'simulation',
  source: {name: 'white star'},
  target: {name: 'teal bowl'},
  joints: null,
});
assert.deepEqual(context.G3D.objectVisuals[0].mesh.position, {x: -0.09, y: 0.075, z: 0.14});
assert.deepEqual(context.G3D.objectVisuals[0].sprite.position, {x: -0.09, y: 0.107, z: 0.14});

context.g3dApplyTaskStatus({state: 'completed', phase: 'completed', running: false, joints: null});
assert.equal(elements.g3dRunTaskBtn.disabled, false);
assert.equal(elements.g3dStopTaskBtn.style.display, 'none');
assert.equal(elements.g3dTaskStatus.textContent, 'Task completed');

assert.match(extractFunction('g3dResumeTaskStatus'), /g3dPollTaskStatus/);

assert.match(extractFunction('g3dDetectObjects'), /g3dSetTaskReady\(/);
assert.match(extractFunction('g3dDetectObjects'), /append\(['"]instruction['"],\s*instruction\)/);
assert.match(extractFunction('g3dDetectObjects'), /d\.pick_height_cm/);
assert.match(extractFunction('g3dDetectObjects'), /d\.place_height_cm/);
assert.doesNotMatch(extractFunction('g3dDetectObjects'), /recommended_pick_height_cm/);
assert.doesNotMatch(extractFunction('g3dDetectObjects'), /recommended_place_height_cm/);
assert.match(extractFunction('g3dDetectObjects'), /g3dSetTaskReady\(ready\)/);
assert.doesNotMatch(extractFunction('g3dDetectObjects'), /g3dUpdateDefaultInstruction/);
assert.doesNotMatch(extractFunction('g3dCaptureSceneImage'), /g3dTaskInstruction['"]\)\.value\s*=\s*['"]/);
assert.match(html, /g3dTaskInstruction['"]\)\.addEventListener\(['"]input['"][\s\S]*?G3D\.detectionId\s*=\s*null/);
assert.match(extractFunction('g3dSyncEditedObjects'), /\/api\/generate3d\/detection\/manual/);
assert.match(extractFunction('g3dRunTask'), /\/api\/generate3d\/task\/start/);
assert.match(extractFunction('g3dStopTask'), /\/api\/generate3d\/task\/stop/);

vm.runInContext(`${extractFunction('g3dDrawInputImage')}; this.g3dDrawInputImage = g3dDrawInputImage;`, context);
context.G3D.imageSize = [640, 480];
context.G3D.detectionId = null;
context.G3D.renderedDetectionId = null;
context.g3dDrawInputImage('uploaded-image', 4, null).then(drawn => {
  assert.equal(drawn, true);
  assert.deepEqual(JSON.parse(JSON.stringify(context.G3D.imageSize)), [1024, 768]);
  console.log('Generate 3D task controls and live robot-state rendering pass');
});
