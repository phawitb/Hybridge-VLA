const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(new URL('../static/index.html', `file://${__dirname}/`), 'utf8');
const match = html.match(/function calConnectWs\(\)\{[\s\S]*?\n\}/);
assert.ok(match, 'calConnectWs must exist');
assert.doesNotMatch(html, /id=["']g3dCaptureBtn["']/, 'Generate 3D calibration must not require a capture button');
assert.match(match[0], /g3dShowLiveCalibFrame\(m\.data\.top\)/, 'top-camera frames must update Generate 3D calibration');

let openedUrl = null;
const status = {};
const context = {
  location: { protocol: 'https:', host: 'robot.example.test' },
  WebSocket: function WebSocket(url) {
    openedUrl = url;
    return {};
  },
  document: {
    getElementById(id) {
      if (id === 'calStatus') return status;
      return { src: '' };
    },
  },
  JSON,
  Object,
  setTimeout() {},
  calRunning: false,
  calUpd() {},
  console,
};

vm.runInNewContext(`${match[0]}; calConnectWs();`, context);
assert.equal(openedUrl, 'wss://robot.example.test/ws/robot');
console.log('camera WebSocket follows the page security protocol');
