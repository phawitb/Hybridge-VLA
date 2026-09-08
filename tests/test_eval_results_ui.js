const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(new URL('../static/index.html', `file://${__dirname}/`), 'utf8');

function extractFunction(name) {
  const start = html.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${name} must exist`);
  if (name === 'evalResultsRender') {
    const end = html.indexOf('// ── Eval Results Charts ──', start);
    assert.notEqual(end, -1, 'evalResultsRender end marker must exist');
    return html.slice(start, end).trim();
  }
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

const tableHeaders = [...html.matchAll(/<table class="eval-res-table"[\s\S]*?<\/table>/g)]
  .map(match => match[0].match(/<thead>[\s\S]*?<\/thead>/)[0]);
assert.equal(tableHeaders.length, 3);
for (const header of tableHeaders) assert.match(header, />N_400<\/th>/);

function makeBody() {
  const thead = {
    querySelector: () => null,
    insertBefore: () => {},
    firstChild: null,
  };
  return {
    innerHTML: '',
    closest: () => ({querySelector: () => thead}),
  };
}

const bodies = [makeBody(), makeBody(), makeBody()];
const context = {
  console,
  evalResViewMode: false,
  evalResultsGetVisibility: () => ({}),
  evalResultsGetNotes: () => ({}),
  document: {
    getElementById: id => bodies[Number(id.slice(-1)) - 1],
    createElement: () => ({style: {}}),
  },
};
vm.createContext(context);
vm.runInContext(`${extractFunction('evalResultsRender')}; this.evalResultsRender = evalResultsRender;`, context);
context.evalResultsRender([{
  task: 'pick', policy: 'ACT', n: 400, a1: 3, a2: 0, a3: 0, fail: 1, total: 4,
}]);
for (const body of bodies) {
  assert.match(body.innerHTML, /75%/);
  assert.match(body.innerHTML, /\(3\/4\)/);
}

console.log('evaluate results UI includes N_400');
