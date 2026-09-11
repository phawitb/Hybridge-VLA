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

assert.match(html, /id="vizRemoveBeginningBtn"[^>]+onclick="vizShowRemoveBeginning\(\)"/);

const modalBody = {innerHTML: ''};
const context = {
  vizCurrentDataset: 'demo',
  document: {
    getElementById: id => id === 'modalBody' ? modalBody : null,
  },
};
vm.createContext(context);
vm.runInContext(`${extractFunction('vizRemoveBeginningDefaultName')}; this.nameFor = vizRemoveBeginningDefaultName;`, context);
assert.equal(context.nameFor(2), 'demo_remove_2s');
assert.equal(context.nameFor(1.5), 'demo_remove_1.5s');

console.log('dataset beginning-frame removal UI has action and editable default name');
