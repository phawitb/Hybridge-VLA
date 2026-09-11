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

assert.match(html, /id="trainCommandBtn"[^>]+onclick="trainShowServerCommand\(\)"/);
assert.match(html, /id="trainServerCommand"/);

const context = {};
vm.createContext(context);
const buildStart = html.indexOf('function trainBuildServerCommand(');
const buildEnd = html.indexOf('\nfunction trainShowServerCommand(', buildStart);
assert.notEqual(buildStart, -1);
assert.notEqual(buildEnd, -1);
vm.runInContext(`${html.slice(buildStart, buildEnd)}; this.build = trainBuildServerCommand;`, context);

const command = context.build({
  policy_type: 'smolvla',
  dataset_repo_id: 'demo-user/cups',
  dataset_root: './data/cups',
  use_amp: true,
  steps: 2500,
  batch_size: 8,
  optimizer_lr: 0.0002,
  wandb_enable: false,
  wandb_project: 'ignored-project',
  resume: true,
  job_name: 'smolvla_cups_test',
  policy_repo_id: 'demo-user/smolvla_cups',
  vlm_model_name: 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
  load_vlm_weights: true,
  train_expert_only: false,
  freeze_vision_encoder: true,
  chunk_size: 40,
  n_action_steps: 20,
  gradient_checkpointing: true,
});

assert.match(command, /^cd \/home\/jupyter-phawit\/so101-v2\nconda run -n lerobot lerobot-train \\\n/);
assert.match(command, /--policy\.type=smolvla/);
assert.match(command, /--dataset\.repo_id=demo-user\/cups/);
assert.match(command, /--dataset\.root=\.\/data\/cups/);
assert.match(command, /--policy\.load_vlm_weights=true/);
assert.match(command, /--policy\.train_expert_only=false/);
assert.match(command, /--policy\.freeze_vision_encoder=true/);
assert.match(command, /--policy\.chunk_size=40/);
assert.match(command, /--policy\.n_action_steps=20/);
assert.match(command, /--policy\.gradient_checkpointing=true/);
assert.match(command, /--wandb\.enable=false/);
assert.doesNotMatch(command, /--wandb\.project=/);
assert.match(command, /--resume=true/);
assert.match(command, /--output_dir=outputs\/train\/smolvla_cups_test/);
assert.match(command, /--policy\.repo_id=demo-user\/smolvla_cups/);
assert.match(command, /--resume=true$/);

const piCommand = context.build({
  policy_type: 'pi05',
  dataset_repo_id: 'demo-user/blocks',
  dataset_root: './data/blocks',
  use_amp: true,
  steps: 100,
  batch_size: 2,
  optimizer_lr: 0.0003,
  wandb_enable: true,
  wandb_project: 'Robots',
  resume: false,
  job_name: 'pi05_blocks',
  policy_repo_id: 'demo-user/pi05_blocks',
  paligemma_variant: 'paligemma2-3b-pt-224',
  action_expert_variant: 'gemma-2-2b-it',
});

assert.match(piCommand, /--policy\.type=pi0/);
assert.match(piCommand, /--policy\.pretrained_path=lerobot\/pi0\.5/);
assert.match(piCommand, /--policy\.paligemma_variant=paligemma2-3b-pt-224/);
assert.match(piCommand, /--policy\.action_expert_variant=gemma-2-2b-it/);
assert.match(piCommand, /--wandb\.enable=true/);
assert.match(piCommand, /--wandb\.project=Robots/);
assert.doesNotMatch(piCommand, /--resume=true/);

console.log('training page builds direct server commands from the selected configuration');
