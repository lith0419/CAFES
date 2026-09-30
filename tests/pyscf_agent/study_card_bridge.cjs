// Exercise the bundled card against a small, explicit MCP Apps host fixture.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../../pyscf_agent/web_assets/study-card.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const tick = () => new Promise(resolve => setImmediate(resolve));
function host(capabilities = {serverTools: {}, openLinks: {}}) {
  const nodes = {}, sent = [], timers = new Map();
  let listener, timerId = 0;
  function node() {
    return {textContent: '', disabled: true, hidden: true, children: [], events: {},
      append(...items) {this.children.push(...items);}, replaceChildren() {this.children = [];},
      addEventListener(type, fn) {this.events[type] = fn;},
      set innerHTML(value) {throw Error('Unsafe HTML rendering: ' + value);}};
  }
  const parent = {postMessage: item => sent.push(item)};
  const context = {
    document: {getElementById: id => nodes[id] ||= node(), createElement: node,
      documentElement: {style: {setProperty() {}}, clientWidth: 700}, body: {scrollHeight: 450}},
    window: {parent, addEventListener: (type, fn) => {listener = fn;}},
    ResizeObserver: class {observe() {} disconnect() {}}, URL, Date,
    setTimeout: fn => {timers.set(++timerId, fn); return timerId;}, clearTimeout: id => timers.delete(id),
  };
  vm.runInNewContext(source, context);
  const emit = (data, source = parent) => listener({source, data: {jsonrpc: '2.0', ...data}});
  const reply = (request, result) => emit({id: request.id, result});
  const data = {study_id: 'saved-study', name: '<img src=x onerror=bad()>', mode: 'static',
    execution_status: 'running', study_status: 'failed', task_status_counts: {pending: 1, succeeded: 1},
    count_source: 'agent_task_state', rows_source: 'saved_report', case_count: 1, can_open_workbench: true,
    checked_at: '2026-09-16T12:00:00Z', rows: [{label: '<script>bad()</script>', method: 'casscf',
      active_space: {nelecas: [1, 1], ncas: 2}, execution_status: 'failed', energy: 0, energy_unit: 'Ha', converged: false}]};
  return {nodes, sent, emit, reply, timers, data, async initialize() {
    assert.equal(sent[0].method, 'ui/initialize');
    assert.equal(sent[0].params.protocolVersion, '2026-01-26');
    reply(sent[0], {protocolVersion: '2026-01-26', hostCapabilities: capabilities, hostContext: {theme: 'dark'}});
    await tick();
    assert.equal(sent[1].method, 'ui/notifications/initialized');
    emit({method: 'ui/notifications/tool-input', params: {arguments: {study_id: 'saved-study'}}});
    emit({method: 'ui/notifications/tool-result', params: {structuredContent: data}});
  }};
}
async function main() {
  const h = host(); await h.initialize();
  assert.equal(h.nodes.name.textContent, h.data.name);
  assert.equal(h.nodes.rows.children[0].children[0].textContent, '<script>bad()</script>');
  assert.equal(h.nodes.rows.children[0].children[4].textContent, '0.0000000000 Ha');
  assert.equal(h.nodes.rows.children[0].children[5].textContent, 'No');
  assert.match(h.nodes['results-note'].textContent, /precede an active retry/);
  h.emit({method: 'ui/notifications/tool-result', params: {structuredContent: {...h.data, name: 'spoof'}}}, {});
  h.emit({method: 'ui/notifications/tool-result', params: {structuredContent: {...h.data, study_id: 'other'}}});
  assert.equal(h.nodes.name.textContent, h.data.name);
  const refresh = h.nodes.refresh.events.click();
  assert.equal(h.sent.at(-1).params.name, 'refresh_study_view');
  assert.equal(h.sent.at(-1).params.arguments.study_id, 'saved-study');
  assert.equal(h.nodes.refresh.disabled, true);
  const n = h.sent.length; await h.nodes.refresh.events.click(); assert.equal(h.sent.length, n);
  h.reply(h.sent.at(-1), {structuredContent: {...h.data, execution_status: 'results_available'}});
  await refresh;
  assert.equal(h.nodes.execution.textContent, 'Results available');
  const failed = h.nodes.refresh.events.click();
  h.reply(h.sent.at(-1), {isError: true, content: [{type: 'text', text: 'Connection unavailable'}]});
  await failed;
  assert.equal(h.nodes.message.textContent, 'Connection unavailable');
  assert.equal(h.nodes.name.textContent, h.data.name); // Preserve last visible result.
  assert.equal(h.nodes.refresh.disabled, false);
  const open = h.nodes.open.events.click();
  assert.equal(h.sent.at(-1).params.name, 'open_workbench');
  h.reply(h.sent.at(-1), {structuredContent: {url: 'http://127.0.0.1:53407/computational-study/?study=saved-study'}});
  await tick();
  assert.equal(h.sent.at(-1).method, 'ui/open-link');
  h.emit({id: h.sent.at(-1).id, error: {message: 'Link denied'}});
  await open;
  assert.equal(h.nodes.message.textContent, 'Link denied');
  assert.equal(h.nodes.link.hidden, false);
  const timeout = h.nodes.refresh.events.click();
  [...h.timers.values()][0](); await timeout;
  assert.match(h.nodes.message.textContent, /did not respond/);
  h.emit({id: 90, method: 'ui/resource-teardown'});
  assert.equal(h.sent.at(-1).id, 90);
  assert.equal(h.nodes.refresh.disabled, true);
  const readonly = host({}); await readonly.initialize();
  assert.equal(readonly.nodes.refresh.disabled, true);
  assert.match(readonly.nodes.message.textContent, /does not support card actions/);
  const linkOnly = host({serverTools: {}}); await linkOnly.initialize();
  const fallback = linkOnly.nodes.open.events.click();
  linkOnly.reply(linkOnly.sent.at(-1), {structuredContent: {url: 'http://127.0.0.1:1/'}});
  await fallback;
  assert.equal(linkOnly.nodes.link.hidden, false);
  assert.match(linkOnly.nodes.message.textContent, /link below/);
  assert.deepEqual([...new Set(h.sent.filter(x => x.method === 'tools/call').map(x => x.params.name))],
    ['refresh_study_view', 'open_workbench']);
  console.log('Study card bridge: lifecycle, refresh, navigation, errors and safe text rendering passed');
}
main().catch(error => {console.error(error); process.exitCode = 1;});
