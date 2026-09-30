const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../pyscf_agent/web_assets/task-monitor.js'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
const handle = {job_id: 'job', run_id: 'job', executor_id: 'local-process', work_dir: '/runs', submitted_at: 'now'};
const data = () => ({handle, status: {state: 'running', terminal: false, report_available: false},
  elapsed_seconds: 0, checked_at: '2026-09-20T00:00:00Z', can_open_workbench: true,
  summary: null, solver_progress: null, logs: [], notices: []});

function environment(embedded, initial = data(), capabilities = {serverTools: {}, openLinks: {}}) {
  const nodes = {}, events = {}, documentEvents = {}, timers = new Map(), sent = [], requests = [];
  let timerId = 0, next = initial;
  function node() { return {textContent: '', disabled: false, hidden: false, checked: true, children: [], dataset: {}, events: {},
    append(...items) { this.children.push(...items); }, replaceChildren() { this.children = []; },
    addEventListener(type, fn) { this.events[type] = fn; },
    set innerHTML(value) { throw new Error('Unsafe HTML: ' + value); }}; }
  const parent = {postMessage: message => sent.push(message)};
  const win = {location: {hash: '#' + new URLSearchParams({handle: JSON.stringify(handle), execution_target: 'local'})},
    addEventListener: (name, fn) => { events[name] = fn; }};
  win.parent = embedded ? parent : win;
  const document = {hidden: false, getElementById: id => nodes[id] ||= node(), createElement: node,
    addEventListener: (name, fn) => { documentEvents[name] = fn; }, documentElement: {style: {}, clientWidth: 600}, body: {scrollHeight: 500}};
  vm.runInNewContext(source, {window: win, document, Date, URL, URLSearchParams, AbortController,
    ResizeObserver: class {observe() {} disconnect() {}},
    setTimeout: (fn, delay) => { timers.set(++timerId, {fn, delay}); return timerId; }, clearTimeout: id => timers.delete(id),
    fetch: async (url, options) => { requests.push({url, options}); return {ok: true, json: async () => next}; }});
  const emit = (value, origin = parent) => events.message({source: origin, data: {jsonrpc: '2.0', ...value}});
  const reply = (message, result) => emit({id: message.id, result});
  const runTimer = async delay => {
    const entry = [...timers].find(([, value]) => value.delay === delay);
    assert.ok(entry, 'Expected timer ' + delay);
    timers.delete(entry[0]); const result = entry[1].fn(); await tick(); return result;
  };
  return {nodes, sent, timers, document, events, documentEvents, requests, emit, reply, runTimer,
    setNext(value) { next = value; },
    async init() {
      if (embedded) {
        reply(sent[0], {protocolVersion: '2026-01-26', hostCapabilities: capabilities});
        await tick(); emit({method: 'ui/notifications/tool-input', params: {arguments: {handle}}});
        emit({method: 'ui/notifications/tool-result', params: {structuredContent: initial}});
      }
      await tick();
    }};
}

async function main() {
  const browser = environment(false); await browser.init();
  assert.equal(browser.requests.length, 1);
  assert.equal(browser.requests[0].url, '/api/task-view');
  assert.equal(browser.nodes.elapsed.textContent, '0m 0s');
  assert.equal(browser.nodes['sweep-energy'].textContent, '');
  await browser.runTimer(15000); assert.equal(browser.requests.length, 2);
  browser.document.hidden = true; browser.documentEvents.visibilitychange(); assert.equal(browser.timers.size, 0);
  browser.document.hidden = false; browser.documentEvents.visibilitychange(); assert.equal(browser.timers.size, 1);
  browser.nodes.auto.checked = false; browser.nodes.auto.events.change(); assert.equal(browser.timers.size, 0);
  browser.nodes.auto.checked = true; browser.nodes.auto.events.change();
  const failed = {...data(), status: {state: 'completed', terminal: true, report_available: true, task_status: 'failed'},
    summary: {execution_status: 'failed', energy: null, errors: [{message: '<script>bad()</script>'}]},
    solver_progress: {observed_sweep: {sweep_index: 8, bond_dimension: 2000},
      last_completed_sweep: {sweep_index: 7, bond_dimension: 2000, energy: -7, energy_change: 0, discarded_weight: 0},
      source: 'solver.log', logged_at: '2026-09-20T00:00:00Z', note: 'Last observed'},
    logs: [{path: 'solver.log', updated_at: '2026-09-20T00:00:00Z', tail: '<img onerror=bad()>'}]};
  browser.setNext(failed); await browser.runTimer(15000);
  assert.equal(browser.nodes.outcome.textContent, 'Failed');
  assert.equal(browser.nodes.scheduler.textContent, 'Completed');
  assert.equal(browser.nodes.errors.textContent, '<script>bad()</script>');
  assert.equal(browser.nodes.logs.children[0].children[1].textContent, '<img onerror=bad()>');
  assert.equal(browser.nodes.delta.textContent, '0.000e+0');
  assert.equal(browser.nodes['result-energy'].textContent, '');
  assert.equal(browser.timers.size, 0);
  await browser.nodes.refresh.events.click(); assert.equal(browser.requests.length, 4);
  assert.equal(browser.timers.size, 0);

  const h = environment(true); await h.init();
  // Timer launches one request. A manual click while in flight cannot overlap.
  const entry = [...h.timers].find(([, v]) => v.delay === 15000);
  h.timers.delete(entry[0]); const pending = entry[1].fn();
  const request = h.sent.at(-1);
  assert.equal(request.params.name, 'refresh_task_view');
  await h.nodes.refresh.events.click(); assert.equal(h.sent.at(-1), request);
  h.reply(request, {isError: true, content: [{type: 'text', text: 'Disconnected'}]}); await pending;
  assert.match(h.nodes.message.textContent, /Disconnected/);
  assert.equal(h.nodes.scheduler.textContent, 'Running');
  assert.ok([...h.timers.values()].some(v => v.delay === 30000));
  h.document.hidden = true; h.documentEvents.visibilitychange(); assert.equal(h.timers.size, 0);
  h.document.hidden = false;
  const manual = h.nodes.refresh.events.click();
  h.reply(h.sent.at(-1), {structuredContent: failed}); await manual;
  assert.equal(h.nodes.outcome.textContent, 'Failed'); assert.equal(h.timers.size, 0);
  h.emit({method: 'ui/notifications/tool-result', params: {structuredContent: data()}}, {});
  assert.equal(h.nodes.outcome.textContent, 'Failed');
  h.emit({method: 'ui/notifications/tool-result', params: {structuredContent: {...data(), handle: {...handle, run_id: 'other'}}}});
  assert.equal(h.nodes.outcome.textContent, 'Failed');
  const open = h.nodes.open.events.click();
  assert.equal(h.sent.at(-1).params.name, 'open_task_monitor');
  h.reply(h.sent.at(-1), {structuredContent: {url: 'http://127.0.0.1:1/task-monitor/'}}); await tick();
  assert.equal(h.sent.at(-1).method, 'ui/open-link');
  h.reply(h.sent.at(-1), {}); await open;
  h.emit({id: 100, method: 'ui/resource-teardown'});
  assert.equal(h.nodes.refresh.disabled, true); assert.equal(h.timers.size, 0);
  const readonly = environment(true, data(), {}); await readonly.init();
  assert.equal(readonly.nodes.refresh.disabled, true); assert.equal(readonly.timers.size, 0);
  console.log('Task monitor: HTTP and MCP, active polling, terminal stop, hidden/pause, errors, no overlap, safe text, identity and teardown passed');
}
main().catch(error => {console.error(error); process.exitCode = 1;});
