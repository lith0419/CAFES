(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const embedded = window.parent !== window;
  const interval = 15000;
  let handle = null, target = 'local', snapshot = null, timer = null;
  let busy = false, ready = !embedded, disposed = false, failures = 0, sequence = 0;
  let capabilities = {}, controller = null;
  const pending = new Map();
  const labels = {queued: 'Queued', running: 'Running', completed: 'Completed', failed: 'Failed',
    succeeded: 'Succeeded', unconverged: 'Unconverged', cancelled: 'Cancelled', blocked: 'Blocked'};
  const label = value => labels[value] || value || 'Not reported';
  const number = value => Number.isFinite(value) ? value.toExponential(3) : '—';
  const stamp = value => value ? new Date(value).toLocaleString() : 'Not reported';
  const sameHandle = other => other && ['job_id', 'executor_id', 'run_id', 'work_dir', 'submitted_at', 'backend_job_id', 'runtime_release_id'].every(key => (handle?.[key] ?? null) === (other[key] ?? null));
  function message(text, error = false) { $('message').textContent = text; $('message').className = error ? 'error' : ''; }
  function stopTimer() { clearTimeout(timer); timer = null; }
  function canRefresh() { return ready && handle && !busy && !disposed && (!embedded || capabilities.serverTools); }
  function controls() {
    $('refresh').disabled = !canRefresh();
    $('open').hidden = !embedded || !snapshot?.can_open_workbench;
    $('open').disabled = !canRefresh();
    $('auto').disabled = disposed || (embedded && ready && !capabilities.serverTools);
    $('refresh-state').textContent = snapshot?.status.terminal ? 'Finished · automatic refresh stopped'
      : !ready ? 'Connecting…' : embedded && !capabilities.serverTools ? 'Host does not support refresh'
      : !$('auto').checked ? 'Paused' : document.hidden ? 'Paused while hidden'
      : failures ? 'Connection interrupted · retrying' : 'Every 15 seconds while active';
  }
  function schedule() {
    stopTimer(); controls();
    if (!canRefresh() || !$('auto').checked || document.hidden || snapshot?.status.terminal) return;
    timer = setTimeout(refresh, Math.min(interval * 2 ** failures, 60000));
  }
  function render(data) {
    if (!data?.handle || !data.status || (handle && !sameHandle(data.handle))) throw new Error('The returned task does not match this monitor.');
    handle = data.handle; snapshot = data;
    const status = data.status, report = data.summary;
    const science = report?.execution_status || status.task_status;
    $('title').textContent = 'Calculation progress';
    $('identity').textContent = `${handle.run_id} · ${handle.executor_id}${handle.backend_job_id ? ' · Job ' + handle.backend_job_id : ''}`;
    const outcome = status.terminal ? science || status.state : status.state;
    $('outcome').textContent = label(outcome);
    $('outcome').className = 'badge ' + (['failed', 'unconverged', 'blocked'].includes(outcome) ? 'failed' : outcome === 'succeeded' ? 'succeeded' : '');
    $('scheduler').textContent = label(status.state);
    $('science').textContent = science ? label(science) : 'Not available';
    $('report').textContent = status.report_available ? 'Available' : 'Not available';
    const seconds = data.elapsed_seconds;
    $('elapsed').textContent = Number.isFinite(seconds) ? `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s` : '—';
    $('result-section').hidden = !report && !status.message;
    $('result-energy').textContent = Number.isFinite(report?.energy) ? `Reported task energy: ${report.energy}${report.energy_unit ? ' ' + report.energy_unit : ''}` : '';
    $('summary').textContent = report?.analysis_summary || status.message || '';
    $('errors').textContent = (report?.errors || []).map(error => error.message)
      .filter(text => !$('summary').textContent.includes(text)).join('\n');
    const progress = data.solver_progress, completed = progress?.last_completed_sweep;
    $('progress-title').textContent = progress ? 'Last observed DMRG progress' : 'Solver progress';
    $('sweeps').hidden = !progress;
    $('progress-note').textContent = progress?.note || 'No solver metrics have been reported. Task status remains available.';
    $('observed').textContent = progress ? `Latest sweep header: index ${progress.observed_sweep.sweep_index}, M=${progress.observed_sweep.bond_dimension}.` : '';
    $('sweep').textContent = completed ? `Index ${completed.sweep_index}` : '—';
    $('bond').textContent = completed?.bond_dimension ?? '—';
    $('delta').textContent = number(completed?.energy_change);
    $('weight').textContent = number(completed?.discarded_weight);
    $('sweep-energy').textContent = completed ? `Solver sweep energy: ${completed.energy}. This may be an impurity or intermediate calculation.` : '';
    $('progress-source').textContent = progress ? `Source: ${progress.source}` : '';
    $('log-age').textContent = progress ? `Log updated ${stamp(progress.logged_at)}` : '';
    const opened = new Set(Array.from($('logs').children).filter(node => node.open).map(node => node.dataset.path));
    $('logs').replaceChildren();
    for (const log of data.logs || []) {
      const details = document.createElement('details'); details.dataset.path = log.path; details.open = opened.has(log.path);
      const title = document.createElement('summary'); title.textContent = `${log.path} · ${stamp(log.updated_at)}`;
      const pre = document.createElement('pre'); pre.textContent = log.tail;
      details.append(title, pre); $('logs').append(details);
    }
    if (!data.logs?.length) $('logs').textContent = 'No text log is available yet.';
    $('notices').textContent = (data.notices || []).join('\n');
    $('checked').textContent = `Last checked ${stamp(data.checked_at)}`;
    message(''); controls();
  }
  const send = value => window.parent.postMessage({jsonrpc: '2.0', ...value}, '*');
  function request(method, params) {
    return new Promise((resolve, reject) => {
      const id = ++sequence;
      const timeout = setTimeout(() => { pending.delete(id); reject(new Error('The host did not respond. Last known data is retained.')); }, 30000);
      pending.set(id, {resolve, reject, timeout}); send({id, method, params});
    });
  }
  function unpack(result) {
    if (result?.isError) throw new Error((result.content || []).filter(item => item.type === 'text').map(item => item.text).join('\n') || 'Task inspection failed.');
    return result?.structuredContent;
  }
  async function refresh() {
    if (!canRefresh()) return;
    stopTimer(); busy = true; controls();
    try {
      let data;
      if (embedded) data = unpack(await request('tools/call', {name: 'refresh_task_view', arguments: {handle}}));
      else {
        controller = new AbortController();
        const timeout = setTimeout(() => controller?.abort(), 30000);
        try {
          const response = await fetch('/api/task-view', {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({handle, execution_target: target}), signal: controller.signal});
          data = await response.json();
          if (!response.ok) throw new Error(data.error || 'Task inspection failed.');
        } finally { clearTimeout(timeout); controller = null; }
      }
      if (!disposed) { render(data); failures = 0; }
    } catch (error) {
      failures += 1;
      if (!disposed) message(`${error.message} Last known data is retained.`, true);
    } finally { busy = false; schedule(); }
  }
  $('refresh').addEventListener('click', refresh);
  $('auto').addEventListener('change', schedule);
  document.addEventListener('visibilitychange', schedule);
  function teardown() {
    disposed = true; ready = false; stopTimer(); controller?.abort();
    for (const entry of pending.values()) { clearTimeout(entry.timeout); entry.reject(new Error('Monitor closed.')); }
    pending.clear(); controls();
  }
  window.addEventListener('pagehide', teardown);
  window.addEventListener('pageshow', event => {
    if (!embedded && event.persisted) { disposed = false; ready = true; refresh(); }
  });
  $('open').addEventListener('click', async () => {
    if (!canRefresh()) return;
    stopTimer(); busy = true; controls();
    try {
      const data = unpack(await request('tools/call', {name: 'open_task_monitor', arguments: {handle}}));
      const url = new URL(data?.url);
      if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Invalid monitor URL.');
      $('link').href = url.href; $('link').textContent = 'Open task monitor'; $('link').hidden = false;
      if (capabilities.openLinks) await request('ui/open-link', {url: url.href});
    } catch (error) { message(error.message, true); }
    finally { busy = false; schedule(); }
  });
  if (embedded) {
    function hostContext(context = {}) {
      if (['light', 'dark'].includes(context.theme)) document.documentElement.style.colorScheme = context.theme;
    }
    const observer = new ResizeObserver(() => {
      if (ready && !disposed) send({method: 'ui/notifications/size-changed', params: {width: document.documentElement.clientWidth, height: document.body.scrollHeight}});
    });
    observer.observe(document.body);
    window.addEventListener('message', event => {
      const data = event.data;
      if (disposed || event.source !== window.parent || data?.jsonrpc !== '2.0') return;
      if (pending.has(data.id) && !data.method) {
        const entry = pending.get(data.id); pending.delete(data.id); clearTimeout(entry.timeout);
        if (data.error) entry.reject(new Error(data.error.message)); else entry.resolve(data.result);
      } else if (data.method === 'ping') send({id: data.id, result: {}});
      else if (data.method === 'ui/resource-teardown') { observer.disconnect(); teardown(); send({id: data.id, result: {}}); }
      else if (data.method === 'ui/notifications/host-context-changed') hostContext(data.params);
      else if (data.method === 'ui/notifications/tool-input' && !handle) handle = data.params?.arguments?.handle || null;
      else if (data.method === 'ui/notifications/tool-result') {
        try { render(unpack(data.params)); failures = 0; schedule(); } catch (error) { message(error.message, true); }
      } else if (data.method === 'ui/notifications/tool-cancelled') { stopTimer(); $('auto').checked = false; message('Loading cancelled. Refresh to reconnect.'); controls(); }
    });
    request('ui/initialize', {appInfo: {name: 'pyscf-task-monitor', version: '1.0.0'}, appCapabilities: {availableDisplayModes: ['inline']}, protocolVersion: '2026-01-26'})
      .then(result => {
        if (disposed) return;
        if (result.protocolVersion !== '2026-01-26') throw new Error('This host does not support the monitor protocol. Open the task in the workbench.');
        capabilities = result.hostCapabilities || {}; hostContext(result.hostContext); ready = true;
        send({method: 'ui/notifications/initialized', params: {}}); schedule();
      }).catch(error => message(error.message, true));
  } else {
    try {
      const params = new URLSearchParams(window.location.hash.slice(1));
      handle = JSON.parse(params.get('handle') || 'null'); target = params.get('execution_target') || 'local';
      if (!handle?.run_id) throw new Error('No task selected. Ask Codex to open an existing calculation.');
      refresh();
    } catch (error) { message(error.message, true); }
  }
})();
