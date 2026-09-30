// Pure UI helpers. All mutable task/scientific state stays with its page owner.
globalThis.AgentUI = (() => {
  'use strict';

  function escapeHtml(value) {
    return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function makeRunId() {
    const timestamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..*$/, '').replace('T', '-');
    return `${timestamp}-${Math.random().toString(16).slice(2, 10).padEnd(8, '0')}`;
  }

  function makeSessionId(prefix) {
    return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  }

  function findSession(sessions, id) {
    return sessions.find(item => item.id === id) || null;
  }

  function sessionOptions(sessions, activeId, label, status) {
    return sessions.map((session, index) => {
      const selected = session.id === activeId ? ' selected' : '';
      const text = `${index + 1}. ${label(session)} · ${status(session.status)}`;
      return `<option value="${escapeHtml(session.id)}"${selected}>${escapeHtml(text)}</option>`;
    }).join('');
  }

  function tableMarkup(headers, rows, {className = '', rowHeaders = false} = {}) {
    const head = headers.length
      ? `<thead><tr>${headers.map(value => `<th>${escapeHtml(value)}</th>`).join('')}</tr></thead>` : '';
    const body = rows.map(row => `<tr>${row.map((value, index) => {
      const tag = rowHeaders && index === 0 ? 'th' : 'td';
      return `<${tag}>${escapeHtml(value)}</${tag}>`;
    }).join('')}</tr>`).join('');
    return `<table${className ? ` class="${escapeHtml(className)}"` : ''}>${head}<tbody>${body}</tbody></table>`;
  }

  return Object.freeze({escapeHtml, makeRunId, makeSessionId, findSession, sessionOptions, tableMarkup});
})();
