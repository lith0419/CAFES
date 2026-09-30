const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const CANDIDATES_DIR = path.join(ROOT, '.llmwiki', 'candidates');
const EXPORT_DIR = path.join(ROOT, '.llmwiki', 'review-export');

function readCandidate(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8');
  const candidate = JSON.parse(raw);
  return {
    id: candidate.id || path.basename(filePath, '.json'),
    title: candidate.title || candidate.slug || path.basename(filePath, '.json'),
    slug: candidate.slug || '',
    summary: candidate.summary || '',
    sources: Array.isArray(candidate.sources) ? candidate.sources : [],
    body: candidate.body || '',
    generatedAt: candidate.generatedAt || '',
  };
}

function stripFrontmatter(markdown) {
  if (!markdown.startsWith('---\n')) return markdown.trim();
  const end = markdown.indexOf('\n---\n', 4);
  if (end === -1) return markdown.trim();
  return markdown.slice(end + 5).trim();
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function markdownToHtml(markdown) {
  const lines = markdown.split(/\r?\n/);
  const output = [];
  let inList = false;
  let inCode = false;

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (line.startsWith('```')) {
      if (inList) {
        output.push('</ul>');
        inList = false;
      }
      output.push(inCode ? '</code></pre>' : '<pre><code>');
      inCode = !inCode;
      continue;
    }
    if (inCode) {
      output.push(escapeHtml(rawLine));
      continue;
    }
    if (!line.trim()) {
      if (inList) {
        output.push('</ul>');
        inList = false;
      }
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      if (inList) {
        output.push('</ul>');
        inList = false;
      }
      const level = Math.min(heading[1].length + 1, 6);
      output.push(`<h${level}>${escapeHtml(heading[2])}</h${level}>`);
      continue;
    }
    const item = line.match(/^-\s+(.*)$/);
    if (item) {
      if (!inList) {
        output.push('<ul>');
        inList = true;
      }
      output.push(`<li>${escapeHtml(item[1])}</li>`);
      continue;
    }
    if (inList) {
      output.push('</ul>');
      inList = false;
    }
    output.push(`<p>${escapeHtml(line)}</p>`);
  }

  if (inList) output.push('</ul>');
  if (inCode) output.push('</code></pre>');
  return output.join('\n');
}

function groupName(candidate) {
  const source = candidate.sources[0] || candidate.slug || candidate.id;
  if (source.startsWith('architecture-')) return 'Architecture';
  if (source.startsWith('model-hamiltonian-')) return 'Model Hamiltonian';
  if (source.startsWith('planner-agent-')) return 'Planner Agent';
  if (source.startsWith('pyscf-methods-')) return 'PySCF Methods';
  if (source.startsWith('workflow-rules-')) return 'Workflow Rules';
  if (source.startsWith('ui-design-')) return 'UI Design';
  if (source.startsWith('error-book-')) return 'Error Book';
  if (source === '_index.md') return 'Overview';
  return 'Other';
}

function main() {
  if (!fs.existsSync(CANDIDATES_DIR)) {
    throw new Error(`No candidate directory found: ${CANDIDATES_DIR}`);
  }

  fs.mkdirSync(EXPORT_DIR, { recursive: true });
  const candidates = fs
    .readdirSync(CANDIDATES_DIR)
    .filter((name) => name.endsWith('.json'))
    .sort()
    .map((name) => readCandidate(path.join(CANDIDATES_DIR, name)))
    .sort((a, b) => groupName(a).localeCompare(groupName(b)) || a.title.localeCompare(b.title));

  const groups = new Map();
  for (const candidate of candidates) {
    const group = groupName(candidate);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(candidate);
  }

  const markdown = [];
  markdown.push('# llmwiki Review Candidates');
  markdown.push('');
  markdown.push(`Generated candidates: ${candidates.length}`);
  markdown.push('');
  markdown.push('## Table of Contents');
  markdown.push('');
  for (const [group, items] of groups.entries()) {
    markdown.push(`- ${group} (${items.length})`);
  }
  markdown.push('');

  for (const [group, items] of groups.entries()) {
    markdown.push(`## ${group}`);
    markdown.push('');
    for (const candidate of items) {
      markdown.push(`### ${candidate.title}`);
      markdown.push('');
      markdown.push(`- Candidate id: \`${candidate.id}\``);
      if (candidate.slug) markdown.push(`- Slug: \`${candidate.slug}\``);
      if (candidate.sources.length) markdown.push(`- Sources: ${candidate.sources.map((s) => `\`${s}\``).join(', ')}`);
      if (candidate.summary) markdown.push(`- Summary: ${candidate.summary}`);
      markdown.push('');
      markdown.push(stripFrontmatter(candidate.body));
      markdown.push('');
      markdown.push('---');
      markdown.push('');
    }
  }

  const markdownPath = path.join(EXPORT_DIR, 'review-candidates.md');
  fs.writeFileSync(markdownPath, markdown.join('\n'), 'utf8');

  const nav = [];
  const sections = [];
  for (const [group, items] of groups.entries()) {
    const groupId = group.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    nav.push(`<a href="#${groupId}">${escapeHtml(group)} <span>${items.length}</span></a>`);
    sections.push(`<section id="${groupId}"><h2>${escapeHtml(group)}</h2>`);
    for (const candidate of items) {
      sections.push(`<article class="candidate">
        <div class="candidate-header">
          <h3>${escapeHtml(candidate.title)}</h3>
          <code>${escapeHtml(candidate.id)}</code>
        </div>
        <p class="summary">${escapeHtml(candidate.summary)}</p>
        <p class="meta">Sources: ${candidate.sources.map(escapeHtml).join(', ') || 'none'}</p>
        <div class="body">${markdownToHtml(stripFrontmatter(candidate.body))}</div>
      </article>`);
    }
    sections.push('</section>');
  }

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>llmwiki Review Candidates</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #64748b;
      --line: #d9e2ef;
      --accent: #475569;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 18px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(8px);
    }
    header h1 { margin: 0; font-size: 24px; }
    header p { margin: 4px 0 0; color: var(--muted); }
    .layout {
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      gap: 24px;
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    nav {
      position: sticky;
      top: 92px;
      align-self: start;
      display: grid;
      gap: 8px;
    }
    nav a {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 9px 12px;
      color: var(--ink);
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    nav span { color: var(--muted); }
    main { min-width: 0; }
    section { margin-bottom: 28px; }
    section h2 { font-size: 22px; margin: 0 0 14px; }
    .candidate {
      margin: 0 0 16px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    .candidate-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 10px;
      margin-bottom: 12px;
    }
    h3 { margin: 0; font-size: 20px; }
    code {
      padding: 2px 6px;
      border-radius: 6px;
      background: #eef2f7;
      color: #334155;
      white-space: nowrap;
    }
    .summary { margin: 0 0 6px; color: var(--accent); }
    .meta { margin: 0 0 14px; color: var(--muted); }
    .body h2 { font-size: 18px; margin-top: 22px; }
    .body h3 { font-size: 16px; margin-top: 18px; }
    .body p { margin: 8px 0; }
    .body ul { margin: 8px 0 12px 22px; padding: 0; }
    pre {
      overflow: auto;
      padding: 12px;
      border-radius: 8px;
      background: #0f172a;
      color: #e2e8f0;
    }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      nav { position: static; }
      .candidate-header { display: block; }
      code { display: inline-block; margin-top: 8px; white-space: normal; }
    }
  </style>
</head>
<body>
  <header>
    <h1>llmwiki Review Candidates</h1>
    <p>${candidates.length} candidates generated from ${groups.size} groups. Use each candidate id with <code>npx llmwiki review approve &lt;id&gt;</code>.</p>
  </header>
  <div class="layout">
    <nav>${nav.join('\n')}</nav>
    <main>${sections.join('\n')}</main>
  </div>
</body>
</html>`;

  const htmlPath = path.join(EXPORT_DIR, 'review-candidates.html');
  fs.writeFileSync(htmlPath, html, 'utf8');

  console.log(`Wrote ${path.relative(ROOT, markdownPath)}`);
  console.log(`Wrote ${path.relative(ROOT, htmlPath)}`);
}

main();
