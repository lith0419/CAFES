const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const SOURCES_DIR = path.join(ROOT, 'sources');
const CANDIDATES_DIR = path.join(ROOT, '.llmwiki', 'candidates');
const STRUCTURE_PATH = path.join(ROOT, 'curated_structure.json');
const WIKI_DIR = path.join(ROOT, 'wiki');
const CONCEPTS_DIR = path.join(WIKI_DIR, 'concepts');
const EXPORT_DIR = path.join(ROOT, '.llmwiki', 'review-export');
const PACKAGE_EXPORT_DIR = path.join(ROOT, '..', 'computational_study_agent', 'knowledge');

function slugify(value) {
  return String(value)
    .trim()
    .toLowerCase()
    .replace(/_/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function normalizeTitle(value) {
  return String(value)
    .toLowerCase()
    .replace(/_/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function expandFallbackAliases(candidate, aliases) {
  return [candidate, ...aliases
    .filter((alias) => normalizeTitle(alias) !== normalizeTitle(candidate.title))
    .map((alias, index) => ({
      ...candidate,
      id: `${candidate.id}:alias:${index}`,
      title: alias,
      slug: slugify(alias),
      fallbackSource: true,
    }))];
}

function readCandidates() {
  if (!fs.existsSync(CANDIDATES_DIR)) {
    return [];
  }
  return fs
    .readdirSync(CANDIDATES_DIR)
    .filter((name) => name.endsWith('.json'))
    .map((name) => {
      const candidate = readJson(path.join(CANDIDATES_DIR, name));
      return {
        id: candidate.id || path.basename(name, '.json'),
        title: candidate.title || candidate.slug || path.basename(name, '.json'),
        slug: candidate.slug || slugify(candidate.title || name),
        summary: candidate.summary || '',
        sources: Array.isArray(candidate.sources) ? candidate.sources : [],
        body: candidate.body || '',
        sourceStates: candidate.sourceStates || {},
      };
    });
}

function readCompiledCandidates() {
  if (!fs.existsSync(CONCEPTS_DIR)) {
    return [];
  }
  return fs
    .readdirSync(CONCEPTS_DIR)
    .filter((name) => name.endsWith('.md'))
    .flatMap((name) => {
      const body = fs.readFileSync(path.join(CONCEPTS_DIR, name), 'utf8');
      if (extractFrontmatterValue(body, 'curated') === 'true') {
        return [];
      }
      const title = extractFrontmatterValue(body, 'title') || path.basename(name, '.md');
      const candidate = {
        id: `compiled:${name}`,
        title,
        slug: path.basename(name, '.md'),
        summary: extractFrontmatterValue(body, 'summary'),
        sources: extractFrontmatterList(body, 'sources'),
        body,
        sourceStates: {},
      };
      return expandFallbackAliases(candidate, extractFrontmatterList(body, 'aliases'));
    });
}

function readPublishedCandidates() {
  const publishedPath = path.join(PACKAGE_EXPORT_DIR, 'curated_wiki.json');
  if (!fs.existsSync(publishedPath)) {
    return [];
  }
  const payload = readJson(publishedPath);
  if (!Array.isArray(payload.pages)) {
    return [];
  }
  return payload.pages.flatMap((page) => {
    const body = String(page.body || '');
    const sourcesSection = body.split(/^## Sources\s*$/m)[1] || '';
    const sources = [...sourcesSection.matchAll(/^-\s+`([^`]+)`/gm)].map((match) => match[1]);
    const candidate = {
      id: `published:${page.slug}`,
      title: page.title,
      slug: page.slug,
      summary: page.summary || '',
      sources,
      body,
      sourceStates: {},
      fallbackSource: true,
    };
    return expandFallbackAliases(candidate, Array.isArray(page.aliases) ? page.aliases : []);
  });
}

function readSourceFallbackCandidates() {
  if (!fs.existsSync(SOURCES_DIR)) {
    return [];
  }
  return fs
    .readdirSync(SOURCES_DIR)
    .filter((name) => name.endsWith('.md'))
    .map((name) => {
      const sourcePath = path.join(SOURCES_DIR, name);
      const body = fs.readFileSync(sourcePath, 'utf8');
      const heading = body.match(/^#\s+(.+)$/m);
      const title = heading ? heading[1].trim() : path.basename(name, '.md');
      const summaryLine = body
        .split(/\r?\n/)
        .map((line) => line.trim())
        .find((line) => line && !line.startsWith('#'));
      return {
        id: `source:${name}`,
        title,
        slug: slugify(title),
        summary: summaryLine || title,
        sources: [name],
        body,
        sourceStates: {},
        fallbackSource: true,
        maintainedSource: true,
      };
    });
}

function stripFrontmatter(markdown) {
  if (!markdown.startsWith('---\n')) return markdown.trim();
  const end = markdown.indexOf('\n---\n', 4);
  if (end === -1) return markdown.trim();
  return markdown.slice(end + 5).trim();
}

function removeFirstHeading(markdown) {
  const lines = markdown.split(/\r?\n/);
  if (lines[0] && lines[0].startsWith('# ')) {
    return lines.slice(1).join('\n').trim();
  }
  return markdown.trim();
}

function removeGeneratedSections(markdown) {
  const lines = markdown.split(/\r?\n/);
  const output = [];
  let skip = false;
  for (const line of lines) {
    if (/^Here is the wiki page/i.test(line.trim())) {
      continue;
    }
    if (/^---\s*$/.test(line.trim())) {
      continue;
    }
    if (/^>\s*Curated section:/i.test(line.trim())) {
      continue;
    }
    if (/^##\s+(Sources|Related Concepts|Related Pages)\s*$/i.test(line.trim())) {
      skip = true;
      continue;
    }
    if (skip && /^##\s+/.test(line.trim())) {
      skip = false;
    }
    if (!skip) output.push(line);
  }
  return output.join('\n').trim();
}

function stripClaimCitations(markdown) {
  return markdown
    .replace(/\^\[[^\]]+\]/g, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function extractFrontmatterValue(markdown, key) {
  const frontmatter = markdown.match(/^---\n([\s\S]*?)\n---/);
  if (!frontmatter) return '';
  const match = frontmatter[1].match(new RegExp(`^${key}:\\s+"?([^"\\n]+)"?\\s*$`, 'm'));
  return match ? match[1].trim() : '';
}

function extractFrontmatterList(markdown, key) {
  const frontmatter = markdown.match(/^---\n([\s\S]*?)\n---/);
  if (!frontmatter) return [];
  const lines = frontmatter[1].split(/\r?\n/);
  const output = [];
  let inKey = false;
  for (const line of lines) {
    if (line.startsWith(`${key}:`)) {
      inKey = true;
      continue;
    }
    if (inKey && /^[A-Za-z0-9_-]+:/.test(line)) break;
    if (inKey) {
      const match = line.match(/^\s*-\s+"?([^"\n]+)"?\s*$/);
      if (match) output.push(match[1].trim());
    }
  }
  return output;
}

function extractWikiLinks(markdown) {
  return [...markdown.matchAll(/\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)].map((match) => match[1]);
}

function extractBody(candidate) {
  return stripClaimCitations(removeGeneratedSections(removeFirstHeading(stripFrontmatter(candidate.body))));
}

function yamlScalar(value) {
  return JSON.stringify(String(value));
}

function yamlList(values, indent = '') {
  if (!values.length) return `${indent}[]`;
  return values.map((value) => `${indent}- ${yamlScalar(value)}`).join('\n');
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function buildCandidateIndexes(candidates) {
  const byTitle = new Map();
  const bySlug = new Map();
  // Reviewed source notes outrank historical compiler candidates and exports.
  // pageGuidance remains the explicit editorial override in renderGuidance.
  const priority = (candidate) => candidate?.maintainedSource ? 2 : candidate?.fallbackSource ? 0 : 1;
  const put = (index, key, candidate) => {
    if (!index.has(key) || priority(candidate) > priority(index.get(key)) ||
        (priority(candidate) === priority(index.get(key)) && !candidate.fallbackSource)) {
      index.set(key, candidate);
    }
  };
  for (const candidate of candidates) {
    put(byTitle, normalizeTitle(candidate.title), candidate);
    put(bySlug, candidate.slug, candidate);
    put(bySlug, slugify(candidate.title), candidate);
  }
  return { byTitle, bySlug };
}

function getPageTitle(page) {
  return typeof page === 'string' ? page : page.title;
}

function getPageSource(page) {
  if (typeof page === 'string') return page;
  return page.source || page.title;
}

function getPageKind(page) {
  if (typeof page === 'string') return 'concept';
  return page.kind || 'concept';
}

function findCandidate(title, indexes) {
  return indexes.byTitle.get(normalizeTitle(title)) || indexes.bySlug.get(slugify(title)) || null;
}

function buildTitleRoute(structure) {
  const route = new Map();
  for (const section of structure.sections) {
    for (const page of section.pages) {
      const title = getPageTitle(page);
      const source = getPageSource(page);
      route.set(normalizeTitle(title), title);
      route.set(normalizeTitle(source), title);
    }
  }
  for (const [target, mergedTitles] of Object.entries(structure.mergeInto || {})) {
    for (const title of mergedTitles) {
      route.set(normalizeTitle(title), target);
    }
  }
  return route;
}

function normalizeWikiLinks(markdown, titleRoute) {
  return markdown.replace(/\[\[([^\]]+)\]\]/g, (_match, rawTarget) => {
    const [target, label] = String(rawTarget).split('|');
    const routed = titleRoute.get(normalizeTitle(target));
    if (routed) return `[[${routed}${label ? `|${label}` : ''}]]`;
    return label || target;
  });
}

function renderGuidance(guidance, fallbackBody, titleRoute) {
  if (!guidance) return normalizeWikiLinks(fallbackBody, titleRoute);
  const lines = [];
  if (guidance.summary) {
    lines.push(`> ${guidance.summary}`);
    lines.push('');
  }
  for (const block of guidance.sections || []) {
    lines.push(`## ${block.heading}`);
    lines.push('');
    for (const item of block.items || []) {
      lines.push(`- ${normalizeWikiLinks(item, titleRoute)}`);
    }
    lines.push('');
  }
  return lines.join('\n').trim();
}

function renderPage({ title, section, pageKind, candidate, mergedCandidates, seeAlso, titleRoute, guidance }) {
  const mergedTitles = mergedCandidates.map((item) => item.title);
  const sources = unique([
    ...candidate.sources,
    ...mergedCandidates.flatMap((item) => item.sources),
  ]);
  const aliases = unique([
    candidate.slug,
    candidate.title,
    ...mergedTitles,
    ...(guidance?.aliases || []),
  ]).filter((alias) => normalizeTitle(alias) !== normalizeTitle(title));
  const frontmatter = [
    '---',
    `title: ${yamlScalar(title)}`,
    `summary: ${yamlScalar(guidance?.summary || candidate.summary)}`,
    `sources:`,
    yamlList(sources, '  '),
    'kind: concept',
    `ruleKind: ${yamlScalar(pageKind)}`,
    `section: ${yamlScalar(section)}`,
    'curated: true',
    `curatedAt: ${yamlScalar(new Date().toISOString())}`,
    'tags:',
    `  - ${yamlScalar('curated')}`,
    `  - ${yamlScalar(slugify(section))}`,
    'aliases:',
    yamlList(aliases, '  '),
    'provenanceState: merged',
    'promptVersion: curated-v1',
    '---',
    '',
  ].join('\n');

  const body = [];
  body.push(`# ${title}`);
  body.push('');
  body.push(`> Curated section: ${section}.`);
  body.push('');
  body.push(renderGuidance(guidance, extractBody(candidate), titleRoute));
  body.push('');

  if (seeAlso.length) {
    body.push('## Related Pages');
    body.push('');
    for (const target of seeAlso) {
      body.push(`- [[${target}]]`);
    }
    body.push('');
  }

  body.push('## Sources');
  body.push('');
  for (const source of sources) {
    body.push(`- \`${source}\``);
  }
  body.push('');
  return `${frontmatter}${body.join('\n')}`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function isTableRow(line) {
  return /^\s*\|.*\|\s*$/.test(line);
}

function isTableSeparator(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function splitTableRow(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

function markdownToHtml(markdown) {
  const lines = markdown.split(/\r?\n/);
  const output = [];
  let inList = false;
  let inCode = false;
  for (let i = 0; i < lines.length; i += 1) {
    const rawLine = lines[i];
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
    if (!inCode && isTableRow(line) && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      if (inList) {
        output.push('</ul>');
        inList = false;
      }
      const headers = splitTableRow(line);
      const rows = [];
      i += 2;
      while (i < lines.length && isTableRow(lines[i])) {
        rows.push(splitTableRow(lines[i]));
        i += 1;
      }
      i -= 1;
      output.push('<table>');
      output.push(`<thead><tr>${headers.map((cell) => `<th>${escapeHtml(cell)}</th>`).join('')}</tr></thead>`);
      output.push('<tbody>');
      for (const row of rows) {
        output.push(`<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`);
      }
      output.push('</tbody>');
      output.push('</table>');
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      if (inList) {
        output.push('</ul>');
        inList = false;
      }
      output.push(`<h${heading[1].length}>${escapeHtml(heading[2])}</h${heading[1].length}>`);
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

function renderCuratedExports(pages, missing, unassigned) {
  fs.mkdirSync(EXPORT_DIR, { recursive: true });

  const markdown = [];
  markdown.push('# Curated CAFES Wiki');
  markdown.push('');
  markdown.push(`Curated pages: ${pages.length}`);
  markdown.push('');
  markdown.push('## Directory');
  markdown.push('');
  for (const page of pages) {
    markdown.push(`- ${page.section}: [${page.title}](../../wiki/concepts/${page.slug}.md)`);
  }
  markdown.push('');
  if (missing.length) {
    markdown.push('## Missing Candidates');
    markdown.push('');
    for (const item of missing) markdown.push(`- ${item}`);
    markdown.push('');
  }
  if (unassigned.length) {
    markdown.push('## Unassigned Candidates');
    markdown.push('');
    for (const item of unassigned) markdown.push(`- ${item.title} (${item.id})`);
    markdown.push('');
  }

  fs.writeFileSync(path.join(EXPORT_DIR, 'curated-wiki.md'), markdown.join('\n'), 'utf8');

  const nav = pages
    .map((page) => `<a href="#${page.slug}">${escapeHtml(page.section)} / ${escapeHtml(page.title)}</a>`)
    .join('\n');
  const sections = pages
    .map((page) => {
      const content = fs.readFileSync(path.join(CONCEPTS_DIR, `${page.slug}.md`), 'utf8');
      return `<article id="${page.slug}">${markdownToHtml(stripFrontmatter(content))}</article>`;
    })
    .join('\n');

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Curated CAFES Wiki</title>
  <style>
    :root { --bg: #f6f8fb; --panel: #fff; --ink: #1f2937; --muted: #64748b; --line: #d9e2ef; }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: var(--bg); font: 15px/1.56 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    header { position: sticky; top: 0; z-index: 2; padding: 18px 28px; border-bottom: 1px solid var(--line); background: rgba(255,255,255,.94); backdrop-filter: blur(8px); }
    header h1 { margin: 0; font-size: 24px; }
    header p { margin: 4px 0 0; color: var(--muted); }
    .layout { display: grid; grid-template-columns: 300px minmax(0,1fr); gap: 24px; max-width: 1480px; margin: 0 auto; padding: 24px; }
    nav { position: sticky; top: 94px; align-self: start; display: grid; gap: 7px; max-height: calc(100vh - 120px); overflow: auto; }
    nav a { padding: 8px 10px; color: var(--ink); text-decoration: none; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
    article { margin: 0 0 18px; padding: 22px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
    h1 { margin-top: 0; }
    h2 { margin-top: 30px; border-top: 1px solid var(--line); padding-top: 18px; }
    h3 { margin-top: 24px; }
    blockquote { margin: 12px 0; padding: 10px 14px; border-left: 3px solid #94a3b8; background: #f1f5f9; color: #334155; }
    code { padding: 2px 5px; border-radius: 5px; background: #eef2f7; }
    table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 14px; }
    th, td { border: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: #f1f5f9; }
    pre { overflow: auto; padding: 12px; border-radius: 8px; background: #0f172a; color: #e2e8f0; }
    @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } nav { position: static; max-height: none; } }
  </style>
</head>
<body>
  <header>
    <h1>Curated CAFES Wiki</h1>
    <p>${pages.length} curated pages generated from llmwiki candidates.</p>
  </header>
  <div class="layout">
    <nav>${nav}</nav>
    <main>${sections}</main>
  </div>
</body>
</html>`;
  fs.writeFileSync(path.join(EXPORT_DIR, 'curated-wiki.html'), html, 'utf8');
}

function writeRuntimeExport(pages) {
  const exportedPages = pages.map((page) => {
    const relativePath = `wiki/concepts/${page.slug}.md`;
    const markdown = fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
    const body = stripFrontmatter(markdown);
    return {
      title: extractFrontmatterValue(markdown, 'title') || page.title,
      slug: page.slug,
      summary: extractFrontmatterValue(markdown, 'summary') || '',
      body,
      aliases: extractFrontmatterList(markdown, 'aliases'),
      links: extractWikiLinks(body),
      path: relativePath,
      curated: true,
      section: page.section,
    };
  });
  const payload = {
    kind: 'curated_action_guide',
    generatedAt: new Date().toISOString(),
    pageCount: exportedPages.length,
    pages: exportedPages,
  };
  const json = JSON.stringify(payload, null, 2);
  fs.mkdirSync(PACKAGE_EXPORT_DIR, { recursive: true });
  fs.writeFileSync(path.join(PACKAGE_EXPORT_DIR, 'curated_wiki.json'), `${json}\n`, 'utf8');
}

function main() {
  const structure = readJson(STRUCTURE_PATH);
  const reviewCandidates = readCandidates();
  const compiledCandidates = readCompiledCandidates();
  const publishedCandidates = readPublishedCandidates();
  const sourceCandidates = readSourceFallbackCandidates();
  if (!sourceCandidates.length && !reviewCandidates.length && !compiledCandidates.length && !publishedCandidates.length) {
    throw new Error(
      'Wiki curation requires maintained sources, review candidates, compiled concept pages, or a published runtime Wiki.'
    );
  }
  const candidates = [
    ...sourceCandidates,
    ...publishedCandidates,
    ...compiledCandidates,
    ...reviewCandidates,
  ];
  const indexes = buildCandidateIndexes(candidates);
  const titleRoute = buildTitleRoute(structure);
  const pageTitles = new Set(
    structure.sections.flatMap((section) => section.pages.map((page) => getPageTitle(page)))
  );
  const assigned = new Set();
  const pages = [];
  const missing = [];
  const brokenLinks = [];

  fs.rmSync(CONCEPTS_DIR, { recursive: true, force: true });
  fs.mkdirSync(CONCEPTS_DIR, { recursive: true });

  for (const section of structure.sections) {
    for (const pageSpec of section.pages) {
      const title = getPageTitle(pageSpec);
      const source = getPageSource(pageSpec);
      const pageKind = getPageKind(pageSpec);
      const candidate = findCandidate(source, indexes);
      if (!candidate) {
        missing.push(`${section.name} / ${title} <- ${source}`);
        continue;
      }
      assigned.add(candidate.id);
      const mergedCandidates = [];
      for (const mergedTitle of structure.mergeInto?.[title] || []) {
        const merged = findCandidate(mergedTitle, indexes);
        if (!merged) {
          missing.push(`${section.name} / ${title} <- ${mergedTitle}`);
          continue;
        }
        assigned.add(merged.id);
        mergedCandidates.push(merged);
      }

      const slug = slugify(title);
      const page = { title, section: section.name, slug };
      pages.push(page);
      const seeAlso = structure.seeAlso?.[title] || [];
      for (const target of seeAlso) {
        if (!pageTitles.has(target)) {
          brokenLinks.push(`${title} -> ${target}`);
        }
      }
      const markdown = renderPage({
        title,
        section: section.name,
        pageKind,
        candidate,
        mergedCandidates,
        seeAlso,
        titleRoute,
        guidance: structure.pageGuidance?.[title],
      });
      fs.writeFileSync(path.join(CONCEPTS_DIR, `${slug}.md`), markdown, 'utf8');
    }
  }

  const unassigned = candidates
    .filter((candidate) => !candidate.fallbackSource && !assigned.has(candidate.id))
    .sort((a, b) => a.title.localeCompare(b.title));

  const index = [];
  index.push('# Curated CAFES Wiki');
  index.push('');
  index.push('This index is generated from `curated_structure.json`.');
  index.push('');
  for (const section of structure.sections) {
    index.push(`## ${section.name}`);
    index.push('');
    for (const pageSpec of section.pages) {
      const title = getPageTitle(pageSpec);
      index.push(`- [${title}](concepts/${slugify(title)}.md)`);
    }
    index.push('');
  }
  if (brokenLinks.length) {
    index.push('## Broken Internal Links');
    index.push('');
    for (const item of brokenLinks) index.push(`- ${item}`);
    index.push('');
  }
  if (unassigned.length) {
    index.push('## Unassigned Candidate Pages');
    index.push('');
    index.push('These candidates were not included in the curated structure.');
    index.push('');
    for (const candidate of unassigned) {
      index.push(`- ${candidate.title} (${candidate.id})`);
    }
    index.push('');
  }
  if (missing.length) {
    index.push('## Missing Candidate Lookups');
    index.push('');
    for (const item of missing) index.push(`- ${item}`);
    index.push('');
  }
  fs.mkdirSync(WIKI_DIR, { recursive: true });
  fs.writeFileSync(path.join(WIKI_DIR, 'index.md'), index.join('\n'), 'utf8');
  renderCuratedExports(pages, missing, unassigned);
  writeRuntimeExport(pages);

  console.log(`Wrote ${pages.length} curated pages to wiki/concepts/`);
  console.log(`Wrote wiki/index.md`);
  console.log(`Wrote .llmwiki/review-export/curated-wiki.md`);
  console.log(`Wrote .llmwiki/review-export/curated-wiki.html`);
  console.log(`Wrote computational_study_agent/knowledge/curated_wiki.json`);
  if (missing.length) {
    console.log(`Missing candidate lookups: ${missing.length}`);
    for (const item of missing) console.log(`  - ${item}`);
  }
  if (brokenLinks.length) {
    console.log(`Broken internal links: ${brokenLinks.length}`);
    for (const item of brokenLinks) console.log(`  - ${item}`);
  }
  if (unassigned.length) {
    console.log(`Unassigned candidates: ${unassigned.length}`);
  }
}

main();
