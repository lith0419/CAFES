from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+(?:[+.'-][a-zA-Z0-9_]+)*")
_SCIENTIFIC_SYMBOLS = {'u', 'v', 't'}
_SEARCH_STOPWORDS = {
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'by',
    'for',
    'from',
    'in',
    'is',
    'of',
    'on',
    'or',
    'the',
    'to',
    'with',
}


@dataclass
class WikiPage:
    title: str
    slug: str
    summary: str
    body: str
    aliases: List[str]
    links: List[str]
    path: str

    @property
    def search_text(self) -> str:
        return ' '.join([self.title, self.summary, ' '.join(self.aliases), self.body])


@dataclass
class WikiHit:
    page: WikiPage
    score: float
    reasons: List[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_wiki_root() -> Path:
    return _repo_root() / 'agent_knowledge'


def _tokenize(text: Any) -> List[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(str(text or ''))
        if (len(token) > 1 or token.lower() in _SCIENTIFIC_SYMBOLS)
        and token.lower() not in _SEARCH_STOPWORDS
    ]


def _scientific_values(value: Any, *, parameter_names: bool = False) -> Iterable[str]:
    """Read scientific values without turning JSON containers into search terms."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'schema', 'label', 'notes', 'description'} or str(key).endswith(('_file', '_path', '_dir')):
                continue
            if parameter_names and key not in {'name', 'options', 'enabled'}:
                yield str(key)
            yield from _scientific_values(item, parameter_names=parameter_names)
    elif isinstance(value, list):
        for item in value:
            yield from _scientific_values(item, parameter_names=parameter_names)
    elif isinstance(value, str) and not value.startswith('$'):
        yield value


def build_study_wiki_query(goal: str, seed_spec: Optional[Dict[str, Any]] = None) -> str:
    """Project a study into scientific search context, separate from the full LLM seed."""
    seed = seed_spec or {}
    terms = [goal, str(seed.get('objective') or '')]
    terms.append(str(seed.get('system_type') or '').replace('_', ' '))
    terms.extend(_scientific_values(seed.get('observables')))
    model = seed.get('base_model_spec') or {}
    if isinstance(model, dict):
        for field in ('model', 'preset', 'boundary'):
            terms.extend(_scientific_values(model.get(field)))

    requests = [seed.get('base_task') or {}]
    sweep = seed.get('sweep') or {}
    if isinstance(sweep, dict):
        terms.extend(str(parameter) for parameter in sweep)
        requests.append(sweep)
    design = seed.get('case_design') or {}
    if isinstance(design, dict):
        variables = design.get('variables')
        if isinstance(variables, dict):
            terms.extend(str(parameter) for parameter in variables)
        entries = [design.get('template') or {}]
        for field in ('cases', 'overrides'):
            if isinstance(design.get(field), list):
                entries.extend(design[field])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            requests.append(entry.get('request_updates') or {})
            variables = entry.get('variables')
            if isinstance(variables, dict):
                terms.extend(str(parameter) for parameter in variables)
            operations = entry.get('operations')
            for operation in operations if isinstance(operations, list) else []:
                if isinstance(operation, dict):
                    for field in ('op', 'parameter'):
                        terms.extend(_scientific_values(operation.get(field)))

    for request in requests:
        if not isinstance(request, dict):
            continue
        for field in ('method', 'basis', 'xc', 'reference'):
            terms.extend(_scientific_values(request.get(field)))
        for field in ('solver', 'active_space', 'density_fitting', 'correlation_diagnostics'):
            value = request.get(field)
            if isinstance(value, dict) and value.get('enabled') is False:
                continue
            if value:
                if field != 'solver':
                    terms.append(field)
                terms.extend(_scientific_values(value, parameter_names=True))
    return '\n'.join(dict.fromkeys(term.strip() for term in terms if term.strip()))


def _clip_passage(text: str, *, max_chars: int, weights: Dict[str, float]) -> str:
    """For an oversized paragraph, keep the best matching window instead of its prefix."""
    if len(text) <= max_chars:
        return text
    marker = '[...]'
    width = max_chars - 2 * len(marker)
    if width <= 0:
        return text[:max(0, max_chars)]
    starts = {0}
    for match in _TOKEN_RE.finditer(text):
        if match.group().lower() in weights:
            starts.add(max(0, min(match.start() - width // 3, len(text) - width)))
    start = max(sorted(starts), key=lambda offset: sum(
        weights.get(token, 0.0) for token in set(_tokenize(text[offset:offset + width]))
    ))
    end = start + width
    return (marker if start else '') + text[start:end] + (marker if end < len(text) else '')


def _compact_body(body: str, *, max_chars: int, query_tokens: Sequence[str] = ()) -> str:
    normalized = re.sub(r'\n{3,}', '\n\n', str(body or '').strip())
    if max_chars <= 0:
        return ''
    if len(normalized) <= max_chars:
        return normalized

    # Keep section headings with selected paragraphs. Rank against the query,
    # then restore document order so the excerpt remains readable.
    passages = []
    headings: List[tuple] = []
    for block in re.split(r'\n\s*\n|(?=^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s))', normalized, flags=re.MULTILINE):
        block = block.strip()
        if not block:
            continue
        for heading in re.finditer(r'^(#{1,6})\s+(.+)$', block, re.MULTILINE):
            level = len(heading.group(1))
            headings = [(depth, text) for depth, text in headings if depth < level]
            headings.append((level, heading.group()))
        context = '\n'.join(text for _depth, text in headings if text not in block)
        passages.append((context, block))
    token_sets = [set(_tokenize(block)) for _heading, block in passages]
    weights = {
        token: math.log((len(passages) + 1) / (1 + sum(token in terms for terms in token_sets))) + 1
        for token in query_tokens
    }
    scores = [sum(weights.get(token, 0.0) for token in terms) for terms in token_sets]
    if not any(scores):
        return _clip_passage(normalized, max_chars=max_chars, weights={})

    selected = {}
    remaining = max_chars
    separator = '\n\n[...]\n\n'
    for index in sorted(range(len(passages)), key=lambda i: (-scores[i], i)):
        if scores[index] <= 0:
            break
        heading, block = passages[index]
        prefix = heading + '\n\n' if heading else ''
        budget = remaining - (len(separator) if selected else 0)
        if budget <= len(prefix) + 2 * len('[...]'):
            continue
        excerpt = prefix + _clip_passage(block, max_chars=budget - len(prefix), weights=weights)
        selected[index] = excerpt
        remaining = budget - len(excerpt)
    return separator.join(selected[index] for index in sorted(selected)) or _clip_passage(
        normalized, max_chars=max_chars, weights=weights,
    )


def _strip_frontmatter(markdown: str) -> str:
    if not markdown.startswith('---\n'):
        return markdown.strip()
    end = markdown.find('\n---\n', 4)
    if end < 0:
        return markdown.strip()
    return markdown[end + 5 :].strip()


def _frontmatter_value(markdown: str, key: str) -> Optional[str]:
    match = re.search(r'^{}:\s+"?([^"\n]+)"?\s*$'.format(re.escape(key)), markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _frontmatter_list(markdown: str, key: str) -> List[str]:
    lines = markdown.splitlines()
    output: List[str] = []
    in_key = False
    for line in lines:
        if line.startswith('{0}:'.format(key)):
            in_key = True
            continue
        if in_key and re.match(r'^[A-Za-z0-9_-]+:', line):
            break
        if in_key:
            match = re.match(r'\s*-\s+"?([^"\n]+)"?\s*$', line)
            if match:
                output.append(match.group(1).strip())
    return output


def _pages_from_export_payload(payload: Dict[str, Any]) -> List[WikiPage]:
    pages = []
    for item in payload.get('pages') or []:
        if not isinstance(item, dict):
            continue
        pages.append(WikiPage(
            title=str(item.get('title') or item.get('slug') or 'Untitled'),
            slug=str(item.get('slug') or ''),
            summary=str(item.get('summary') or ''),
            body=str(item.get('body') or ''),
            aliases=[str(value) for value in item.get('aliases') or []],
            links=[str(value) for value in item.get('links') or []],
            path=str(item.get('path') or ''),
        ))
    return [page for page in pages if page.slug]


def _load_pages_from_packaged_export() -> List[WikiPage]:
    try:
        content = resources.files('computational_study_agent.knowledge').joinpath(
            'curated_wiki.json'
        ).read_text(encoding='utf-8')
    except (FileNotFoundError, ModuleNotFoundError):
        return []
    return _pages_from_export_payload(json.loads(content))


def _load_pages_from_markdown(wiki_root: Path) -> List[WikiPage]:
    concept_dir = wiki_root / 'wiki' / 'concepts'
    if not concept_dir.exists():
        return []
    pages = []
    for page_path in sorted(concept_dir.glob('*.md')):
        markdown = page_path.read_text(encoding='utf-8')
        body = _strip_frontmatter(markdown)
        title = _frontmatter_value(markdown, 'title') or page_path.stem.replace('-', ' ').title()
        summary = _frontmatter_value(markdown, 'summary') or ''
        links = [match.group(1) for match in re.finditer(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', body)]
        pages.append(WikiPage(
            title=title,
            slug=page_path.stem,
            summary=summary,
            body=body,
            aliases=_frontmatter_list(markdown, 'aliases'),
            links=links,
            path=str(page_path.relative_to(wiki_root)),
        ))
    return pages


def load_wiki_pages(wiki_root: Optional[Path] = None) -> List[WikiPage]:
    if wiki_root is None:
        return _load_pages_from_packaged_export()
    return _load_pages_from_markdown(Path(wiki_root))


def _document_frequencies(pages: Sequence[WikiPage]) -> Dict[str, int]:
    frequencies: Dict[str, int] = {}
    for page in pages:
        for token in set(_tokenize(page.search_text)):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def _score_page(page: WikiPage, query_tokens: Sequence[str], frequencies: Dict[str, int], page_count: int) -> WikiHit:
    page_tokens = _tokenize(page.search_text)
    if not page_tokens:
        return WikiHit(page=page, score=0.0, reasons=[])
    counts: Dict[str, int] = {}
    for token in page_tokens:
        counts[token] = counts.get(token, 0) + 1
    title_tokens = set(_tokenize(page.title))
    alias_tokens = set(_tokenize(' '.join(page.aliases)))
    summary_tokens = set(_tokenize(page.summary))
    score = 0.0
    reasons = []
    for token in query_tokens:
        frequency = counts.get(token, 0)
        if not frequency:
            continue
        idf = math.log((page_count + 1) / (1 + frequencies.get(token, 0))) + 1.0
        weight = 1.0
        if token in title_tokens:
            weight += 2.5
        if token in alias_tokens:
            weight += 1.5
        if token in summary_tokens:
            weight += 1.0
        score += weight * idf * (1.0 + math.log(frequency))
        reasons.append(token)
    return WikiHit(page=page, score=score, reasons=sorted(set(reasons)))


def retrieve_wiki_pages(
    prompt: str,
    *,
    top_k: int = 5,
    wiki_root: Optional[Path] = None,
    include_neighbors: bool = True,
) -> List[WikiHit]:
    pages = load_wiki_pages(wiki_root)
    query_tokens = sorted(set(_tokenize(prompt)))
    if not pages or not query_tokens:
        return []
    frequencies = _document_frequencies(pages)
    hits = [
        _score_page(page, query_tokens, frequencies, len(pages))
        for page in pages
    ]
    hits = [hit for hit in hits if hit.score > 0]
    hits.sort(key=lambda hit: (-hit.score, hit.page.title))
    primary = hits[:max(1, top_k)]
    if not include_neighbors:
        return primary
    by_slug = {page.slug: page for page in pages}
    by_title = {page.title.lower(): page for page in pages}
    selected = {hit.page.slug for hit in primary}
    expanded = list(primary)
    for hit in primary:
        for link in hit.page.links:
            linked_page = by_slug.get(link) or by_title.get(link.lower())
            if linked_page and linked_page.slug not in selected:
                selected.add(linked_page.slug)
                expanded.append(WikiHit(page=linked_page, score=hit.score * 0.35, reasons=['linked']))
                if len(expanded) >= top_k + 3:
                    return expanded
    return expanded


def build_wiki_evidence_pack(
    prompt: str,
    *,
    top_k: int = 5,
    wiki_root: Optional[Path] = None,
    max_chars_per_page: int = 1800,
) -> Dict[str, Any]:
    hits = retrieve_wiki_pages(prompt, top_k=top_k, wiki_root=wiki_root)
    return {
        'query': prompt,
        'pages': [
            {
                'title': hit.page.title,
                'slug': hit.page.slug,
                'summary': hit.page.summary,
                'score': round(hit.score, 4),
                'reasons': hit.reasons,
                'path': hit.page.path,
                'content': _compact_body(hit.page.body, max_chars=max_chars_per_page,
                                         query_tokens=_tokenize(prompt)),
            }
            for hit in hits
        ],
    }


def format_wiki_evidence_for_prompt(pack: Dict[str, Any]) -> str:
    pages = pack.get('pages') or []
    if not pages:
        return 'No local wiki rules were retrieved.'
    blocks = [
        'Local PySCF Agent Wiki Rules:',
        'Use these excerpts as planning and interpretation guidance. '
        'Runtime Registry capabilities, input schemas, and backend validation remain authoritative. '
        'Distinguish implemented behavior from heuristics, roadmap proposals, and archived evidence.',
    ]
    for page in pages:
        blocks.append('\n[Wiki Page: {0}]'.format(page.get('title')))
        if page.get('summary'):
            blocks.append('Summary: {0}'.format(page.get('summary')))
        blocks.append(str(page.get('content') or '').strip())
    return '\n'.join(blocks).strip()
