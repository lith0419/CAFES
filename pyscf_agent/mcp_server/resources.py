"""Resource projections over existing reports and packaged knowledge."""

from __future__ import annotations

import base64
import json
from typing import Any

from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError

from ..executors import JobHandle
from ..executors.base import ExecutorContractError, JobError
from ..remote.protocol import RemoteProtocolError
from ..schema_contracts import public_schema_manifest


def report_uri(handle: JobHandle) -> str:
    # A reversible transport encoding of JobHandle, not a new stored task ID.
    data = json.dumps(handle.to_dict(), separators=(',', ':'), sort_keys=True).encode('utf-8')
    reference = base64.urlsafe_b64encode(data).decode('ascii').rstrip('=')
    return 'pyscf://jobs/{0}/report'.format(reference)


def register_resources(server: Any, service: Any, checked_handle: Any) -> None:
    @server.resource('pyscf://schemas', mime_type='application/json')
    def schemas() -> str:
        """Stable public TaskSpec, TaskReport and execution contract identifiers."""
        return json.dumps(public_schema_manifest(), ensure_ascii=False)

    @server.resource('pyscf://capabilities', mime_type='application/json')
    def capabilities() -> str:
        """Full runtime Registry, including parameters, limitations and providers."""
        return json.dumps(service.capabilities(), ensure_ascii=False)

    @server.resource('pyscf://jobs/{reference}/report', mime_type='application/json')
    def task_report(reference: str) -> str:
        """Read an existing TaskReport using the URI returned by submit/collect. Never submits."""
        try:
            data = base64.b64decode(reference + '=' * (-len(reference) % 4), altchars=b'-_', validate=True)
            handle = checked_handle(json.loads(data))
        except (TypeError, ValueError) as exc:
            raise ResourceError('Invalid task report reference') from exc
        try:
            report = service.collect_request(handle, '', include_llm_feedback=False)
        except (ValueError, OSError, ExecutorContractError, JobError, RemoteProtocolError) as exc:
            raise ResourceError(str(exc)) from exc
        return json.dumps(report, ensure_ascii=False)

    @server.resource('pyscf://wiki', mime_type='application/json')
    def wiki_index() -> str:
        """Index of packaged scientific and architecture guidance; Registry owns executable support."""
        from computational_study_agent.wiki_retriever import load_wiki_pages
        return json.dumps([{'title': page.title, 'summary': page.summary,
                            'uri': 'pyscf://wiki/' + page.slug}
                           for page in load_wiki_pages()], ensure_ascii=False)

    @server.resource('pyscf://wiki/{slug}', mime_type='text/markdown')
    def wiki_page(slug: str) -> str:
        """Read one page from the installed wiki snapshot."""
        from computational_study_agent.wiki_retriever import load_wiki_pages
        for page in load_wiki_pages():
            if page.slug == slug:
                return page.body
        raise ResourceNotFoundError('Unknown wiki page: ' + slug)
