#!/usr/bin/env python3
"""Offline WebUI/Planner token audit; no provider requests or numerical jobs.

Run with tiktoken available (kept outside project dependencies for this audit).
The fixture reconstructs the documented 32-site / 16-case request; it is not a
captured historical request. Exact provider billing cannot be inferred here.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from computational_study_agent import llm_planner, postprocessing
from computational_study_agent.web_api import (
    handle_saved_study_request, handle_study_llm_draft_request, handle_study_result_analysis_request,
)
from computational_study_agent.wiki_retriever import build_study_wiki_query
from pyscf_agent.registry.platform import (
    result_analysis_output_contracts, result_analysis_contract_prompt,
)
from pyscf_agent.request_builder import llm
from pyscf_agent.request_builder.constants import _STRUCTURED_OUTPUT_SUPPORT

ROOT = Path(__file__).resolve().parents[1]
ENCODINGS = {name: tiktoken.get_encoding(name) for name in ('o200k_base', 'cl100k_base')}


def serialized(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def count(value):
    value = serialized(value)
    return {'characters': len(value), **{name: len(enc.encode(value, disallowed_special=()))
                                      for name, enc in ENCODINGS.items()}}


def request_count(payload):
    messages = payload['messages']
    content_counts = [count(item['content']) for item in messages]
    summed = {key: sum(item[key] for item in content_counts) for key in content_counts[0]}
    schema = count(payload['response_format']) if 'response_format' in payload else dict.fromkeys(summed, 0)
    user = json.loads(messages[1]['content'])
    return {
        'message_count': len(messages), 'message_content': summed,
        'response_format_serialized': schema,
        'content_plus_response_format': {key: summed[key] + schema[key] for key in summed},
        'system': count(messages[0]['content']),
        'user_fields': {key: count(value) for key, value in user.items()},
        'wire_json': count(payload),
    }


def fixture():
    sites = [{'id': y * 8 + x, 'x': x, 'y': y, 'epsilon': 0, 'U': 4}
             for y in range(4) for x in range(8)]
    bonds = []
    for y in range(4):
        for x in range(8):
            for dx, dy in ((1, 0), (0, 1)):
                if x + dx < 8 and y + dy < 4:
                    bonds.append({'id': len(bonds), 'source': y * 8 + x,
                                  'target': (y + dy) * 8 + x + dx,
                                  't': -1, 'V': 0.1, 'effective_t': -1, 'effective_V': 0.1})
    return {
        'name': 'dmet-uv-context-audit', 'objective': 'Scan U and V with DMET',
        'system_type': 'model_hamiltonian',
        'base_model_spec': {'schema': 'pyscf-agent.model-hamiltonian.v1',
                            'model': 'hubbard', 'dimension': 2, 'preset': 'square',
                            'boundary': 'open', 'energy_unit': 'a.u.',
                            'nelec': [16, 16], 'sites': sites, 'bonds': bonds},
        'base_task': {'solver': {'name': 'dmet', 'options': {
            'execution_mode': 'finite_graph', 'impurity_solver': 'fci'}}},
        'sweep': {'U': [1, 2, 3, 4], 'V': [0.1, 0.2, 0.3, 0.4]},
        'observables': ['energy'],
    }


def frontend_request(seed, goal):
    # Execute repository JavaScript functions in a VM, with DOM and network
    # replaced by fixtures. This is a code-level replay, not a browser click.
    script = r'''
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const assets = input.root + '/computational_study_agent/web_assets/';
function extract(file, name) {
  const src = fs.readFileSync(assets + file, 'utf8');
  const match = src.match(new RegExp('(?:async )?function ' + name + '\\([^]*?^}', 'm'));
  if (!match) throw new Error('Missing function ' + name);
  return match[0];
}
const elements = {'goal': {value: input.goal}, 'study-spec': {value: JSON.stringify(input.seed)}};
const noOp = () => {};
let captured, prepared;
const context = {
  document: {getElementById: id => elements[id] || (elements[id] = {value: '', disabled: false})},
  LLM_CONFIGURED: true, currentReport: null, currentAdaptivePreview: null, currentPlan: null,
  currentSavedStudy: null, activeTaskSession: () => null, selectedExecutionTarget: () => 'local',
  executionWorkDir: () => '/tmp/offline-audit', selectedResourceProfile: () => null,
  refreshSavedStudy: async () => {}, snapshotCurrentTaskSession: noOp,
  appendPlannerMessage: noOp, setStatus: noOp, executionConversationIntent: () => null,
  isHamiltonianDatasetPlanner: () => false, currentStudyMode: () => 'static',
  adaptiveOptions: () => null, parseStudySpec: () => JSON.parse(elements['study-spec'].value),
  selectedStudySystem: () => input.seed.system_type,
  applyPlannerModelSolverToSpec: noOp, normalizeSystemType: x => x,
  syncModelInputFileFromStudySpec: () => '', planningDecomposition: () => '',
  postJson: async (url, payload) => {
    if (url === '/api/study-prepare') { prepared = {url, payload}; return {study_id: 'audit-saved', work_dir: payload.work_dir}; }
    captured = {url, payload}; return {study_spec: payload.study_spec, plan: null};
  },
  prettyJson: JSON.stringify, syncStudySystemFromSpec: noOp, renderPlan: noOp,
  renderReviewGate: noOp, wikiEvidenceSummary: () => '', summarizePlannerSpec: () => '',
  validationSummary: () => '', errorDetailMessage: (_, message) => {throw new Error(message);},
  deepCopy: x => JSON.parse(JSON.stringify(x)),
  postprocessContext: () => input.postprocessing_context,
};
vm.createContext(context);
for (const [file, names] of [
  ['planner-spec.js', ['applyPlannerModeContext', 'prepareStudySpecForRequest', 'plannerRequestContext']],
  ['planner-core.js', ['studyStateForReviewAction']],
  ['planner-rendering.js', ['draftWithLlm', 'buildPlan']],
  ['planner-postprocessing.js', ['analysisRows', 'postprocessRows']],
]) for (const name of names) vm.runInContext(extract(file, name), context);
(async () => {
  await context.draftWithLlm();
  await context.buildPlan();
  context.currentReport = {comparison_table: input.quality_rows}; context.currentAnalysisRows = [];
  console.log(JSON.stringify({request: captured, preparation_request: prepared, quality_rows: context.postprocessRows(),
    browser_numeric_parser_removed: typeof context.parseNumericCell === 'undefined'}));
})();
'''
    result = subprocess.run(['node', '-e', script], input=json.dumps({
        'root': str(ROOT), 'seed': seed, 'goal': goal, 'quality_rows': quality_rows(),
        'postprocessing_context': postprocessing.PostprocessContext.from_report({'comparison_table': quality_rows()}).view(),
    }), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def quality_rows():
    return [
        {'case_id': 'failed-legacy', 'status': 'failed', 'x': 1, 'energy': -1},
        {'case_id': 'ineligible', 'status': 'succeeded', 'publication_eligible': False, 'x': 2, 'energy': -2},
        {'case_id': 'success', 'status': 'succeeded', 'publication_eligible': True, 'x': 3, 'energy': -3},
    ]


def planner_replay(payload, mode, directory):
    calls = []
    output = copy.deepcopy(payload['study_spec'])
    invalid = 'Use finite_graph to preserve V.'

    def post(_url, body, _headers, _timeout):
        calls.append(copy.deepcopy(body))
        if mode == 'transport_then_repair' and len(calls) == 1:
            raise ValueError('response_format json_schema is unsupported by this fixture')
        if mode == 'invalid_twice' or (mode in ('repair', 'transport_then_repair') and len(calls) == (2 if mode == 'transport_then_repair' else 1)):
            return {'choices': [{'message': {'content': invalid}}]}
        return {'choices': [{'message': {'parsed': output}}]}

    _STRUCTURED_OUTPUT_SUPPORT.clear()
    request = copy.deepcopy(payload)
    # The actual JS currently omits work_dir. Keep failure records in this audit.
    request['work_dir'] = str(directory)
    with patch('computational_study_agent.llm_planner._default_http_post', post):
        status, _, body = handle_study_llm_draft_request(json.dumps(request).encode())
    response = json.loads(body)
    return {
        'http_status': status.value, 'status': response.get('status'),
        'case_count': len((response.get('plan') or {}).get('cases') or []),
        'validation_issues': response.get('validation_issues'),
        'returned_study_spec': 'study_spec' in response,
        'calls': [request_count(call) for call in calls],
        'synthetic_output_fixture': count(json.dumps(output, ensure_ascii=False)),
        'sum_message_content': {key: sum(request_count(c)['message_content'][key] for c in calls)
                                for key in ('characters', *ENCODINGS)},
        'error': response.get('error'),
    }, calls


def analysis_replay(report):
    calls = []

    def post(_url, body, _headers, _timeout):
        calls.append(copy.deepcopy(body))
        return {'choices': [{'message': {'content': 'Offline token audit; no scientific interpretation generated.'}}]}

    builder = SimpleNamespace(build_result_analysis=lambda *a, **kw: llm.build_result_analysis(*a, http_post=post, **kw))
    status, _, body = handle_study_result_analysis_request(json.dumps({'report': report, 'locale': 'en'}).encode(),
                                                          llm_request_builder=builder)
    response = json.loads(body)
    assert status.value == 200, response
    return {'http_status': status.value, 'report_characters': len(json.dumps(report)),
            'case_count': len(report.get('cases') or []), 'calls': [request_count(call) for call in calls]}, calls


def synthetic_report(size):
    # Synthetic values are counting fixtures, never calculation results.
    rows = [{'case_id': 'audit-%04d' % i, 'status': 'succeeded', 'publication_eligible': True,
             'U': '%s a.u.' % (1 + i // 4), 'V': '%s a.u.' % ((1 + i % 4) / 10),
             'energy': '%s a.u.' % (-20 - i / 10)} for i in range(size)]
    return {'study_id': 'context-audit-synthetic', 'name': 'Synthetic DMET table for token counting',
            'objective': 'Scan U and V with DMET', 'system_type': 'model_hamiltonian',
            'status': 'succeeded', 'comparison_table': rows,
            'cases': [{'case_id': row['case_id'], 'task_report': {'execution_status': 'succeeded'}} for row in rows]}


def postprocessing_probes(directory):
    def outcome(callback):
        try:
            return {'accepted': True, 'result': callback()}
        except (ValueError, TypeError) as exc:
            return {'accepted': False, 'error': str(exc)}

    base = {'study_id': 'postprocess-audit', 'system_type': 'molecular', 'comparison_table': quality_rows()}
    context = postprocessing.PostprocessContext.from_report(base)
    options = {name: outcome(lambda extra=extra: postprocessing._normalized_plot_spec(context, {
        'x': 'x', 'y': 'energy', **extra,
    })) for name, extra in {'invalid_group': {'group': 'misspelled_method'},
                            'false_string': {'sort_by_x': 'false'}, 'zero_dpi': {'dpi': 0}}.items()}
    mixed = postprocessing.PostprocessContext.from_report({
        'system_type': 'molecular', 'comparison_table': [
            {'case_id': 'bohr-1', 'bond': 1, 'energy': '-1 Ha'},
            {'case_id': 'bohr-2', 'bond': 2, 'energy': '-27.211386 eV'},
        ], 'cases': [{'request': {'unit': 'Bohr'}}],
    })
    mixed_result = outcome(lambda: postprocessing._normalized_plot_spec(mixed, {'x': 'bond', 'y': 'energy'}))
    rows = [{'x': x, 'y': y, 'z': x + y} for x in [0, 1] for y in [0, 1]]
    rows.append({'x': 0, 'y': 0, 'z': 999})
    heat_context = postprocessing.PostprocessContext.from_report({'comparison_table': rows})
    heat_result = outcome(lambda: postprocessing.run_plot_spec(heat_context,
        {'tool': 'heatmap', 'x': 'x', 'y': 'y', 'color': 'z', 'dpi': 60}, output_dir=directory))
    return {
        'backend_accepted_case_ids': [row['case_id'] for row in context.comparison_table],
        'plot_parameter_outcomes': options,
        'mixed_units': mixed_result,
        'duplicate_heatmap': heat_result,
        'backend_numeric_cells': {value: postprocessing._parse_numeric_value(value) for value in ['1.2 eV extra', '1e999']},
    }


def source_inventory():
    files = sorted([*ROOT.glob('pyscf_agent/**/*.py'), *ROOT.glob('computational_study_agent/**/*.py')])
    handlers = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ExceptHandler):
                kind = ast.unparse(node.type) if node.type else 'bare'
                handlers.append({'path': str(path.relative_to(ROOT)), 'line': node.lineno, 'type': kind,
                                 'body': ast.unparse(ast.Module(body=node.body, type_ignores=[]))[:220]})
    js = sorted([*ROOT.glob('pyscf_agent/web_assets/*.js'), *ROOT.glob('computational_study_agent/web_assets/*.js')])
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files + js}
    return {'python_files': len(files), 'javascript_files': len(js), 'exception_handlers': len(handlers),
            'broad_handlers': [h for h in handlers if h['type'] in ('Exception', 'BaseException', 'bare')],
            'source_sha256': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    goal = 'Keep all 16 cases and change the DMET execution_mode to finite_graph.'
    seed = fixture()
    frontend = frontend_request(seed, goal)
    payload = frontend['request']['payload']
    data = {'provenance': 'Reconstructed 4x8 open square graph, 32 sites, 52 bonds, 4x4 U/V grid. Not the lost historical request.',
            'measurement': 'Exact local tokenization of message text and serialized schema in two encodings. Excludes provider-specific chat framing, schema serialization, hidden reasoning and cached-token pricing. No live LLM calls.',
            'fixture': seed, 'frontend': frontend, 'wiki_query': build_study_wiki_query(goal, payload['study_spec']),
            'planner': {}, 'analysis': {}}
    with tempfile.TemporaryDirectory(prefix='pyscf-context-audit-') as tmp, patch.dict(os.environ, {
        'PYSCF_AGENT_LLM_BASE_URL': 'https://offline-audit.invalid/v1',
        'PYSCF_AGENT_LLM_MODEL': 'offline-audit-fixture',
    }):
        for mode in ('success', 'repair', 'transport_then_repair', 'invalid_twice'):
            result, calls = planner_replay(payload, mode, Path(tmp))
            assert len(calls) == {'success': 1, 'repair': 2, 'transport_then_repair': 3, 'invalid_twice': 2}[mode]
            assert result['http_status'] == (400 if mode == 'invalid_twice' else 200), result
            assert result['case_count'] == (0 if mode == 'invalid_twice' else 16), result
            assert result['returned_study_spec'] == (mode != 'invalid_twice'), result
            if result.get('error'):
                result['error'] = result['error'].split('\nDiagnostic:')[0]
            data['planner'][mode] = result
            if mode == 'success':
                (args.output / 'planner-request.json').write_text(json.dumps(calls[0], ensure_ascii=False, indent=2) + '\n')
        for size in (1, 16, 64, 256):
            data['analysis']['synthetic_%d' % size], calls = analysis_replay(synthetic_report(size))
            if size == 16:
                (args.output / 'analysis-request.json').write_text(json.dumps(calls[0], ensure_ascii=False, indent=2) + '\n')
        preparation = copy.deepcopy(frontend['preparation_request']['payload'])
        preparation['work_dir'] = tmp
        with patch.object(llm_planner, '_default_http_post', side_effect=AssertionError('Preparation must not call the LLM')) as spy:
            status, _, body = handle_saved_study_request(json.dumps(preparation).encode(), action='prepare')
        prepared = json.loads(body)
        assert status.value == 201, prepared
        saved_plan = json.loads(next(Path(tmp).glob('*/study-plan.json')).read_text())
        assert len(saved_plan['cases']) == 16
        data['preparation'] = {'http_status': status.value, 'llm_calls': spy.call_count,
                               'saved_case_count': len(saved_plan['cases'])}
        saved = ROOT / 'runs/20260916-144523-9cf9e952/study-report.json'
        if saved.exists():
            saved_report = json.loads(saved.read_text())
            # Analyze a copy; avoid writing analysis artifacts into the saved run.
            saved_report.pop('work_dir', None)
            data['analysis']['saved_report'], _ = analysis_replay(saved_report)
            data['analysis']['saved_report']['source'] = str(saved.relative_to(ROOT))
        data['postprocessing'] = postprocessing_probes(Path(tmp))
        feedback_calls = []
        for reply in ('无需调整。', 'No adjustment is needed.'):
            def feedback_post(_url, body, _headers, _timeout):
                feedback_calls.append(copy.deepcopy(body))
                return {'choices': [{'message': {'content': reply}}]}
            feedback = llm.build_execution_feedback(
                'Run the requested calculation.',
                {'execution_status': 'failed', 'errors': ['Synthetic solver failure for the context audit.']},
                http_post=feedback_post, locale='en',
            )
            data.setdefault('task_feedback', []).append({'fixture_reply': reply, 'returned_feedback': feedback,
                                                        'calls': [request_count(feedback_calls[-1])]})
    data['analysis_contract_overhead'] = {
        'note': 'Available complete Registry catalog for reference; current result analysis sends only selected contracts in the captured request.',
        'contract_count': len(result_analysis_output_contracts()),
        'user_catalog': count(result_analysis_output_contracts()),
        'system_catalog': count(result_analysis_contract_prompt()),
    }
    data['inventory'] = source_inventory()
    (args.output / 'audit.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'planner': {mode: {k: v for k, v in value.items() if k in ('http_status', 'case_count', 'sum_message_content', 'validation_issues')}
                    for mode, value in data['planner'].items()},
        'analysis': {name: value['calls'][0]['message_content'] for name, value in data['analysis'].items()},
        'contracts': data['analysis_contract_overhead'],
        'inventory': {k: data['inventory'][k] for k in ('python_files', 'javascript_files', 'exception_handlers')},
    }, indent=2))


if __name__ == '__main__':
    main()
