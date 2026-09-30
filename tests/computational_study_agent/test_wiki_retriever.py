from __future__ import annotations

import unittest
from unittest.mock import patch

from computational_study_agent.wiki_retriever import (
    WikiPage,
    _tokenize,
    build_study_wiki_query,
    build_wiki_evidence_pack,
    format_wiki_evidence_for_prompt,
    retrieve_wiki_pages,
)


def page(title, body, *, slug='test-page'):
    return WikiPage(title=title, slug=slug, summary='', body=body,
                    aliases=[], links=[], path=slug + '.md')


class WikiRetrievalTests(unittest.TestCase):
    def test_study_metadata_does_not_change_scientific_query_or_ranking(self):
        goal = 'Keep all cases and use finite_graph for DMET.'
        seed = {
            'system_type': 'model_hamiltonian',
            'base_task': {'solver': {'name': 'dmet', 'options': {
                'execution_mode': 'finite_graph', 'impurity_solver': 'fci',
            }}},
            'sweep': {'U': [2, 4], 'V': [0, 1], 't': [-1]},
        }
        noisy_seed = dict(seed, name='Artifact Storage Metadata',
                          schema='pyscf-agent.study-spec.v1',
                          base_model_input_file='/tmp/artifact/storage/model.py',
                          resource_policy={'name': 'artifact storage'})
        noisy_seed['base_task'] = dict(seed['base_task'],
                                       atom='C 0 0 0; O 0 0 1.2',
                                       output_dir='/tmp/artifact/storage')
        clean = build_study_wiki_query(goal, seed)
        noisy = build_study_wiki_query(goal, noisy_seed)
        self.assertEqual(clean, noisy)
        self.assertTrue({'u', 'v', 't', 'dmet', 'finite_graph', 'fci'} <= set(_tokenize(noisy)))
        self.assertTrue({'name', 'schema', 'solver', 'options', 'base_task', 'storage'}.isdisjoint(
            _tokenize(noisy)))
        pages = [page('DMET', 'DMET finite_graph supports an intersite V.', slug='dmet'),
                 page('Storage', 'name options solver schema artifact storage ' * 100, slug='storage')]
        with patch('computational_study_agent.wiki_retriever.load_wiki_pages', return_value=pages):
            hits = retrieve_wiki_pages(noisy, include_neighbors=False)
        self.assertEqual([hit.page.slug for hit in hits], ['dmet'])

    def test_scientific_choices_in_scans_and_overrides_remain_searchable(self):
        seed = {
            'base_task': {'method': 'hf', 'basis': 'def2-svp'},
            'sweep': {'method': ['hf', 'mp2']},
            'case_design': {
                'variables': {'bond_length': [1, 2]},
                'template': {'operations': [{'op': 'set_site_parameter', 'parameter': 'U', 'value': 4}]},
                'overrides': [{'selector': {'bond_length': 2}, 'request_updates': {
                    'method': 'casscf', 'solver': {'name': 'block2_dmrg'},
                    'active_space': {'ncas': 4, 'nelecas': 4},
                }}],
            },
        }
        terms = set(_tokenize(build_study_wiki_query('Revise the selected point.', seed)))
        self.assertTrue({'hf', 'mp2', 'casscf', 'block2_dmrg', 'def2-svp', 'bond_length',
                         'set_site_parameter', 'u', 'ncas', 'nelecas'} <= terms)
        self.assertTrue({'selector', 'request_updates', 'variables', 'name'}.isdisjoint(terms))

    def test_single_letter_parameters_match_whole_tokens(self):
        pages = [page('Onsite interaction', 'The parameter U controls repulsion.', slug='u'),
                 page('Intersite interaction', 'The parameter V controls repulsion.', slug='v'),
                 page('Hopping', 'The parameter t controls hopping.', slug='t'),
                 page('Unrelated', 'Future values are stored here.', slug='unrelated')]
        with patch('computational_study_agent.wiki_retriever.load_wiki_pages', return_value=pages):
            for symbol in ('U', 'V', 't'):
                with self.subTest(symbol=symbol):
                    hits = retrieve_wiki_pages(symbol, include_neighbors=False)
                    self.assertEqual([hit.page.slug for hit in hits], [symbol.lower()])

    def test_punctuation_does_not_change_scientific_identifiers(self):
        terms = set(_tokenize('Scan U/V/t. Use finite_graph, DMRG-CASSCF and def2-SVPD.'))
        self.assertTrue({'u', 'v', 't', 'finite_graph', 'dmrg-casscf', 'def2-svpd'} <= terms)
        self.assertNotIn('finite_graph.', _tokenize('Use finite_graph.'))
        self.assertNotIn('t', _tokenize("Don't change the method."))

    def test_disabled_features_and_solver_file_paths_are_not_search_context(self):
        query = build_study_wiki_query('Inspect the DMET settings.', {
            'base_task': {
                'solver': {'name': 'dmet', 'options': {
                    'execution_mode': 'finite_graph', 'checkpoint_file': '/tmp/storage/checkpoint.dat',
                }},
                'active_space': {'enabled': False, 'selection_method': 'avas', 'ncas': 4},
            },
        })
        terms = set(_tokenize(query))
        self.assertTrue({'dmet', 'finite_graph', 'execution_mode'} <= terms)
        self.assertTrue({'checkpoint_file', 'storage', 'active_space', 'avas', 'ncas'}.isdisjoint(terms))

    def test_incomplete_scan_input_can_still_be_used_for_drafting(self):
        query = build_study_wiki_query('Complete this DMET scan.', {
            'base_task': {'solver': 'dmet'},
            'case_design': {'variables': 2, 'cases': 3, 'template': {'operations': None}},
        })
        self.assertIn('dmet', query)

    def test_evidence_keeps_a_relevant_rule_after_a_long_introduction(self):
        rule = 'With intersite V, use finite_graph; translated representative DMET is not supported.'
        body = '# Solver Guide\n\n' + ('General background information. ' * 180)
        body += '\n\n## DMET execution mode\n\n' + rule
        with patch('computational_study_agent.wiki_retriever.load_wiki_pages',
                   return_value=[page('Model solvers', body)]):
            evidence = build_wiki_evidence_pack('DMET finite_graph intersite V', max_chars_per_page=350)
        excerpt = evidence['pages'][0]['content']
        self.assertIn(rule, excerpt)
        self.assertIn('## DMET execution mode', excerpt)
        self.assertLessEqual(len(excerpt), 350)
        self.assertNotIn('General background information.', excerpt)

    def test_multiple_relevant_passages_keep_their_order_and_headings(self):
        first = 'DMET constructs the impurity bath from the mean field.'
        second = 'For intersite interactions, choose execution_mode=finite_graph.'
        body = '# Model Guide\n\n## Bath\n\n' + first
        body += '\n\n## Background\n\n' + ('Unrelated background. ' * 120)
        body += '\n\n## Execution\n\n' + second
        with patch('computational_study_agent.wiki_retriever.load_wiki_pages',
                   return_value=[page('Model solvers', body)]):
            evidence = build_wiki_evidence_pack('DMET bath execution_mode finite_graph', max_chars_per_page=500)
        excerpt = evidence['pages'][0]['content']
        self.assertIn(first, excerpt)
        self.assertIn(second, excerpt)
        self.assertLess(excerpt.index(first), excerpt.index(second))
        self.assertIn('## Bath', excerpt)
        self.assertIn('## Execution', excerpt)
        self.assertNotIn('Unrelated background.', excerpt)
        self.assertLessEqual(len(excerpt), 500)

    def test_a_match_at_the_end_of_one_oversized_paragraph_is_retained(self):
        rule = 'Use execution_mode=finite_graph for intersite V in DMET.'
        body = '# Options\n\n' + ('General background. ' * 200) + rule
        with patch('computational_study_agent.wiki_retriever.load_wiki_pages',
                   return_value=[page('Solver options', body)]):
            evidence = build_wiki_evidence_pack('execution_mode finite_graph intersite V DMET',
                                                max_chars_per_page=250)
        excerpt = evidence['pages'][0]['content']
        self.assertIn(rule, excerpt)
        self.assertIn('# Options', excerpt)
        self.assertLessEqual(len(excerpt), 250)

    def test_a_late_list_item_keeps_its_complete_constraint(self):
        rule = '- For intersite V, execution_mode must be finite_graph; do not use translated DMET.'
        body = '# Solver\n\n## Restrictions\n\n' + ('- General background information.\n' * 150) + rule
        with patch('computational_study_agent.wiki_retriever.load_wiki_pages',
                   return_value=[page('DMET', body)]):
            evidence = build_wiki_evidence_pack('DMET intersite V execution_mode finite_graph',
                                                max_chars_per_page=300)
        excerpt = evidence['pages'][0]['content']
        self.assertIn(rule, excerpt)
        self.assertIn('## Restrictions', excerpt)
        self.assertLessEqual(len(excerpt), 300)

    def test_excerpt_budget_is_respected_including_markers_and_headings(self):
        body = '# DMET\n\n' + ('General information. ' * 150) + '\n\n## Options\n\nUse finite_graph.'
        with patch('computational_study_agent.wiki_retriever.load_wiki_pages',
                   return_value=[page('DMET', body)]):
            for limit in (0, 1, 5, 20, 90, 1800):
                with self.subTest(limit=limit):
                    evidence = build_wiki_evidence_pack('DMET finite_graph', max_chars_per_page=limit)
                    self.assertLessEqual(len(evidence['pages'][0]['content']), limit)

    def test_short_pages_keep_the_complete_body(self):
        body = '# DMET\n\nUse finite_graph for intersite V.'
        with patch('computational_study_agent.wiki_retriever.load_wiki_pages',
                   return_value=[page('DMET', body)]):
            evidence = build_wiki_evidence_pack('DMET')
        self.assertEqual(evidence['pages'][0]['content'], body)


class PublishedWikiGuidanceTests(unittest.TestCase):
    """Exercise the shipped corpus and bounded excerpts, not synthetic pages."""

    def evidence(self, query):
        pack = build_wiki_evidence_pack(query)
        self.assertLessEqual(len(pack['pages']), 8)
        for item in pack['pages']:
            self.assertLessEqual(len(item['content']), 1800)
        return {item['title']: item['content'] for item in pack['pages']}

    def test_failure_diagnostics_keep_physics_and_evidence_confidence_distinct(self):
        pages = self.evidence('SCF nonconvergence strong correlation physics_level routing_level confidence')
        text = pages['Correlation Diagnostic Interpretation']
        self.assertIn('does not prove strong correlation', text)
        self.assertIn('not a probability of correctness', text)
        self.assertIn('next-method route', text)

    def test_dmrg_final_solve_does_not_claim_orbital_reoptimization(self):
        pages = self.evidence('DMRG CASSCF final fixed-orbital M discarded weight orbital convergence')
        self.assertIn('does not reoptimize', pages['PySCF Roadmap Multireference Active Space'])
        self.assertIn('discarded', pages['Correlation Diagnostic Interpretation'])

    def test_qh9_dynamics_is_retrieved_as_a_scoped_implemented_job(self):
        pages = self.evidence('QH9 molecular dynamics trajectory qh9_relaxed_scf B3LYP def2-svp')
        text = pages['PySCF Roadmap Properties and Extensions']
        self.assertIn('`implemented`', text)
        self.assertIn('qh9_relaxed_scf', text)
        self.assertIn('mid-trajectory restart', text)

    def test_hubbard_dmet_request_keeps_explicit_solver_and_finite_graph_rules(self):
        pages = self.evidence(
            'Run the supplied Hubbard Honeycomb model with DMET. Scan U = 2, 4 and V = 0, 0.5. '
            'Use FCI as the impurity solver and finite_graph execution. Report ground-state energies.'
        )
        self.assertIn('finite_graph', pages['Model Hamiltonian Solver Support'])
        text = '\n'.join(pages.values())
        self.assertIn('static', text)
        self.assertNotIn('correlation energy, filling', text)

    def test_prompt_preserves_registry_authority_and_distinguishes_evidence(self):
        text = format_wiki_evidence_for_prompt(build_wiki_evidence_pack('correlation diagnostics'))
        self.assertIn('backend validation remain authoritative', text)
        self.assertIn('heuristics, roadmap proposals, and archived evidence', text)
        self.assertNotIn('the rule wins', text)


if __name__ == '__main__':
    unittest.main()
