import json
from pathlib import Path
import tempfile
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from computational_study_agent.application import StudyApplicationService
from computational_study_agent.application import cancellation
from computational_study_agent.application.background import inspect_invocation
from computational_study_agent.web_api import handle_saved_study_request


class StudyCancellationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root/'study-1'
        self.directory.mkdir()
        self.inv = {'study_id':'study-1', 'request':{'mode':'static'},
                    'execution_config':{'execution_target':'local'},
                    'started_at':'start', 'finished_at':None, 'error':None}
        self.save_inv()
        self.handle = {'schema':'pyscf-agent.job-handle.v1','job_id':'one','run_id':'one',
                       'executor_id':'local-process','work_dir':str(self.directory/'cases'),
                       'submitted_at':'today'}
        case=self.directory/'cases'/'one';case.mkdir(parents=True)
        (case/'job-state.json').write_text(json.dumps({'handle':self.handle}))
        self.executor=Mock()
        self.executor.describe.return_value={'executor_id':'local-process'}
        self.executor.status.return_value=SimpleNamespace(terminal=False)
        self.executor.cancel.return_value=SimpleNamespace(terminal=True,to_dict=lambda:{'state':'cancelled','terminal':True})
        self.service=StudyApplicationService(task_executor=self.executor)
    def save_inv(self):
        (self.directory/'study-invocation.json').write_text(json.dumps(self.inv))
    def stop(self):
        return self.service.stop_study('study-1',work_dir=str(self.root))
    def test_reconnected_worker_cancelled_without_coordinator(self):
        self.assertTrue(self.stop()['cancelled'])
        self.executor.cancel.assert_called_once()
        self.assertTrue(inspect_invocation(self.directory)['cancelled'])
        self.assertTrue((self.directory/'cases'/'one'/'job-state.json').exists())
    def test_terminal_results_preserved_and_repeated_stop_is_safe(self):
        self.executor.status.return_value=SimpleNamespace(terminal=True,to_dict=lambda:{'state':'completed','terminal':True})
        self.assertTrue(self.stop()['cancelled']);self.assertTrue(self.stop()['cancelled'])
        self.executor.cancel.assert_not_called()
    def test_coordinator_terminated_before_cancelling_workers(self):
        order=[]
        self.executor.cancel.side_effect=lambda h: (order.append('worker') or SimpleNamespace(terminal=True,to_dict=lambda:{'state':'cancelled'}))
        with patch.object(cancellation,'study_is_locked',return_value=True), patch.object(cancellation,'_coordinator_pid',return_value=123456), patch.object(cancellation.os,'kill',side_effect=lambda *a:order.append('coordinator')):
            self.assertTrue(self.stop()['cancelled'])
        self.assertEqual(order,['coordinator','worker'])
    def test_remote_study_rejected_before_signalling(self):
        self.inv['execution_config']['execution_target']='remote';self.save_inv()
        with patch.object(cancellation.os,'kill') as kill:
            with self.assertRaisesRegex(ValueError,'local'):self.stop()
            kill.assert_not_called()
        self.executor.cancel.assert_not_called()
    def test_worker_outside_study_is_not_cancelled(self):
        self.handle['work_dir']=str(self.root/'other')
        (self.directory/'cases'/'one'/'job-state.json').write_text(json.dumps({'handle':self.handle}))
        result=self.stop();self.assertFalse(result['cancelled']);self.assertTrue(result['errors'])
        self.executor.cancel.assert_not_called()
    def test_unconfirmed_termination_is_reported(self):
        self.executor.cancel.side_effect=RuntimeError('still running')
        result=self.stop();self.assertEqual(result['status'],'stop_incomplete')
        self.assertFalse(inspect_invocation(self.directory)['cancelled'])
    def test_pid_matching_rejects_other_study_and_shell_wrapper(self):
        with patch.object(cancellation.subprocess,'check_output',return_value='1 python -m computational_study_agent.application.background --study-directory /other\n2 zsh -c echo hi\n'):
            with self.assertRaises(ValueError):cancellation._coordinator_pid(self.directory)
    @unittest.skipUnless(shutil.which('node'), 'Node required')
    def test_stop_button_states_and_duplicate_click(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(['node', str(Path(__file__).with_name('study_stop_ui.cjs')),
            str(root/'computational_study_agent/web_assets/planner-saved-studies.js')],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stop_api_uses_saved_identity(self):
        with patch('computational_study_agent.web_api.get_study_application_service',return_value=self.service):
            status,headers,body=handle_saved_study_request(json.dumps({'study_id':'study-1','work_dir':str(self.root),'execution_target':'local'}).encode(),action='stop')
        self.assertEqual(status,200);self.assertTrue(json.loads(body)['cancelled'])

if __name__=='__main__':unittest.main()
