import json
import tempfile
import unittest
from pathlib import Path
from computational_study_agent.study_state import load_checkpoint

class CheckpointReferenceTests(unittest.TestCase):
    def test_loads_existing_file_backed_report_with_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'report.json').write_text(json.dumps({'run_id':'case-0001','execution_status':'succeeded'}))
            payload={'schema':'pyscf-agent.study-state.v1','study_id':'old-study','cases':{'case-0001':{'attempt_count':1,'task_report_ref':{'path':'report.json'},'task_summary':{'run_id':'case-0001','execution_status':'succeeded'},'execution':{'run_id':'case-0001','pending':False,'attempt_count':1}}}}
            (root/'study-state.json').write_text(json.dumps(payload))
            loaded=load_checkpoint(root/'study-state.json','old-study')
            self.assertEqual(loaded['cases']['case-0001']['task_report']['execution_status'],'succeeded')
            self.assertNotIn('task_report_ref',loaded['cases']['case-0001'])
            self.assertEqual(json.loads((root/'study-state.json').read_text()),payload)
