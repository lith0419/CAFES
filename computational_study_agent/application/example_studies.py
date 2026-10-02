"""Install source-checkout examples as ordinary saved Studies."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from pyscf_agent.paths import resolve_work_dir
from ..example_storage import materialize_study


# Console scripts can load an installed package while running in a source checkout.
EXAMPLES_ROOT = next((root for root in (
    Path.cwd() / 'examples', Path(__file__).resolve().parents[2] / 'examples',
) if (root / 'manifest.json').is_file()), Path(__file__).resolve().parents[2] / 'examples')


def list_examples(self):
    manifest = EXAMPLES_ROOT / 'manifest.json'
    if not manifest.is_file():
        return []
    return [{key: item[key] for key in ('study_id', 'name', 'cases')}
            for item in json.loads(manifest.read_text())['studies']]


def import_example(self, study_id: str, *, work_dir=None):
    items = {item['study_id']: item for item in self.list_examples()}
    if study_id not in items or Path(study_id).name != study_id or study_id in ('.', '..'):
        raise ValueError('Unknown bundled example')
    root = resolve_work_dir(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / study_id
    if destination.is_symlink():
        raise ValueError('Example destination must not be a symbolic link')
    if destination.exists():
        marker = destination / 'example-origin.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('study_id') != study_id:
            raise ValueError('An unrelated Study already uses this ID; choose another work directory')
    else:
        source = (EXAMPLES_ROOT / 'studies' / study_id).resolve()
        source.relative_to((EXAMPLES_ROOT / 'studies').resolve())
        with tempfile.TemporaryDirectory(prefix='.example-import-', dir=root) as temporary:
            staged = Path(temporary) / study_id
            shutil.copytree(source, staged)
            materialize_study(staged)
            report_path = staged / 'study-report.json'
            report = json.loads(report_path.read_text())
            plan = json.loads((staged / 'study-plan.json').read_text())
            if report['study_id'] != study_id or plan['study_id'] != study_id:
                raise ValueError('Example files belong to a different Study')
            report['work_dir'] = str(destination)
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            (staged / 'example-origin.json').write_text(json.dumps(items[study_id], indent=2) + '\n')
            staged.rename(destination)
    return {'study_id': study_id, 'work_dir': str(root), 'execution_target': 'local'}
