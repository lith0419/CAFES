from __future__ import annotations

from pyscf_agent.serialization import json_default

import argparse
import json
from pathlib import Path

from pyscf_agent.executors import (
    add_executor_arguments,
    create_task_executor_from_args,
)

from .application import StudyApplicationService
from computational_study_agent.datasets.hamiltonian.contracts import (
    HamiltonianDatasetSpec,
    MolecularGeometry,
)
from .schema import StudyPlan, StudySpec


def _load_json(path: str):
    return json.loads(Path(path).expanduser().read_text(encoding='utf-8'))


def _load_seed_geometries(path: str):
    text = Path(path).expanduser().read_text(encoding='utf-8')
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get('seed_geometries')
    if not isinstance(payload, list):
        raise ValueError(
            'Seed geometry input must be a JSON array, JSONL file, or an object '
            'with a seed_geometries array.'
        )
    return [MolecularGeometry.from_dict(item) for item in payload]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run a PySCF Agent computational study.')
    parser.add_argument('study_spec', nargs='?', help='Path to a StudySpec JSON file')
    parser.add_argument(
        '--dataset-spec',
        default=None,
        help='Path to a HamiltonianDatasetSpec JSON file',
    )
    parser.add_argument(
        '--seed-geometries',
        default=None,
        help='Path to seed geometries as JSON, JSONL, or {"seed_geometries": [...]}.',
    )
    parser.add_argument('--work-dir', default=None, help='Directory where the study run should be written')
    parser.add_argument('--locale', default='en', help='Locale passed to single-case workflows')
    parser.add_argument('--plan-only', action='store_true', help='Print the expanded StudyPlan without executing cases')
    parser.add_argument('--saved-plan', help='Existing StudyPlan JSON, retaining its study_id')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--collect-only', action='store_true', help='Collect existing jobs without submitting any task')
    actions.add_argument('--reconcile-submission', help='Server submission.json path to associate with an unknown execution')
    parser.add_argument('--execution-receipt', help='Local execution receipt to reconcile (defaults to the study root)')
    parser.add_argument('--postprocess', action='store_true', help='Generate built-in postprocessing plots after the study run')
    parser.add_argument('--postprocess-spec', default=None, help='Path to a JSON plot spec or list of plot specs')
    add_executor_arguments(parser)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.collect_only or args.reconcile_submission) and not args.saved_plan:
        parser.error('Recovery actions require --saved-plan to retain the original study identity')

    try:
        task_executor = create_task_executor_from_args(args)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    service = StudyApplicationService(task_executor=task_executor)
    if args.saved_plan:
        if args.study_spec or args.dataset_spec or args.seed_geometries:
            parser.error('--saved-plan cannot be combined with new study inputs')
        plan = StudyPlan.from_dict(_load_json(args.saved_plan))
    elif args.dataset_spec or args.seed_geometries:
        if not args.dataset_spec or not args.seed_geometries:
            parser.error('--dataset-spec and --seed-geometries must be provided together')
        if args.study_spec:
            parser.error('study_spec cannot be combined with dataset input options')
        try:
            dataset_spec = HamiltonianDatasetSpec.from_dict(_load_json(args.dataset_spec))
            seed_geometries = _load_seed_geometries(args.seed_geometries)
            plan = service.build_hamiltonian_dataset_plan(dataset_spec, seed_geometries)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
    else:
        if not args.study_spec:
            parser.error('study_spec is required unless dataset input options are provided')
        spec = StudySpec.from_dict(_load_json(args.study_spec))
        plan = service.build_plan(spec)
    if args.plan_only:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, default=json_default, allow_nan=False))
        return 0
    if args.execution_receipt and not args.reconcile_submission:
        parser.error('--execution-receipt requires --reconcile-submission')
    if args.reconcile_submission:
        result = service.reconcile_execution(
            plan, submission_path=args.reconcile_submission, work_dir=args.work_dir,
            receipt_file=args.execution_receipt,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.collect_only:
        report = service.collect_study(plan, work_dir=args.work_dir, locale=args.locale)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=json_default, allow_nan=False))
        return 0

    postprocess_specs = None
    if args.postprocess_spec:
        raw_specs = _load_json(args.postprocess_spec)
        postprocess_specs = raw_specs if isinstance(raw_specs, list) else [raw_specs]
    report = service.run_study(
        plan,
        work_dir=args.work_dir,
        locale=args.locale,
        postprocess=args.postprocess or bool(postprocess_specs),
        postprocess_specs=postprocess_specs,
    )
    print(json.dumps({
        'status': report.status,
        'summary': report.summary,
        'work_dir': report.work_dir,
        'artifacts': report.artifacts,
    }, ensure_ascii=False, indent=2, default=json_default, allow_nan=False))
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
