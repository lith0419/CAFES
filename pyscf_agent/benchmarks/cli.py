from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .suite import list_benchmarks, run_benchmark_suite, write_benchmark_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='pyscf-agent-benchmark',
        description='Run the versioned PySCF Agent scientific benchmark suite.',
    )
    parser.add_argument(
        'benchmarks',
        nargs='*',
        help='Benchmark IDs. Omit to run the complete suite.',
    )
    parser.add_argument('--list', action='store_true', help='List benchmark IDs and exit.')
    parser.add_argument('--work-dir', help='Directory for calculation artifacts.')
    parser.add_argument(
        '--include-scaling',
        action='store_true',
        help='Include the opt-in medium-scale block2 benchmarks when no IDs are supplied.',
    )
    parser.add_argument('--output', help='Write the JSON benchmark report to this path.')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON output.')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for item in list_benchmarks():
            print('{0}\t{1}\t{2}'.format(item['id'], item['domain'], item['description']))
        return 0

    report = run_benchmark_suite(
        benchmark_ids=args.benchmarks or None,
        work_dir=args.work_dir,
        include_scaling=args.include_scaling,
    )
    if args.output:
        write_benchmark_report(Path(args.output), report)
    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        sort_keys=bool(args.pretty),
    ))
    return 0 if report['status'] == 'passed' else 1


__all__ = ['build_parser', 'main']
