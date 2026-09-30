"""Start a stdio MCP server using the existing executor configuration."""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

from ..application import CalculationApplicationService
from ..paths import resolve_work_dir
from ..executors import LocalProcessExecutor
from ..executors.factory import add_executor_arguments, create_task_executor_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='PySCF Agent task and Study MCP server (stdio).')
    parser.add_argument('--work-dir', default=None, help='Calculation output root; defaults to PYSCF_AGENT_RUNS_DIR or the user data directory.')
    add_executor_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from .server import create_server
        from computational_study_agent.application import StudyApplicationService
        from ..workbench import LocalWorkbench
    except ModuleNotFoundError as exc:
        if exc.name == 'mcp':
            parser.error("MCP support is optional. Install it with: python -m pip install 'pyscf-agent[mcp]'")
        raise
    logging.basicConfig(level=logging.WARNING)  # stderr; stdout belongs to MCP.
    try:
        executor = (LocalProcessExecutor(wall_time_seconds=args.local_wall_time_seconds)
                    if args.executor == 'local' else create_task_executor_from_args(args))
        server = create_server(
            CalculationApplicationService(task_executor=executor),
            work_dir=resolve_work_dir(args.work_dir),
            executor_description=executor.describe(),
            workbench=LocalWorkbench.from_args(args),
            study_service=StudyApplicationService(task_executor=executor, execution_config={
                'execution_target': args.executor, 'slurm_config': args.slurm_config,
                'slurm_profile': args.slurm_profile, 'remote_config': args.remote_config,
                'remote_profile': args.remote, 'local_wall_time_seconds': args.local_wall_time_seconds,
            }),
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    server.run(transport='stdio')
    return 0
