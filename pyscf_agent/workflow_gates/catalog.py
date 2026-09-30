from __future__ import annotations

from typing import List

from .contracts import EvidenceContract, GateContract
from ..workflow_modules.contracts import ActivationRule, ModuleCompatibility


def _evidence(name: str, path: str, *, optional: bool = False) -> EvidenceContract:
    return EvidenceContract(name=name, path=path, optional=optional)


def build_default_task_gate_contracts() -> List[GateContract]:
    return [
        GateContract(
            gate_id='core.task_compilation',
            version='1.0',
            scope='task',
            permitted_hooks=('task.after_compile',),
            default_hook='task.after_compile',
            evaluator_id='pyscf_agent.gates.compilation',
            required_evidence=(
                _evidence('workflow_configuration', 'workflow_configuration'),
                _evidence('workflow_provenance', 'workflow_provenance'),
            ),
            mandatory=True,
            priority=0,
            description='Reject an incomplete or rejected module compilation before execution.',
        ),
        GateContract(
            gate_id='task.input_readiness',
            version='1.0',
            scope='task',
            permitted_hooks=('task.before_execute',),
            default_hook='task.before_execute',
            evaluator_id='pyscf_agent.gates.input_readiness',
            required_evidence=(
                _evidence('task_spec', 'task_spec'),
                _evidence('validation_errors', 'validation_errors', optional=True),
                _evidence('generated_input', 'generated_input', optional=True),
            ),
            always_select=True,
            priority=10,
            description='Confirm that the validated TaskSpec and generated input are executable.',
        ),
        GateContract(
            gate_id='task.block2_availability',
            version='1.0',
            scope='task',
            permitted_hooks=('task.before_execute',),
            default_hook='task.before_execute',
            evaluator_id='pyscf_agent.gates.block2_availability',
            required_evidence=(
                _evidence('task_spec', 'task_spec'),
                _evidence('solver_provider', 'solver_provider', optional=True),
            ),
            compatibility=ModuleCompatibility(task_types=('molecular', 'model_hamiltonian')),
            activation_rules=(ActivationRule('solver.name', 'in', ('block2_dmrg', 'block2', 'block2-dmrg', 'dmrg')),),
            after=('task.input_readiness',),
            priority=20,
            description='Require one usable block2 provider distribution before DMRG execution.',
        ),
        GateContract(
            gate_id='task.execution_quality',
            version='1.0',
            scope='task',
            permitted_hooks=('task.after_recovery',),
            default_hook='task.after_recovery',
            evaluator_id='pyscf_agent.gates.execution_quality',
            required_evidence=(
                _evidence('execution_status', 'execution_status'),
                _evidence('retry_count', 'retry_count', optional=True),
                _evidence('max_retries', 'max_retries', optional=True),
                _evidence('errors', 'errors', optional=True),
                _evidence('structured_results', 'structured_results', optional=True),
            ),
            always_select=True,
            priority=10,
            description='Classify the final execution state after bounded automatic recovery.',
        ),
        GateContract(
            gate_id='task.artifact_integrity',
            version='1.0',
            scope='task',
            permitted_hooks=('task.before_finalize',),
            default_hook='task.before_finalize',
            evaluator_id='pyscf_agent.gates.artifact_integrity',
            required_evidence=(
                _evidence('execution_status', 'execution_status'),
                _evidence('structured_results', 'structured_results', optional=True),
                _evidence('artifacts', 'artifacts', optional=True),
            ),
            always_select=True,
            priority=10,
            description='Check result presence and artifact references before TaskReport assembly.',
        ),
    ]


def build_default_study_gate_contracts() -> List[GateContract]:
    return [
        GateContract(
            gate_id='core.study_compilation',
            version='1.0',
            scope='study',
            permitted_hooks=('study.after_compile',),
            default_hook='study.after_compile',
            evaluator_id='computational_study_agent.gates.compilation',
            required_evidence=(
                _evidence('workflow_configuration', 'workflow_configuration', optional=True),
                _evidence('workflow_provenance', 'workflow_provenance', optional=True),
            ),
            mandatory=True,
            priority=0,
            description='Confirm that the study module graph compiled successfully.',
        ),
        GateContract(
            gate_id='study.active_space_approval',
            version='1.0',
            scope='study',
            permitted_hooks=('study.before_execute',),
            default_hook='study.before_execute',
            evaluator_id='computational_study_agent.gates.active_space_approval',
            required_evidence=(
                _evidence('study_report', 'study_report', optional=True),
                _evidence('study_plan', 'study_plan', optional=True),
            ),
            always_select=True,
            priority=10,
            description='Require explicit approval for proposed CASSCF/CASCI active spaces.',
        ),
        GateContract(
            gate_id='study.resource_feasibility',
            version='1.0',
            scope='study',
            permitted_hooks=('study.before_execute',),
            default_hook='study.before_execute',
            evaluator_id='computational_study_agent.gates.resource_feasibility',
            required_evidence=(
                _evidence('study_report', 'study_report', optional=True),
                _evidence('study_plan', 'study_plan', optional=True),
            ),
            always_select=True,
            priority=20,
            description='Pause plans that cross configured determinant, work, or memory review thresholds.',
        ),
        GateContract(
            gate_id='study.initial_scan_quality',
            version='1.0',
            scope='study',
            permitted_hooks=('study.after_execute',),
            default_hook='study.after_execute',
            evaluator_id='computational_study_agent.gates.initial_scan_quality',
            required_evidence=(_evidence('study_report', 'study_report', optional=True),),
            always_select=True,
            priority=10,
            description='Reject adaptive routing when initial calculations did not produce valid diagnostics.',
        ),
        GateContract(
            gate_id='study.execution_quality',
            version='1.0',
            scope='study',
            permitted_hooks=('study.after_execute',),
            default_hook='study.after_execute',
            evaluator_id='computational_study_agent.gates.execution_quality',
            required_evidence=(_evidence('study_report', 'study_report', optional=True),),
            always_select=True,
            priority=20,
            description='Expose unresolved refined cases with case-specific recovery actions.',
        ),
        GateContract(
            gate_id='study.continuation_approval',
            version='1.0',
            scope='study',
            permitted_hooks=('study.after_review',),
            default_hook='study.after_review',
            evaluator_id='computational_study_agent.gates.continuation_approval',
            required_evidence=(_evidence('study_report', 'study_report', optional=True),),
            always_select=True,
            priority=10,
            description='Require approval before reusing neighboring one-particle states across a path.',
        ),
        GateContract(
            gate_id='study.path_consistency',
            version='1.0',
            scope='study',
            permitted_hooks=('study.after_review',),
            default_hook='study.after_review',
            evaluator_id='computational_study_agent.gates.path_consistency',
            required_evidence=(_evidence('study_report', 'study_report', optional=True),),
            always_select=True,
            priority=20,
            description='Review cross-case continuity only when study-level result analysis was requested.',
        ),
    ]


__all__ = ['build_default_study_gate_contracts', 'build_default_task_gate_contracts']
