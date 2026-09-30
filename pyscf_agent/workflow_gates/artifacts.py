from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping

from .runtime import gate_execution_trace_payload


def gate_artifact_documents(context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return versioned gate documents ready for an artifact writer."""

    configuration = context.get('gate_configuration')
    configuration = copy.deepcopy(configuration) if isinstance(configuration, Mapping) else {}
    provenance = context.get('gate_provenance')
    if not isinstance(provenance, Mapping):
        provenance = configuration.get('provenance') if isinstance(configuration, dict) else {}
    provenance = copy.deepcopy(provenance) if isinstance(provenance, Mapping) else {}
    documents: List[Dict[str, Any]] = []
    if configuration:
        documents.append({
            'kind': 'gate-configuration',
            'filename': 'gate-configuration.json',
            'payload': configuration,
            'description': 'Deterministic quality gates compiled around the workflow.',
        })
    if provenance:
        documents.append({
            'kind': 'gate-provenance',
            'filename': 'gate-provenance.json',
            'payload': provenance,
            'description': 'Quality-gate selection, configuration, and contract provenance.',
        })
    if configuration:
        documents.append({
            'kind': 'gate-execution-trace',
            'filename': 'gate-execution-trace.json',
            'payload': gate_execution_trace_payload(context),
            'description': 'Ordered quality-gate evaluations and deterministic decisions.',
        })
    return documents


__all__ = ['gate_artifact_documents']
