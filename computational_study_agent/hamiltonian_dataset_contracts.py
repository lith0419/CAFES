"""Compatibility import; implementation lives in computational_study_agent.datasets.hamiltonian.contracts."""

import sys
from importlib import import_module

_implementation = import_module('computational_study_agent.datasets.hamiltonian.contracts')
sys.modules[__name__] = _implementation
