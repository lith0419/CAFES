"""Compatibility import; implementation lives in computational_study_agent.datasets.hamiltonian.planner."""

import sys
from importlib import import_module

_implementation = import_module('computational_study_agent.datasets.hamiltonian.planner')
sys.modules[__name__] = _implementation
